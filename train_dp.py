import os
import time
import argparse
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader

from tqdm import tqdm
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from mlsfs.utils import logging_utils
logging_utils.config_logger()

from mlsfs.utils.data_loader import GetDataset
from mlsfs.utils.YParams import YParams
from mlsfs.utils.grids import GridQuadrature
from mlsfs.utils.losses import l2loss_sphere
from mlsfs.utils.weighted_acc_rmse import weighted_acc, weighted_rmse, weighted_rmse_torch, unlog_tp_torch
from mlsfs.models import SFNO

class Trainer:

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def __init__(self, params):
        # set seed
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.params = params

        self.train_dataset = GetDataset(params, params.train_data_path, train=True)
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size = params.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

        logging.info(f'Length of train_data_loader {len(self.train_data_loader.dataset)}')
        self.valid_dataset = GetDataset(params, params.valid_data_path, train=False)
        self.valid_data_loader = DataLoader(
            self.valid_dataset,
            batch_size = params.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

        nlat = self.train_dataset.img_shape_x
        nlon = self.train_dataset.img_shape_y

        self.quadrature = GridQuadrature( quadrature_rule = "legendre-gauss",
            img_shape=(nlat, nlon),
            normalize=True, 
        ).to(self.device)

        if params.nettype == 'sfno':
            model = SFNO(
                filter_type = params.filter_type,
                img_size = (nlat, nlon),
                in_chans = params.n_in_channels, 
                out_chans = params.n_out_channels,
                embed_dim = params.embed_dim,
                num_layers = params.num_layers,
                mlp_mode = params.mlp_mode,
            )
        else:
            raise Exception("Model {params.nettype} is not implemented")

        if torch.cuda.device_count() > 1:
            logging.info(f'Using {torch.cuda.device_count()} GPUs!')
            self.model = nn.DataParallel(model).to(self.device)

        self.loss_obj = nn.MSELoss().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        
        self.startEpoch = 0
        self.max_train_batch = 20
        self.max_valid_batch = 4

        
        if params.two_step_training:
            #if params.resuming == False and params.pretrained == True:
            if params.pretrained == True:
                logging.info("Starting from pretrained one-step sfno model at %s"%params.pretrained_ckpt_path)
                self.restore_checkpoint(params.pretrained_ckpt_path)
                self.iters = 0
                self.startEpoch = 0
        
        self.epoch = self.startEpoch
        # print(len(train_dataset))
        #self.solver = SphereSolver(nlat, nlon, params.dt).to(self.device).float()


        if params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif params.scheduler == 'CosineAnnealingLR':
            if not hasattr(params, "scheduler_min_lr"):
                params["scheduler_min_lr"] = 0.0
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.scheduler_T_max, eta_min=params.scheduler_min_lr)
        elif params.scheduler == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=params.scheduler_step_size, gamma=params.scheduler_gamma)
        else:
            self.scheduler = None

        if params.log_to_screen:
            logging.info(f"Number of trainable model parameters: {self.count_parameters()}")

    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.e6
        for epoch in tqdm(range(self.startEpoch, self.params.max_epochs), desc='Epoch ', leave=True):

            self.epoch += 1

            start = time.time()

            acc_train_loss = 0
            acc_valid_loss = 0

            self.model.train()
            train_steps = 0
            #for i, data in enumerate(tqdm(self.train_data_loader, desc="Training progress ", disable=not self.params.log_to_screen)):
            for i, data in enumerate(self.train_data_loader):

                #if i >= self.max_train_batch:
                #    break

                data_start = time.time()
                
                inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
                #inp, tar = inp.to(self.device, dtype=torch.float), tar.to(self.device, dtype=torch.float)


                self.optimizer.zero_grad()

                tr_start = time.time()

                if self.params.two_step_training:

                    forcing = inp[:, self.params.n_out_channels:]

                    gen_step_one = self.model(inp).to(self.device, dtype=torch.float)
                    loss_step_one = l2loss_sphere(self.quadrature, gen_step_one, tar[:, 0:self.params.n_out_channels], self.params.nhw.to(self.device))
                   

                    gen_step_two = self.model(torch.cat((gen_step_one, forcing), axis=1)).to(self.device, dtype = torch.float)
                    loss_step_two = l2loss_sphere(self.quadrature, gen_step_two, tar[:, self.params.n_out_channels:2*self.params.n_out_channels], self.params.nhw.to(self.device))
                    #loss = loss_step_one + loss_step_two
                    loss = loss_step_one + loss_step_two
                else:
                    gen = self.model(inp) #.to(self.device, dtype=torch.float)
                    loss = l2loss_sphere(self.quadrature, gen, tar[:, 0:self.params.n_out_channels], self.params.nhw.to(self.device))


                acc_train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            nbatches1 = i
            #logging.info(f"Epoch {epoch}, Training Loss: {acc_train_loss/(nbatches1*self.params.batch_size)}") 
            acc_train_loss = acc_train_loss/(len(self.train_data_loader))


            self.model.eval()

            if epoch%2 == 0:
                with torch.no_grad():
                    for i, data in enumerate(tqdm(self.valid_data_loader, desc="Validation progress ", disable=not self.params.log_to_screen)):
                        #logging.info(f'valid on batch {i}')
                        #if i >= self.max_valid_batch:
                        #    break
                        inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)


                        if self.params.two_step_training:
                            forcing = inp[:, self.params.n_out_channels:]

                            gen_step_one = self.model(inp.to(self.device, dtype=torch.float))
                            loss_step_one = l2loss_sphere(self.quadrature, gen_step_one, tar[:, 0:self.params.n_out_channels], self.params.nhw.to(self.device))

                            gen_step_two = self.model(torch.cat((gen_step_one, forcing), axis=1)).to(self.device, dtype=torch.float)
                            loss_step_two = l2loss_sphere(self.quadrature, gen_step_two, tar[:, self.params.n_out_channels:2*self.params.n_out_channels], self.params.nhw.to(self.device))
                            valid_loss = loss_step_one + loss_step_two
                        else:
                            gen = self.model(inp.to(self.device, dtype = torch.float))
                            valid_loss = l2loss_sphere(self.quadrature, gen, tar[:, 0:self.params.n_out_channels], self.params.nhw.to(self.device))

                        acc_valid_loss += valid_loss.item()

                nbatches2 = i
                acc_valid_loss = acc_valid_loss/(len(self.valid_data_loader))


                if self.params.save_checkpoint:
                    self.save_checkpoint(self.params.checkpoint_path)
                    if acc_valid_loss <= best_valid_loss:
                        logging.info(f"Val loss improved from {best_valid_loss} to {acc_valid_loss}, lr {self.get_lr()}")
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = acc_valid_loss

                if self.params.log_to_screen:
                    logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                    logging.info(f"Epoch {epoch+1} Accumulated training loss: {acc_train_loss} validation loss: {acc_valid_loss}, lr: {self.get_lr()}")


    def save_checkpoint(self, checkpoint_path, model=None):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """

        if not model:
            model = self.model

        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)    

    def restore_checkpoint(self, checkpoint_path):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
        #map_location = {"cuda:0": "cuda:{}".format(self.params.local_rank)}
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val 
            self.model.load_state_dict(new_state_dict)
        self.iters = checkpoint['iters']
        print(self.iters)
        self.startEpoch = checkpoint['epoch']
        print(self.startEpoch)
        if self.params.resuming:  #restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/SFNO.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--epsilon_factor", default = 0, type = float)

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['epsilon_factor'] = args.epsilon_factor
    params['enable_amp'] = args.enable_amp
    params['resuming'] = False

    # set up output directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    params['experiment_dir'] = os.path.abspath(expDir)

    cpt_path = os.path.join(expDir, 'training_checkpoints')
    if not os.path.exists(cpt_path):
        os.makedirs(cpt_path)

    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

    params['n_out_channels'] = params.n_out_channels
    params['out_channels'] = np.arange(params.n_out_channels)
    
    params['n_in_channels'] = params.n_out_channels + params.n_forcing_channels

    nstatic = 0
    if params.orography:
        logging.info(f'Adding orography, in_channels add 1')
        params.n_in_channels += 1
        nstatic += 1

    if params.lsmask:
        logging.info(f'Adding lsm, in_channels add 1')
        params.n_in_channels += 1
        nstatic += 1

    if params.lakemask:
        logging.info(f'Adding lake, in_channels add 1')
        params.n_in_channels += 1
        nstatic += 1
     
    params['nstatic'] = nstatic
    params['in_channels'] = np.arange(params.n_in_channels)
    logging.info(f"n_in_channels: {params['n_in_channels']}")
    logging.info(f"n_out_channels: {params['n_out_channels']}")

    #channel_names = ['u10', 'v10', 't2m', 'msl', 'tp', 'cloud', 'lh', 'sh', 'ice', 'sst']
    channel_names = ['u10', 'v10', 't2m', 'msl']
    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    variables = ['u', 'v', 'w', 't', 'q', 'z']
    for var in variables:
        for lev in levels:
            channel_names.append(f'{var}{lev}')

    channel_weights = torch.ones(params.n_out_channels, dtype=torch.float32)
    for c, chn in enumerate(channel_names):
        #if chn in ['u10', 'v10', 'msl', 'tp', 'cloud', 'lh', 'sh', 'ice', 'sst']:
        if chn in ['u10', 'v10', 'msl']:
            channel_weights[c] = 0.1
        elif chn in ['t2m']:
            channel_weights[c] = 1.0
        else:
            pressure_level = float(chn[1:])
            channel_weights[c] = 0.001 * pressure_level

    #remormalize the weights to one
    channel_weights = channel_weights.reshape(1, -1)
    channel_weights = channel_weights / torch.sum(channel_weights)
    params['nhw'] = channel_weights
    # print(params['train_data_path'])
    # print(params.dt)
    # print(params.n_history)
    # print(params.lr)
    # print(params.two_step_training)
    # print(params.enable_amp)
    # print(params.crop_size_x)
    # print(params.crop_size_y)
    # print(params.resuming)
    # print(f'dist is {dist.is_initialized()}')

    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
    params.log()

    hparams = ruamelDict()
    yaml = YAML()
    for key, value in params.params.items():
        hparams[str(key)] = str(value)
    with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
        yaml.dump(hparams, hpfile)

    trainer = Trainer(params)
    trainer.train()
    logging.info('DONE ----')

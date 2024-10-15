import os
import time
import argparse
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
#from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torchvision.utils import save_image
from tqdm import tqdm
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from mlsfs.utils import logging_utils
logging_utils.config_logger()

#from utils.data_loader2 import get_data_loader
from mlsfs.utils.data_loader import get_data_loader
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

    def __init__(self, params, world_rank):
        # set seed
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        self.params = params
        self.world_rank = world_rank
        # set device
        #self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.device = torch.device("cuda:{}".format(params.local_rank))
        #n = torch.cuda.device_count() // params.world_size
        n = 1
        device_ids = list(range(world_rank * n, (world_rank + 1) * n))

        #device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        # self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # if  torch.cuda.is_available():
        #     torch.cuda.set_device(self.device.index)

        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, params.train_data_path, dist.is_initialized(), train=True, world_size=params.world_size, rank=world_rank)
        
        self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.valid_data_path, dist.is_initialized(), train=False, world_size=params.world_size, rank=world_rank)
        
        nlat = self.train_dataset.img_shape_x
        nlon = self.train_dataset.img_shape_y
        
        # print(len(train_dataset))
        #self.solver = SphereSolver(nlat, nlon, params.dt).to(self.device).float()
        self.quadrature = GridQuadrature(
            quadrature_rule = "legendre-gauss",
            img_shape=(nlat, nlon),
            normalize=True, 
        ).to(self.device)

        if params.nettype == 'sfno':
            self.model = SFNO(
                filter_type = params.filter_type,
                img_size = (nlat, nlon),
                in_chans = params.n_in_channels, 
                out_chans = params.n_out_channels,
                embed_dim = params.embed_dim,
                num_layers = params.num_layers,
                mlp_mode = params.mlp_mode,
            ).to(self.device)
        else:
            raise Exception("Model {params.nettype} is not implemented")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)

        if params.enable_amp:
            self.gscaler = amp.GradScaler()


        #Move model to GPUs
        if dist.is_initialized():
            self.model = DistributedDataParallel(
                self.model,
                device_ids = [params.local_rank],
                output_device = params.local_rank,
                #find_unused_parameters = True,
            )
            #self.model = FullyShardedDataParallel(
            #    model(),
            #    fsdp_auto_wrap_policy=default_auto_wrap_policy,
            #    cpu_offload=CPUOffload(offload_params=True),
            #)

        self.iters = 0
        self.startEpoch = 0
        if params.two_step_training:
            if params.resuming == False and params.pretrained == True:
                logging.info("Starting from pretrained one-step afno model at %s"%params.pretrained_ckpt_path)
                self.restore_checkpoint(params.pretrained_ckpt_path)
                self.iters = 0
                self.startEpoch = 0


        self.epoch = self.startEpoch

        if params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.max_epochs)
        else:
            self.scheduler = None

        if params.log_to_screen:
            logging.info(f"Number of trainable model parameters: {self.count_parameters()}")

    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)

                #turn on to valid only on the frist batch
                #self.valid_sampler.set_epoch(epoch)

            start = time.time()
            tr_time, train_logs = self.train_one_epoch()
            valid_time, valid_logs = self.validate_one_epoch()
            #if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct':
            #    valid_weighted_rmse = self.validate_final()

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
                if self.epoch >= self.params.max_epochs:
                    logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    exit()
            else:
                if self.epoch >= self.params.max_epochs:
                    logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                    exit()


            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    self.save_checkpoint(self.params.checkpoint_path)
                    if valid_logs['valid_loss'] <= best_valid_loss:
                        logging.info(f"Val loss improved from {best_valid_loss} to {valid_logs['valid_loss']}, lr {self.get_lr()}")
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_logs['valid_loss']

            if self.params.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                #logging.info('train data time={}, train step time={}, valid step time={}'.format(data_time, tr_time, valid_time))
                logging.info(f"Accumulated training loss: {train_logs['loss']} validation loss: {valid_logs['valid_loss']} lr: {self.get_lr()}")

    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0

        #train_loss = 0
        self.model.train()

        #max_train_batch = 10

        for i, data in enumerate(self.train_data_loader, 0):
            #logging.info(f'training on batch {i}')
            #if i >= max_train_batch:
            #    break
            self.iters += 1
            data_start = time.time()
            
            inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
            #inp, tar = inp.to(self.device, dtype=torch.float), tar.to(self.device, dtype=torch.float)
        
            tr_start = time.time()

            self.model.zero_grad()
            with amp.autocast(self.params.enable_amp):
            #with torch.amp.autocast('cuda'):
                if self.params.two_step_training:
                    gen_step_one = self.model(inp) #.to(self.device, dtype=torch.float)
                    loss_step_one = l2loss_sphere(self.quadrature, gen_step_one, tar[:, 0:self.params.n_out_channels], self.params.nhw.to(self.device))
                    ##Temporarily disable step two because of memory issue (lcui)
                    #gen_step_two = self.model(gen_step_one) #.to(self.device, dtype=torch.float)
                    #loss_step_two = spectral_l2loss_sphere(self.solver, gen_step_two, tar[:, self.params.n_out_channels:2*self.params.n_out_channels], relative=False)
                    #loss = loss_step_one + loss_step_two
                    loss = loss_step_one
                else:
                    gen = self.model(inp) #.to(self.device, dtype=torch.float)
                    loss = l2loss_sphere(self.quadrature, gen, tar[:, 0:self.params.n_out_channels], self.params.nhw.to(self.device))
            

            #train_loss += (loss.item()/self.params.world_size)*inp.size[0]

            if self.params.enable_amp:
                self.gscaler.scale(loss).backward()
                self.gscaler.step(self.optimizer)
            else:
                loss.backward()
                self.optimizer.step()

            if self.params.enable_amp:
                self.gscaler.update()

            #logging.info('Time taken for training batch {} on rank {} is {} sec'.format(i + 1, self.world_rank, time.time()-tr_start))
            tr_time += time.time() - tr_start
            #break  #lcui- exist the loop after the first interation

        #if dist.is_initialized():
        #    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        try:
            #logs = {'loss': loss, 'loss_step_one': loss_step_one, 'loss_step_two': loss_step_two}
            logs = {'loss': loss, 'loss_step_one': loss_step_one, }
        except:
            logs = {'loss': loss}

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key]/dist.get_world_size())


        # #train_loss = train_loss / len(self.train_data_loader)
        # logging.info(f'training loss on rank {self.world_rank} before: {train_loss}')
        # train_loss = train_loss / max_train_batch
        # logging.info(f'training loss on rank {self.world_rank} after: {train_loss}')

        return tr_time, logs

    def validate_one_epoch(self):
        #max_valid_batch = 2 #do validation on first 8 batches

        if self.params.normalization == 'minmax':
            raise Exception("minmax normalization not supported")
        elif self.params.normalization == 'zscore':
            mult = torch.as_tensor(np.load(self.params.global_stds_path)[0, self.params.out_channels, 0, 0]).to(self.device)

        valid_buff = torch.zeros((3), dtype=torch.float32, device=self.device)
        valid_loss = valid_buff[0].view(-1)
        valid_l1 = valid_buff[1].view(-1)
        valid_steps = valid_buff[2].view(-1)
        valid_weighted_rmse = torch.zeros((self.params.n_out_channels), dtype=torch.float32, device=self.device)
        valid_weighted_acc = torch.zeros((self.params.n_out_channels), dtype=torch.float32, device=self.device)


        valid_start = time.time()
        #valid_loss = 0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader, 0):
                #logging.info(f'valid on batch {i}')
                #if i >= max_valid_batch:
                #    break
                inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)

                if self.params.two_step_training:
                    gen_step_one = self.model(inp.to(self.device, dtype=torch.float))
                    loss_step_one = l2loss_sphere(self.quadrature, gen_step_one, tar[:, 0:self.params.n_out_channels], self.params.nhw.to(self.device))

                    ##Temporarily disable step two because of memory issue (lcui)
                    #gen_step_two = self.model(gen_step_one.to(self.device, dtype=torch.float))
                    #loss_step_two = spectral_l2loss_sphere(self.solver, gen_step_two, tar[:, self.params.n_out_channels:2*self.params.n_out_channels], relative=False)
                    #valid_loss += loss_step_one + loss_step_two
                    valid_loss += loss_step_one
                else:
                    gen = self.model(inp.to(self.device, dtype = torch.float))
                    valid_loss += l2loss_sphere(self.quadrature, gen, tar[:, 0:self.params.n_out_channels], self.params.nhw.to(self.device))

                valid_steps += 1.

                 #direct prediction weighted rmse
                if self.params.two_step_training:
                    valid_weighted_rmse += weighted_rmse_torch(gen_step_one, tar[:,0:self.params.n_out_channels])
                else:
                    valid_weighted_rmse += weighted_rmse_torch(gen, tar[:,0:self.params.n_out_channels])

                # if dist.is_initialized():
                #     dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                
                # valid_loss += loss.item()/self.params.world_size

                #break # breaking the loop after the first iteration to use only the first batch for validation
                #valid_steps += 1.

                
        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(valid_weighted_rmse)

        # divide by number of steps
        valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
        valid_weighted_rmse = valid_weighted_rmse / valid_buff[2]
        valid_weighted_rmse *= mult

        # download buffers
        valid_buff_cpu = valid_buff.detach().cpu().numpy()
        valid_weighted_rmse_cpu = valid_weighted_rmse.detach().cpu().numpy()

        valid_time = time.time() - valid_start
        valid_weighted_rmse = mult*torch.mean(valid_weighted_rmse, axis = 0)
        try:
            logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'valid_rmse_u10': valid_weighted_rmse_cpu[0], 'valid_rmse_v10': valid_weighted_rmse_cpu[1]}
        except:
            logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'valid_rmse_u10': valid_weighted_rmse_cpu[0]}#, 'valid_rmse_v10': valid_weighted_rmse[1]}
        
        return valid_time, logs    

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
        map_location = {"cuda:0": "cuda:{}".format(self.params.local_rank)}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
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

    # set up parallel environment
   # params['world_size'] = 8
    if 'WORLD_SIZE' in os.environ:
        print(f"WORLD_SIZE is {os.environ['WORLD_SIZE']}")
        params['world_size'] = int(os.environ['WORLD_SIZE'])
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        WORLD_RANK = int(os.environ["RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        # Environment variables set by mpirun
        LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        WORLD_SIZE = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        WORLD_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])
    else:
        import sys

        sys.exit("Can't find the evironment variables for local rank")

    params['world_size'] = WORLD_SIZE

    #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    world_rank = 0
    local_rank = 0
    if params['world_size'] > 1:
        dist.init_process_group(
            backend='nccl', rank=WORLD_RANK, world_size=WORLD_SIZE
        )
        #local_rank = int(os.environ["LOCAL_RANK"])
        local_rank = LOCAL_RANK
        args.gpu = local_rank
        world_rank = dist.get_rank()
        params['global_batch_size'] = params.batch_size
        params['batch_size'] = int(params.batch_size//params['world_size'])
        params['local_rank'] = local_rank

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    # set up output directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    params['experiment_dir'] = os.path.abspath(expDir)

    cpt_path = os.path.join(expDir, 'training_checkpoints')
    if not os.path.exists(cpt_path):
        os.makedirs(cpt_path)

    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

    params['out_channels'] = np.arange(params.num_channels-1) #exclde tisr
    params['n_out_channels'] = params.num_channels-1 #exclude tisr
    
    if params.orography:
        logging.info(f'Adding orography, in_channels add 1')
        params.num_channels += 1

    if params.lsmask:
        logging.info(f'Adding lsm, in_channels add 1')
        params.num_channels += 1

    if params.lakemask:
        logging.info(f'Adding lake, in_channels add 1')
        params.num_channels += 1

    params['in_channels'] = np.arange(params.num_channels)
    params['n_in_channels'] = params.num_channels
    logging.info(f"n_in_channels: {params['n_in_channels']}")
    logging.info(f"n_out_channels: {params['n_out_channels']}")

    channel_names = ['u10', 'v10', 't2m', 'msl', 'tp', 'cloud', 'lh', 'sh', 'ice', 'sst']
    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    variables = ['u', 'v', 'w', 't', 'q', 'z']
    for var in variables:
        for lev in levels:
            channel_names.append(f'{var}{lev}')

    channel_weights = torch.ones(params.n_out_channels, dtype=torch.float32)
    for c, chn in enumerate(channel_names):
        if chn in ['u10', 'v10', 'msl', 'tp', 'cloud', 'lh', 'sh', 'ice', 'sst']:
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

    if world_rank == 0:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        params.log()

        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    trainer = Trainer(params, world_rank)
    trainer.train()
    logging.info('DONE ---- rank %d'%world_rank)

    #torch.autograd.set_detect_anomaly(True)
    dist.destroy_process_group()

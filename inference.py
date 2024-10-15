import argparse
import logging
import glob
import os
from datetime import datetime

import numpy as np
import xarray as xr
import torch

from mlsfs.models import SFNO

from gdas import get_input

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
class MLSFS:
    def __init__(self, start_time, assets, inputs, outputs, leading_time=240):
        self.start_time = start_time
        self.assets = assets
        self.inputs = inputs
        self.outputs = outputs
        self.leading_time = leading_time
        self.backbone_channels = 88

        #lats = np.arange(90, -90.25, -0.25)
        #self.latitude = DimCoord(lats, standard_name='latitude', units='degrees')
        #lons = np.arange(0, 360, 0.25)
        #self.longitude = DimCoord(lons, standard_name='longitude', units='degrees')

        #ds1 = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr')
        #ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr')
    
    def load_statistics(self):
        path = os.path.join('./stats', "global_mean_1979-2016_reordered.npy")
        self.means = np.load(path)
        self.means = self.means[0, :self.backbone_channels, ...].astype(np.float32)
    
        path = os.path.join('./stats', "global_std_1979-2016_reordered.npy")
        self.stds = np.load(path)
        self.stds = self.stds[0, :self.backbone_channels, ...].astype(np.float32)
    
    
    def normalise(self, data, reverse=False):
        if reverse:
            new_data = data * self.stds + self.means
        else:
            new_data = (data - self.means) / self.stds 
    
        return new_data   
    
    def load_model(self, checkpoint_file):
        model = SFNO(
            filter_type = 'linear',
            img_size=(121, 240),
            in_chans=self.backbone_channels+3,
            out_chans=self.backbone_channels,
            embed_dim = 48,
            num_layers = 7,
            mlp_mode = 'serial',

        )
            
        model.zero_grad()
    
        checkpoint = torch.load(checkpoint_file, map_location=device)
        weights = checkpoint["model_state"]
        drop_vars = ["module.norm.weight", "module.norm.bias"]
        weights = {k: v for k, v in weights.items() if k not in drop_vars}
    
        try:
            # Try adding model weights as dictionary
            new_state_dict = dict()
            for k, v in checkpoint["model_state"].items():
                name = k[7:]
                if name != "ged":
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        except Exception:
            model.load_state_dict(checkpoint["model_state"])
    
        # Set model to eval mode and return
        model.eval()
        model.to(device)
    
        return model

    def run(self):
        self.load_statistics()
       
        #read inputs
        with open(self.inputs, 'rb') as f: 
            data = np.load(f)
            
        all_fields_numpy = data.astype(np.float32)



        #input_iter = self.inputs.to(device)
        input_iter = torch.from_numpy(all_fields_numpy).to(device)
        forcing = input_iter[:,88:]

        torch.set_grad_enabled(False)

        path = os.path.join(self.assets, "best_ckpt.tar")
        model = self.load_model(path)
        
        for i in range(self.leading_time // 6):
            print(f'Starting inference for step {i} ')
            
            output = model(input_iter)

            input_iter = torch.cat((output, forcing), dim=1)

            step = (i + 1) * 6
            output = self.normalise(output.cpu().numpy(), reverse=True)
        
            np.save(f'{self.outputs}/output_step{i:02d}.npy', output)
            #self.write(output, step)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("start_datetime", help="Start datetime in the format 'YYYYMMDDHH'")

    parser.add_argument("-w", "--weights", help="parent directory of the weights and stats", required=True)
    parser.add_argument("-i", "--input", help="input file path (including file name)", required=True)
    parser.add_argument("-o", "--output", help="output directory", default=None)
    parser.add_argument("-l", "--length", type=int, help="total hours to forecast", required=True)

    args = parser.parse_args()

    #start_idx = 0
    #zarr_file = "/lustre/Linlin.Cui/s2s/data/out_of_sample/era5_2021_6h-240x121.zarr"
    #stats_path = "/nvidia/sfs/stats"
    #static_path = "/lustre/Linlin.Cui/s2s/data/static/"

    #img = get_input(start_idx, zarr_file, orography=True, lsmask=True, static=static_path, stats=stats_path)

    start_datetime = datetime.strptime(args.start_datetime, "%Y%m%d%H")
    fcn = MLSFS(start_datetime, args.weights, args.input, args.output, args.length)
    fcn.run()

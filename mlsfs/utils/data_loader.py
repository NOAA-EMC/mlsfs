import logging
import glob

import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from mlsfs.utils.img_utils import reshape_fields

def get_data_loader(params, files_pattern, distributed, train, world_size, rank):
    dataset = GetDataset(params, files_pattern, train)
    logging.info(f'distributed is {distributed}')
    sampler = DistributedSampler(dataset, shuffle=train, num_replicas=world_size, rank=rank) if distributed else None

    dataloader = DataLoader(
        dataset,
        batch_size=int(params.batch_size),
        num_workers=params.num_data_workers,
        shuffle=False, #(sampler is None),
        sampler=sampler if train else None,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset



class GetDataset(Dataset):
    def __init__(self, params, location, train):
        self.params = params
        self.location = location
        self.train = train
        self.dt = 1
        self.n_history = 0
        #ds1 = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr')
        #ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr')
        self.attrs = {
            'surface': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', 
                '2m_temperature', 'mean_sea_level_pressure',
                'total_precipitation_6hr', 'total_cloud_cover', 
                'mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux',
                'sea_ice_cover', 'sea_surface_temperature',
            ],
            'pressure_level': [
                'u_component_of_wind', 'v_component_of_wind', 
                'vertical_velocity', 'temperature', 'specific_humidity', 'geopotential'
            ],
            'forcing': ['toa_incident_solar_radiation'],
        }
        self.levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

        self.orography = params.orography
        self.lsmask = params.lsmask
        self.lakemask = params.lakemask
        self.normalize = params.normalize

        self._get_files_stats()

    def _get_files_stats(self):
        self.files_paths = glob.glob(f'{self.location}/*.zarr')
        self.files_paths.sort()
        self.n_years = len(self.files_paths)

        logging.info(f'Getting file stats from {self.files_paths[0]}')
        ds = xr.open_zarr(self.files_paths[0])
        self.n_samples_per_year = ds.time.shape[0]
        self.img_shape_x = ds['latitude'].shape[0]
        self.img_shape_y = ds['longitude'].shape[0]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]
        logging.info(f'Number of samples per year: {self.n_samples_per_year}')

        ds.close()

    def _open_files(self, year_idx):
        self.files[year_idx] = xr.open_zarr(self.files_paths[year_idx])

        if self.orography:
            self.orog = np.load(self.params['orography_file'])[:,::-1] #reverse latitude to [90, -90]
        else:
            self.orog = None

        if self.lsmask:
            self.lsm = np.load(self.params['lsm_file'])[:,::-1] #reverse latitude to [90, -90]
        else:
            self.lsm = None

        if self.lakemask:
            self.lake = np.load(self.params['lake_file'])[:,::-1] #reverse latitude to [90, -90]
        else:
            self.lake = None

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx = int(global_idx / self.n_samples_per_year)
        local_idx = int(global_idx % self.n_samples_per_year)

        if self.files[year_idx] is None:
            self._open_files(year_idx)

        #if local_idx < self.dt * self.n_history:
        #    local_idx += self.dt * self.n_history

        #step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt 
        if local_idx >= self.n_samples_per_year - self.dt:
            local_idx = self.n_samples_per_year - 2 

        step = int(self.dt)

        #logging.info(f'year_idx is {year_idx}, local_idx is {local_idx}')

        data = []
        for key, variables in self.attrs.items():
            for var in variables:
                if key == 'surface' or key == 'forcing':
                    values = self.files[year_idx][var].isel(time=[local_idx, local_idx+step]).transpose('time', 'latitude', 'longitude').values[:,::-1,:] #reverse latitude

                    # check nan
                    if np.sum(np.isnan(values)) > 1:
                        data.append(np.nan_to_num(values, nan=-99.))
                    else:
                        data.append(values)

                elif key == 'pressure_level':
                    values = self.files[year_idx][var].isel(time=[local_idx, local_idx+step]).transpose('time', 'level', 'latitude', 'longitude').sel(level=self.levels).values
                    for ilev in np.arange(len(self.levels)):
                        #if np.sum(np.isnan(values[:, ilev, :, :])) > 1:
                        #    data.append(np.nan_to_num(values[:, ilev, ::-1, :], nan=0.0))
                        #else:
                        data.append(values[:, ilev, ::-1, :]) #reverse latitude

                else:
                    raise valueError(f'{key} is not in ["surface", "pressure_level", "forcing"]')
        
        data = np.array(data)

        return reshape_fields(np.squeeze(data[:,0,:,:]), 'inp', self.params, self.normalize, self.orog, self.lsm, self.lake), \
            reshape_fields(np.squeeze(data[:,1,:,:]), 'tar', self.params, self.normalize, self.orog, self.lsm, self.lake)

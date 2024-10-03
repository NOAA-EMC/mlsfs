from time import time

import xarray as xr
import dask
import gcsfs

vars1 = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', 
    '2m_temperature', 'mean_sea_level_pressure',
    'total_precipitation_6hr', 'sea_ice_cover', 'total_cloud_cover', 'sea_surface_temperature',
    'mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux', 
    #'mean_surface_net_long_wave_radiation_flux', 'mean_surface_net_short_wave_radiation_flux'
    'u_component_of_wind', 'v_component_of_wind', 
    'vertical_velocity', 'temperature', 'specific_humidity', 'geopotential'
]
vars2 = ['toa_incident_solar_radiation']

levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


zarr_path1 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
zarr_path2 = 'gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr'

local_zarr_path = '/lustre/Linlin.Cui/s2s/data'

ds1 = xr.open_zarr(zarr_path1)
ds1 = ds1[vars1]
#for var in ds1.variables:
#    print(f'{var}, encoding: {ds1[var].encoding}')
#    ds1[var].encoding.clear()
#
ds2 = xr.open_zarr(zarr_path2)
ds2 = ds2[vars2]
#for var in ds2.variables:
#    print(f'{var}, encoding: {ds2[var].encoding}')
#    ds2[var].encoding.clear()

data_dict = {
    "train": range(1979, 2017),
    "test": range(2017, 2021),
    "out-of-sample": range(2021, 2023),
}

for k, v in data_dict.items():
    outdir = f'{local_zarr_path}/{k}'
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parent=)

    for year in v:
        print(year)
        fname = f'{outdir}/era5_{year}_6h-240x121.zarr'
    
        t0 = time()
    
        yearly_data1 = ds1.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        with dask.config.set(scheduler='single-threaded'):
            yearly_data1.to_zarr(fname, safe_chunks=False, mode='w')
    
        yearly_data2 = ds2.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        with dask.config.set(scheduler='single-threaded'):
            yearly_data2.to_zarr(fname, safe_chunks=False, mode='a')
    
        print(f'Time: {time()-t0} sec')
    
        del yearly_data1, yearly_data2

ds1.close()
ds2.close()

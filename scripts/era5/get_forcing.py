import numpy as np
import xarray as xr
import dask

variables = [
    'geopotential_at_surface', 'land_sea_mask', 'lake_cover'
]

outdir = '/lustre/Linlin.Cui/mlsfs/data/static'
ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr')
for var in variables:
    print(var)
    values = ds[var].transpose().values[::-1,:] #transpose to lat-lon, and reverse the lat to [90, -90]
    if var == 'geopotential_at_surface':
        #values /= 9.80665
        mean = values.mean()
        std = values.std()
        values_norm = (values - mean) / std
        np.save(f'{outdir}/orog_latlon_ns_normalized.npy', values_norm)
    else:
        np.save(f'{outdir}/{var}_latlon_ns.npy', values)

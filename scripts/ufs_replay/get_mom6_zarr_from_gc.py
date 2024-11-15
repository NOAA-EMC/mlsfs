from time import time
import pathlib

import xarray as xr
import dask
import gcsfs

zarr_path = 'gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/06h-freq/zarr/mom6.zarr'
local_zarr_path = '/lustre/Linlin.Cui/mlsfs/data/UFS/noaa-ufs-gefsv13replay-pds/ocean'

ds = xr.open_zarr(zarr_path)
ds = ds.temp.isel(z_l=0)
ds2 = ds[{'lat': slice(None, None, 6), 'lon': slice(None, None, 6)}]
ds2 = ds2.rename({
    'lat': 'latitude',
    'lon': 'longitude',
})
unused_coords = set(ds2.coords) - set(ds2.dims)
ds2 = ds2.drop_vars(unused_coords)

data_dict = {
    "train": range(1994, 2017),
    "test": range(2017, 2019),
    "out_of_sample": range(2019, 2023),
}

for k, v in data_dict.items():
    outdir = f'{local_zarr_path}/{k}'
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for year in v:
        print(year)
        fname = f'{outdir}/ufs_ocn_{year}_6h-128x256.zarr'
    
        t0 = time()
    
        yearly_data = ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        with dask.config.set(scheduler='single-threaded'):
            yearly_data.to_zarr(fname, safe_chunks=False, mode='w')
    
    
        print(f'Time: {time()-t0} sec')
    
        del yearly_data

ds.close()

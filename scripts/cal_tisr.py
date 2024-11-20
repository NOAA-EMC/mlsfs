from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from mlsfs.utils.tisr import insolation

datevectors = np.arange(datetime(1994, 1, 1), datetime(1995, 1, 1), timedelta(hours=6)).astype(datetime)
zarr_file = "/lustre/Linlin.Cui/mlsfs/data/UFS/train/ufs-replay_1994_6h-128x256.zarr"
ds = xr.open_zarr(zarr_file)
lat = ds.latitude[::-1].values
lon = ds.longitude.values

# with coefficient 0.5e7, the magnitude is similar to era5's tisr
toa = 0.5e7 * insolation(datevectors, lat, lon, enforce_2d=True)
ds_toa = xr.Dataset(
    data_vars={
        'tisr': (['time', 'lat', 'lon'], toa)
    },
    coords={
        'lon': lon,
        'lat': lat,
        'time': datevectors,
    },
)



averaged = ds_toa['tisr'].mean(dim=['lon'])

dates = [datetime.strptime('01-01-1994', '%m-%d-%Y') + timedelta(hours=(d+1)*6) for d in range (averaged.shape[0])]

plt.rcParams.update({
    'font.family': 'Nimbus Roman',
    'savefig.dpi': 300,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

fig = plt.figure(figsize=(6, 4), layout='constrained')
ax = fig.add_subplot(111)
#cxticks = np.arange(490, 600, 10)
cf = ax.contourf(dates, lat, averaged.transpose(), cmap=plt.cm.rainbow, extend='both')
cbar = plt.colorbar(cf, pad=0.01, shrink=0.8)
plt.xticks(rotation=45, ha='right')


plt.savefig(f'tisr_computed_ufs-1994.png')

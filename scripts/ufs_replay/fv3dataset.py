import os
import pathlib
from datetime import datetime, timedelta
import shutil

import numpy as np
import xarray as xr
import pygrib
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def get_dataarray(grbfile, var_name, level_type, desired_level):

    # Find the matching grib message
    variable_message = grbfile.select(shortName=var_name, typeOfLevel=level_type, level=desired_level)

    # create a netcdf dataset using the matching grib message
    lats, lons = variable_message[0].latlons()
    lats = lats[::6,0] #1.5 degree
    lons = lons[0,::6]

    #check latitude range, sfno takes [90, -90]
    reverse_lat = False
    if lats[0] < 0:
        reverse_lat = True
        lats = lats[::-1]

    steps = variable_message[0].validDate

    if len(variable_message) > 2:
        data = []
        for message in variable_message:
            data.append(message.values[::6, ::6])
        data = np.array(data)
        if reverse_lat:
            data = data[:, ::-1, :]
    else:
        data = variable_message[0].values[::6,::6]
        if reverse_lat:
            data = data[::-1, :]

    if len(data.shape) == 2:
        da = xr.DataArray(
            np.expand_dims(data, axis=0), 
            dims=['time', 'latitude', 'longitude'],
            coords={
                'longitude': lons.astype('float32'),
                'latitude': lats.astype('float32'),
                'time': np.array([steps]),  
            },
            name=var_name,
        )

    elif len(data.shape) == 3:

        da = xr.DataArray(
            np.expand_dims(data, axis=0),
            dims=['time', 'level', 'latitude', 'longitude'],
            coords={
                'longitude': lons.astype('float32'),
                'latitude': lats.astype('float32'),
                'level': np.array(desired_level).astype('int32'),
                'time': np.array([steps]),  
            },
            name=var_name,
        )

    return da

class FV3Dataset:
    def __init__(self, outdir, year, initilize=False):
        self.year = year
        self.initilize = initilize
        self.data_path = f'{outdir}/ufs-replay_{year}_6h-128x256.zarr'

        levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        self.vars_dict = {
            'gfsflx': {
                '10u': {
                    'typeOfLevel': 'heightAboveGround',
                    'level': 10,
                },
                '10v': {
                    'typeOfLevel': 'heightAboveGround',
                    'level': 10,
                },
                '2t': {
                    'typeOfLevel': 'heightAboveGround',
                    'level': 2,
                },
                'pwat': {  ## Total column water vapor, taken from GFS precipitable water
                    'typeOfLevel': 'atmosphereSingleLayer',
                    'level': 0,
                },
        
            },
            'gfsprs': {
                'prmsl': {
                    'typeOfLevel': 'meanSea',
                    'level': 0,
                },
                'u': {
                    'typeOfLevel': 'isobaricInhPa',
                    'level': levels,
                },
                'v': {
                    'typeOfLevel': 'isobaricInhPa',
                    'level': levels,
                },
                'w': {
                    'typeOfLevel': 'isobaricInhPa',
                    'level': levels,
                },
                'gh': {
                    'typeOfLevel': 'isobaricInhPa',
                    'level': levels,
                },
                't': {
                    'typeOfLevel': 'isobaricInhPa',
                    'level': levels,
                },
                'q': {
                    'typeOfLevel': 'isobaricInhPa',
                    'level': levels,
                },
            }
        }
    
    def process_data_with_wgrib(self, date):

        self.outdir = pathlib.Path(f'{self.bucket}/{date.strftime("%Y%m%d%H")}')
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.download_data_from_s3(date)
    
        da_all = []
        surface_grbs = pygrib.open(f'{self.bucket}/{date.strftime("%Y%m%d%H")}/GFSFLX.GrbF00')
        pl_grbs = pygrib.open(f'{self.bucket}/{date.strftime("%Y%m%d%H")}/GFSPRS.GrbF00')
        
        for level_type, variables in self.vars_dict.items():
            for var, value in variables.items():
        
                levelType = value['typeOfLevel']
                desired_level = value['level']
            
        
                #print(f'Get variable {var}:')
                if level_type == 'gfsflx':
                    da = get_dataarray(surface_grbs, var, levelType, desired_level)
                elif level_type == 'gfsprs':
                    da = get_dataarray(pl_grbs, var, levelType, desired_level)
                else:
                    raise ValueError(f'Type {levelType} is not supported!')
                da_all.append(da)
        
        ds = xr.merge(da_all)
        ds = ds.rename({
            '10u': '10m_u_component_of_wind',
            '10v': '10m_v_component_of_wind',
            '2t': '2m_temperature',
            'prmsl': 'mean_sea_level_pressure',
            'pwat': 'precipitable_water',
            'gh': 'geopotential',
            'u': 'u_component_of_wind',
            'v': 'v_component_of_wind',
            'w': 'vertical_velocity',
            't': 'temperature',
            'q': 'specific_humidity',
        })

        ds['geopotential'] = ds['geopotential'] * 9.80665


        if self.initilize:
            ds.to_zarr(self.data_path, mode='w', encoding={'time': {'units': f'hours since {self.year}-01-01T00:00:00'}})
        else:
            ds.to_zarr(self.data_path, mode='a', append_dim='time')

        try:
            os.system(f"rm -rf {self.outdir}")
        except Exception as e:
            print(f"Error removing folder {self.outdir}: {str(e)}")

    def download_data_from_s3(self, date):
    
        prefix = ['GFSFLX', 'GFSPRS']
        for pre in prefix:
            file_name = f'{pre}.GrbF00'
            local_file_name = f'{self.outdir}/{file_name}'
            key = f'{date.year}/{date.month:02d}/{date.strftime("%Y%m%d%H")}/{file_name}'
            print(key)
            with open(local_file_name, 'wb') as f:
                self.s3.download_fileobj(self.bucket, key, f)

    @property
    def s3(self):
        return boto3.client('s3', config=Config(signature_version=UNSIGNED))

    @property
    def bucket(self):
        return 'noaa-ufs-gefsv13replay-pds'


if __name__ == '__main__':
    startdate = datetime(1994, 1, 1, 6)
    enddate = datetime(2023, 1, 1, 12)
    datevectors = np.arange(startdate, enddate, timedelta(years=1)).astype(datetime)

    outdir = '/lustre/Linlin.Cui/mlsfs/data/UFS/out_of_sample'

    fv3 = FV3Dataset(outdir)
    for date in datevectors:
        fv3.process_data_with_wgrib(date)

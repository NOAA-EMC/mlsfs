import logging

import numpy as np
import xarray as xr
import dask

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',    
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S') 

variables = {
    'surface': [
        '10m_u_component_of_wind', '10m_v_component_of_wind', 
        '2m_temperature', 'surface_pressure', 'mean_sea_level_pressure', 'total_column_water_vapour', 
        'total_precipitation_6hr', 'sea_ice_cover', 'total_cloud_cover', 
        'sea_surface_temperature', 'toa_incident_solar_radiation',
    ],
    'pressure_level': [
        'u_component_of_wind', 'v_component_of_wind', 
        'vertical_velocity', 'temperature', 'specific_humidity', 'geopotential',
    ], 
    'prescribed': ['geopotential_at_surface', 'land_sea_mask', 'lake_cover'],
}

levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
startdate = "1979-01-01"
enddate = "2016-12-31"

#ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr') 
#var = 'toa_incident_solar_radiation' 
#values = ds[var].sel(time=slice(startdate, enddate)) 
#mean_toa = dask.compute(values.mean()) 
#std_toa = dask.compute(values.std()) 
#ds.close()

zarr_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr'
ds = xr.open_zarr(zarr_path) 
#ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr') 
DAs_mean = []
DAs_std = []
for key, variables in variables.items(): 
    for var in variables:
        logging.info(f"variable {var}")
        if key == 'surface':
            #if var != 'toa_incident_solar_radiation':
            values = ds[var].sel(time=slice(startdate, enddate))
            mean = dask.compute(values.mean())
            std = dask.compute(values.std())
                #data_mean.append(mean[0])
                #data_std.append(std[0])
            #else:
            #    mean = mean_toa
            #    std = std_toa

            da = xr.DataArray(
                data = mean[0],
                name = var,
            )
            DAs_mean.append(da)

            da = xr.DataArray(
                data = std[0],
                name = var,
            )
            DAs_std.append(da)

        elif key == 'pressure_level':
            values = ds[var].sel(time=slice(startdate, enddate)).sel(level=levels)
            mean = dask.compute(values.mean(axis=(0, 2, 3)))
            std = dask.compute(values.std(axis=(0, 2, 3)))
            data_mean, data_std = [], []
            for ilev in np.arange(len(levels)):
                data_mean.append(mean[0][ilev])
                data_std.append(std[0][ilev])

            da = xr.DataArray(
                data = np.array(data_mean),
                dims=['level'],
                coords={'level': levels},
                name = var,
            )
            DAs_mean.append(da)

            da = xr.DataArray(
                data = np.array(data_std),
                dims=['level'],
                coords={'level': levels},
                name = var,
            )
            DAs_std.append(da)

        else:
            if key == 'prescribed':
                values = ds[var]
                mean = dask.compute(values.mean())
                std = dask.compute(values.std())
                data_mean.append(mean[0])
                data_std.append(std[0])

                da = xr.DataArray(
                    data = mean[0],
                    name = var,
                )
                DAs_mean.append(da)

                da = xr.DataArray(
                    data = std[0],
                    name = var,
                )
                DAs_std.append(da)

            else:
                raise valueError(f'{key} is not in ["surface", "pressure_level", "prescribed"]')

ds.close()
ds_mean = xr.merge(DAs_mean)
ds_mean.to_netcdf('mean_by_level_1979-2016.nc')

ds_std = xr.merge(DAs_std)
ds_std.to_netcdf('std_by_level_1979-2016.nc')

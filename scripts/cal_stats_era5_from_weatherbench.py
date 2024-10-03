import numpy as np
import xarray as xr
import dask

variables = {
    'surface': [
        '10m_u_component_of_wind', '10m_v_component_of_wind', 
        '2m_temperature', 'mean_sea_level_pressure', 
        'total_precipitation_6hr', 'sea_ice_cover', 'total_cloud_cover', 'sea_surface_temperature', 
        'mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux', 'toa_incident_solar_radiation',
        ##'mean_surface_net_long_wave_radiation_flux', 'mean_surface_net_short_wave_radiation_flux'
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

data_mean = []
data_std = []
# toa_incident_solar_radiation is avaiable in 1959-2022 dataset 
ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr') 
var = 'toa_incident_solar_radiation' 
values = ds[var].sel(time=slice(startdate, enddate)) 
mean_toa = dask.compute(values.mean()) 
std_toa = dask.compute(values.std()) 
#da = xr.DataArray( 
#    data = mean[0], 
#    name = var, 
#) 
#DAs.append(da) 
ds.close() 

ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr') 
for key, variables in variables.items(): 
    for var in variables:
        if key == 'surface':
            print(var)
            if var != 'toa_incident_solar_radiation':
                values = ds[var].sel(time=slice(startdate, enddate))
                mean = dask.compute(values.mean())
                std = dask.compute(values.std())
                data_mean.append(mean[0])
                data_std.append(std[0])
                breakpoint()
            else:
                data_mean.append(mean_toa[0])
                data_std.append(std_toa[0])
            #da = xr.DataArray(
            #    data = mean[0],
            #    name = var,
            #)
            #DAs.append(da)
            #print(f'{var} shape is {values.shape}')

        elif key == 'pressure_level':
            values = ds[var].sel(time=slice(startdate, enddate)).sel(level=levels)
            mean = dask.compute(values.mean(axis=(0, 2, 3)))
            std = dask.compute(values.std(axis=(0, 2, 3)))
            for ilev in np.arange(len(levels)):
                data_mean.append(mean[0][ilev])
                data_std.append(std[0][ilev])

        else:
            if key == 'prescribed':
                values = ds[var]
                mean = dask.compute(values.mean())
                std = dask.compute(values.std())
                data_mean.append(mean[0])
                data_std.append(std[0])

            else:
                raise valueError(f'{key} is not in ["surface", "pressure_level", "prescribed"]')


        #    da = xr.DataArray(
        #        data = mean[0],
        #        dims=['level'],
        #        name = var,
        #    )
        #    DAs.append(da)

ds.close()
#ds_mean = xr.merge(DAs)
#ds_mean.to_netcdf('mean_surface_1979-2016.nc')

data_mean = np.array(data_mean)
data_mean_exp = np.expand_dims(data_mean, axis=(0, 2, 3))
data_std = np.array(data_std)
data_std_exp = np.expand_dims(data_std, axis=(0, 2, 3))

np.save('global_mean_1979-2016.npy', data_mean_exp)
np.save('global_std_1979-2016.npy', data_std_exp)

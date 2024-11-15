import numpy as np
import xarray as xr


# chn=85 ( 5+6*13+2)
variables= {
    'surface': [
        '10m_u_component_of_wind', '10m_v_component_of_wind', 
        '2m_temperature', 'mean_sea_level_pressure', 'total_column_water_vapour',
    ],
    'pressure_level': [
        'u_component_of_wind', 'v_component_of_wind', 
        'vertical_velocity', 'temperature', 'specific_humidity', 'geopotential',
    ], 
    'prescribed': ['toa_incident_solar_radiation', 'sea_surface_temperature', 'geopotential_at_surface', 'land_sea_mask', 'lake_cover'],
}

levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

ds_means = xr.open_dataset('mean_by_level_1979-2016.nc')
ds_stds = xr.open_dataset('std_by_level_1979-2016.nc')

means, stds = [], []
for key, variables in variables.items(): 
    for var in variables:
        print(var)
        if key == 'surface' or key == 'prescribed':
            means.append(ds_means[var].values.item())
            stds.append(ds_stds[var].values.item())

        elif key == 'pressure_level':
            for level in levels:
                print(level)
                means.append(ds_means[var].sel(level=level).values.item())
                stds.append(ds_stds[var].sel(level=level).values.item())
        else:
            raise valueError(f'{key} is not in ["surface", "pressure_level", "prescribed"]')

breakpoint()
means = np.array(means)
mean_exp = np.expand_dims(means, axis=(0, 2, 3))
stds = np.array(stds)
std_exp = np.expand_dims(stds, axis=(0, 2, 3))

np.save('global_mean_1979-2016_chn85.npy', mean_exp)
np.save('global_std_1979-2016_chn85.npy', std_exp)

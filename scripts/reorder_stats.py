import numpy as np

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

variables_reordered = {
    'surface': [
        '10m_u_component_of_wind', '10m_v_component_of_wind', 
        '2m_temperature', 'mean_sea_level_pressure', 
        #'total_precipitation_6hr', 'total_cloud_cover', 
        #'mean_surface_latent_heat_flux', 'mean_surface_sensible_heat_flux', 
        #'sea_ice_cover', 'sea_surface_temperature', 
    ],
    'pressure_level': [
        'u_component_of_wind', 'v_component_of_wind', 
        'vertical_velocity', 'temperature', 'specific_humidity', 'geopotential',
    ], 
    'prescribed': ['toa_incident_solar_radiation', 'geopotential_at_surface', 'land_sea_mask', 'lake_cover'],
}

levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

means = np.load('global_mean_1979-2016.npy')
stds = np.load('global_std_1979-2016.npy')

i = 0
means_dict = {}
stds_dict = {}
for key, variables in variables.items(): 
    for var in variables:
        if key == 'surface':
            print(f'{i}: {var}')
            if var != 'toa_incident_solar_radiation':
                means_dict[var] = means[0,i,0,0]
                stds_dict[var] = stds[0,i,0,0]
                i += 1
            else:
                means_dict[var] = means[0,i,0,0]
                stds_dict[var] = stds[0,i,0,0]
                i += 1
            #da = xr.DataArray(
            #    data = mean[0],
            #    name = var,
            #)
            #DAs.append(da)
            #print(f'{var} shape is {values.shape}')

        elif key == 'pressure_level':
            for ilev in np.arange(len(levels)):
                print(f'{i}: {var}, level {ilev}')
                means_dict[f'{var}{levels[ilev]}'] = means[0,i,0,0]
                stds_dict[f'{var}{levels[ilev]}'] = stds[0,i,0,0]
                i += 1

        else:
            if key == 'prescribed':
                print(f'{i}: {var}')
                means_dict[var] = means[0,i,0,0]
                stds_dict[var] = stds[0,i,0,0]
                i += 1

            else:
                raise valueError(f'{key} is not in ["surface", "pressure_level", "prescribed"]')

means2, stds2 = [], []

i = 0
for key, variables2 in variables_reordered.items(): 
    for var in variables2:
        if key == 'surface':
            print(f'{i}: {var}')
            if var != 'toa_incident_solar_radiation':
                means2.append(means_dict[var])
                stds2.append(stds_dict[var])
                i += 1
            else:
                means2.append(means_dict[var])
                stds2.append(stds_dict[var])
                i += 1
            #da = xr.DataArray(
            #    data = mean[0],
            #    name = var,
            #)
            #DAs.append(da)
            #print(f'{var} shape is {values.shape}')

        elif key == 'pressure_level':
            for ilev in np.arange(len(levels)):
                print(f'{i}: {var}, level {ilev}')
                means2.append(means_dict[f'{var}{levels[ilev]}'])
                stds2.append(stds_dict[f'{var}{levels[ilev]}']) 
                i += 1

        else:
            if key == 'prescribed':
                print(f'{i}: {var}')
                means2.append(means_dict[var])
                stds2.append(stds_dict[var])
                i += 1

            else:
                raise valueError(f'{key} is not in ["surface", "pressure_level", "prescribed"]')

print(means[0,50,0,0])
print(means2[43])
print(stds[0,88,0,0])
print(stds2[81])

means2 = np.array(means2)
mean_exp = np.expand_dims(means2, axis=(0, 2, 3))
stds2 = np.array(stds2)
std_exp = np.expand_dims(stds2, axis=(0, 2, 3))

np.save('global_mean_1979-2016_less.npy', mean_exp)
np.save('global_std_1979-2016_less.npy', std_exp)

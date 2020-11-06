'''
#from netCDF4 import Dataset  # use scipy instead
£from scipy.io import netcdf #### <--- This is the library to import.

# Open file in a netCDF reader
directory = 'Data/'
wrf_file_name = directory+'Valid_Small_ERA5_US.nc' 
nc = netcdf.netcdf_file(wrf_file_name,'r')

#Look at the variables available
print(nc.variables)

#Look at the dimensions
print(nc.dimensions)

#Look at a specific variable's dimensions
print(nc.variables['T2'].dimensions)   ## output is ('Time', 'south_north', 'west_east')

#Look at a specific variable's units
print(nc.variables['T2'].units)        ## output is ('K')
'''

'''
To do:
What does nv mean?

'''

import netCDF4 as nc
import xarray 
import torch
import pandas as pd
import sys 

filename = 'Data/dt_med_twosat_phy_l4_20190215_vDT2018.nc'
ds = nc.Dataset(filename)
#print(ds['time'][:])
#print(type(ds))
#print(ds.dimensions)

#print(ds.variables)
#Load the data as an xarray:
Y_data=xarray.open_dataset(filename)#.to_array()
Y_data=Y_data.drop_dims('nv').to_array()

#Select the variables of interest:
Y_data=Y_data.loc[['ugos','vgos','ugosa','vgosa']]

#Transpose the data and get the number of observations:
Y_data=Y_data.transpose("time","longitude","latitude","variable") 
n_obs=Y_data.shape[0]

Longitude=torch.tensor(Y_data.coords['longitude'].values,dtype=torch.get_default_dtype())
Latitude=torch.tensor(Y_data.coords['latitude'].values,dtype=torch.get_default_dtype())


n_per_y_axis=Latitude.size(0)
n_per_x_axis=Longitude.size(0)

n_points_per_obs=n_per_y_axis*n_per_x_axis

X_tensor=torch.stack([Longitude.repeat_interleave(n_per_y_axis),Latitude.repeat(n_per_x_axis)],dim=1)

Y_tensor=torch.from_numpy(Y_data.loc[:,:,:,['ugos','vgos']].values).reshape(-1,2)



'''
#Save variables list:
variables=list(Y_data.coords['variable'].values)
#Save the number of variables:
n_variables=len(variables)

#Transpose the data and get the number of observations:
Y_data=Y_data.transpose("time","longitude","latitude","variable","nv") 
n_obs=Y_data.shape[0]

Longitude=torch.tensor(Y_data.coords['longitude'].values,dtype=torch.get_default_dtype())
Latitude=torch.tensor(Y_data.coords['latitude'].values,dtype=torch.get_default_dtype())

variables=list(Y_data.coords['variable'].values)
#Variables should be: ['crs', 'lat_bnds', 'lon_bnds', 'err', 'adt', 'ugos', 'vgos', 'sla', 'ugosa', 'vgosa']
Y_data=Y_data[:,:,:,[5,6,8,9]]

#We pick: ugos, vgos,
#128 grid points per latitude
#344 grid points per longitude
#Goal: pick 32E 34N, radius 2 degrees!
#1. Region west of cyprus: 32N-36N, 28E-32E
#2. Region near sizilia: 33-37N, 17-21 N
#3. Region 
print(Y_data.dims)
print(Y_data.sizes)

print("NV values:")
print(Y_data.coords['nv'].values)
'''
'''
data=[
['cyprus1', 35,30,32,27],
['cyprus2', 36,32,33,29],
['malta1',38.5,20,35.5,17],
['malta2',34.5,19.,31.5,16.],
['marseille',42.5,8,39.5,5],
['algier',40.5,8,37.5,5]
['bengasi,36,23,33,20]
]
region_frame=pd.DataFrame(columns=['name','N','E','S','W'],data=data)
'''


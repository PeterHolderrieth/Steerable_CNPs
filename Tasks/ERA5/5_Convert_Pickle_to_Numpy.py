import pandas as pd
import sys
import numpy as np
import xarray 

old_filename=sys.argv[1]
new_filename=sys.argv[2]
'''
#Load data frame:
df=pd.read_pickle(old_filename)

#Swap Longitude and Latitutde columns and east and north components such that x1 corresponds to longitutde and x2 to latitude:
col_list=list(df)
col_list=[col_list[0]]+[col_list[2]]+[col_list[1]]+col_list[3:6]+[col_list[7]]+[col_list[6]]+[col_list[9]]+[col_list[8]]
df=df.reindex(columns=col_list).reset_index(drop=True)
#Sort:
df.sort_values(by=['datetime','Longitude','Latitude'], axis=0, inplace=True)
df.set_index(keys=['datetime','Longitude','Latitude'], drop=True,inplace=True) 
X=df.to_xarray()
X=X.astype(np.float32,casting='same_kind')
X.to_netcdf("saved_on_disk.nc")
'''
X_disk=xarray.open_dataset("saved_on_disk.nc").to_array()
BATCH_SIZE=100
X_disk=X_disk.transpose("datetime","Longitude","Latitude","variable")
for i in range(10):
    ind=np.random.randint(low=0,high=2184,size=(BATCH_SIZE))
    print("Single batch: ")
    print(X_disk[ind])
    print("Single examples: ")
    print(X_disk[i,0,0,:])
'''
df_list=[]


data=pd.read_pickle(filename)

    

print("Start concatenating:")

data=pd.concat(df_list)

print("Finish concatenating.")

#Save the file:
print("Save the file.")
data.to_pickle(filename_merged)

#Data sample:
print("Data sample:")
print(data.sample(20))
print("Process finished.")
'''
import sys
import pandas as pd
import numpy as np
import h5py 

filename=sys.argv[1]

pd.set_option('display.max_rows', 1000)

#df=pd.read_pickle(filename)
#df.set_index(["datetime","Latitude","Longitude"],inplace=True)
#df.sort_index(level=[0,1,2],inplace=True)
#df.to_hdf('ERA5_data.hdf5', key='ERA5', mode='w')

#hf = pd.read_hdf('ERA5_data.hdf5')

f = h5py.File('ERA5_data.hdf5', 'r')
keys=list(f.keys())
print("Keys:", keys)
group=f['ERA5']
print(type(group))
print(list(group.keys()))
for key in group.keys():
    print(group[key])
    


'''
group_key = list(hf.keys())[0]
ds = hf[group_key]

print(ds)
# load only one example
x = ds[0]

# load a subset, slice (n examples) 
arr = ds[:n]

# should load the whole dataset into memory.
# this should be avoided
arr = ds[:]
'''

'''
store = pd.HDFStore('data.hdf5')
chunksize = 1684
print("Start loop to get chunks.")
for i in range(10):
            chunk = store.select('df',
                                 start=i*chunksize,
                                 stop=(i+1)*chunksize)
            print(chunk)  
store.close()
'''

#df_rel=pd.read_hdf('data.hdf5',chunksize=10)
#for chunk in df_rel:
#    print(chunk)
#with h5py.File('data.hdf5', 'r') as f:
#    data=f['datetime']
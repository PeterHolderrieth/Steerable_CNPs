import sys
import pandas as pd
import numpy as np
import h5py 

filename=sys.argv[1]

pd.set_option('display.max_rows', 1000)

df=pd.read_pickle(filename)
#df.set_index(["datetime","Latitude","Longitude"],inplace=True)
#df.sort_index(level=[0,1,2],inplace=True)
##print(df.head(600))
#print(df.index)
df.to_hdf('data.hdf5', key='df', mode='w',format='table',data_columns=True)

hf = pd.read_hdf('data.hdf5')

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
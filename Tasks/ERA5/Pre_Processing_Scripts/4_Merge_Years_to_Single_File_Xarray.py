import pandas as pd
import sys
import numpy as np
import xarray
import datetime 

location=sys.argv[1]
filename_without_year=sys.argv[2]
filename_merged=sys.argv[3]

MIN_YEAR=2016
MAX_YEAR=2018


df_list=[]

for year in range(MIN_YEAR,MAX_YEAR+1):
    #Get filename:
    filename=location+str(year)+filename_without_year
    print("Get file: ", filename)
    #Read data from pickle file:
    new_data=pd.read_pickle(filename)
    
    df_list.append(new_data)

print("Start concatenating:")
df=pd.concat(df_list)
print("Finished concatenating.")
print("Swap columns.")
#Swap Longitude and Latitutde columns and east and north components such that x1 corresponds to longitutde and x2 to latitude:
col_list=list(df)
col_list=[col_list[0]]+[col_list[2]]+[col_list[1]]+col_list[3:6]+[col_list[7]]+[col_list[6]]+[col_list[9]]+[col_list[8]]
df=df.reindex(columns=col_list).reset_index(drop=True)
print("Finished swap columns.")
print("Start sort:")
df.sort_values(by=['datetime','Longitude','Latitude'], axis=0, inplace=True)
df.set_index(keys=['datetime','Longitude','Latitude'], drop=True,inplace=True) 
print("Finished sort.")
print("Convert to xarray:")
X=df.to_xarray()
print("Cast to dtype:")
X=X.astype(np.float32,casting='same_kind')
print("Finished cast to dtype.")

#Save the file:
print("Convert to xarray:")
X.to_netcdf(location+filename_merged+".nc")

print("Process finished.")


import pandas as pd 
import sys
import numpy as np

location=sys.argv[1]
filename_without_year=sys.argv[2]
filename_merged=sys.argv[3]

MIN_YEAR=1980
MAX_YEAR=2018
df_list=[]

for year in range(MIN_YEAR,MAX_YEAR+1):
    #Get filename:
    filename=location+str(year)+filename_without_year
    print("Get file: ", filename)
    #Read data to csv:
    new_data=pd.read_csv(filename,delimiter=",")
    #Control whether the number of times where the predictions and measurements do not derive from the same time:
    valid_time=new_data.validityTime.to_numpy(dtype=np.int32)
    valid_date=new_data.validityDate.to_numpy(dtype=np.int32)
    data_time=new_data.dataTime.to_numpy(dtype=np.int32)
    data_date=new_data.dataDate.to_numpy(dtype=np.int32)
    n_inequal=np.sum(np.logical_or(valid_time!=data_time,valid_date!=data_date))
    if n_inequal>0:
        print("Filename: ", filename)
        sys.exit("Error: n_inequal is not zero.")
    else:
        new_data.drop(columns=["dataDate"],inplace=True) 
        new_data.drop(columns=["dataTime"],inplace=True) 
        new_data.rename(columns={'validityDate': "Date", 'validityTime': "Time"},inplace=True)
        df_list.append(new_data)

print("Start concatenating:")
data=pd.concat(df_list)
print("Finish concatenating.")



#Gravitation constant:
g=9.80665

#Divide z by geopotential:
print("Divide by gravitation constant to get height.")
data.z=data.z/g
data.rename(columns={'z': "height_in_m" },inplace=True)

#Get temperature in Celsius:
print("Kelvin to Celius.")

data[["2t"]]=data[["2t"]]-273.15
data.rename(columns={'2t': "t_in_Cels" },inplace=True)

#Get pressure in hPa:
print("Pascal to kPa.")

data.sp=data.sp/1000
data.rename(columns={'sp': "sp_in_kPa" },inplace=True)

#Rename the wind components:
data.rename(columns={'10u': 'wind_10m_north', 
                     '10v':'wind_10m_east',
                     '100u':'wind_100m_north',
                     '100v': 'wind_100m_east'},inplace=True)
data.to_csv(filename_merged, index=False)

#Data sample:
print("Data sample:")
print(data.sample(12))
print("Process finished.")
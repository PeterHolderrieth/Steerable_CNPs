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
    #Read data from pickle file:
    new_data=pd.read_pickle(filename)
    
    df_list.append(new_data)

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
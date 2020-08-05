import pandas as pd
import sys
import numpy as np

old_filename=sys.argv[1]
new_filename=sys.argv[2]

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
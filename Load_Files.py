#Tensors:
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import pandas as pd 

#E(2)-steerable CNNs - librar"y:
from e2cnn import gspaces                                          
from e2cnn import nn as G_CNN   
#import e2cnn

#Plotting in 2d/3d:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
import matplotlib.cm as cm

#Tools:
from datetime import datetime
from datetime import timedelta
import sys

#Own files:
import Kernel_and_GP_tools as GP
import My_Tools
import Steerable_CNP_Models as My_Models
import Training

file_single="Tasks/ERA5/ERA5_US/Data/ERA5_US_FILE.pickle"

df_single=pd.read_pickle(file_single)
DATA_ARRAY=df_single.to_numpy().reshape((-1,1681, 7))
print("DATA array shape: ", DATA_ARRAY.shape)

def give_single_file_batch(data_array,batch_size=3):
    n,_,_=data_array.shape
    ind_batch=np.random.randint(low=0,high=n,size=(batch_size))
    return(data_array[ind_batch])

def rand_datetime(starttime,max_n_days):
    n_days=np.random.randint(low=0,high=max_n_days+1)
    n_hours=np.random.randint(low=0,high=24)
    return(starttime+timedelta(days=n_days,hours=n_hours))

def list_rand_datetime(starttime,max_n_days,size):
    return([rand_datetime(starttime,max_n_days) for i in range(size)])

def give_split_file_batch(starttime,max_n_days,batch_size,fileloc,filename):
    list_times=list_rand_datetime(starttime,max_n_days,batch_size)
    list_times_str=[datetime.strftime(format=("%Y_%m_%d_%H")) for datetime in list_times]
    list_filenames=[fileloc+time_str+filename for time_str in list_times_str]
    array_list=[pd.read_pickle(file_str).to_numpy() for file_str in list_filenames]
    return(np.stack(array_list,axis=0))

STARTTIME=datetime.strptime("1980_01_01_00", '%Y_%m_%d_%H')
MAX_N_DAYS=57
FILELOC="Tasks/ERA5/ERA5_US/Data/Single_Files/"
FILENAME="_ERA5_US.pickle"
for BATCH_SIZE in range(3,200,10):
    print("BATCH_SIZE: ", BATCH_SIZE)
    for j in range(10):
        diff_single_agg=timedelta(0.0)
        diff_split_agg=timedelta(0.0)
        point=datetime.today()
        Single_batch=give_single_file_batch(DATA_ARRAY,BATCH_SIZE)
        diff_it_single=datetime.today()-point
        point=datetime.today()
        Split_batch=give_split_file_batch(STARTTIME,MAX_N_DAYS,BATCH_SIZE,FILELOC,FILENAME)
        diff_it_split=datetime.today()-point

        diff_single_agg+=diff_it_single
        diff_split_agg+=diff_it_split
    print("Batch size: ", BATCH_SIZE)
    print("Time single file:", diff_single_agg/BATCH_SIZE)
    print("Time split file:", diff_split_agg/BATCH_SIZE)

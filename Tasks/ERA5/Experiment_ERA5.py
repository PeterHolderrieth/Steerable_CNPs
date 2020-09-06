#LIBRARIES:
#Tensors:
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import xarray

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
import datetime
import sys

sys.path.append("../../")

#Own files:
import Kernel_and_GP_tools as GP
import My_Tools
import Training
import Tasks.ERA5.ERA5_Dataset as Dataset

#HYPERPARAMETERS:
torch.set_default_dtype(torch.float)

'''
SET DEVICE:
'''

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")

'''
DATA
'''
PATH_TO_FILE="ERA5_US/Data/Test_Small_ERA5.nc"
MIN_N_CONT=2
MAX_N_CONT=50
N_TOTAL=None
BATCH_SIZE=60
VAR_NAMES=['sp_in_kPa','t_in_Cels','wind_10m_east', 'wind_10m_north']

ERA5_DATA=Dataset.ERA5Dataset(PATH_TO_FILE,MIN_N_CONT,MAX_N_CONT)

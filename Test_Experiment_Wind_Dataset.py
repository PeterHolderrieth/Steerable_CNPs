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

#Own files:
import Kernel_and_GP_tools as GP
import My_Tools
import Steerable_CNP_Models as My_Models
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
PATH_TO_FILE="Tasks/ERA5/ERA5_US/Data/16_to_18_ERA5_US.nc"
MIN_N_CONT=50
MAX_N_CONT=100
N_TOTAL=None
BATCH_SIZE=10
VAR_NAMES=['wind_10m_east', 'wind_10m_north']

ERA5_WIND_DATA=Dataset.ERA5Dataset(PATH_TO_FILE,MIN_N_CONT,MAX_N_CONT,N_TOTAL,VAR_NAMES)
DATA_IDENTIFIER="ERA5_TEST"
ARCHITECTURE=sys.argv[1]

'''
ENCODER
'''

#Set hyperparamters:
X_RANGE=[-98.,-84.]
Y_RANGE=[28.,42]
N_X_AXIS=50
L_SCALE_ENC=5.
#Define the model:
Encoder=My_Models.Steerable_Encoder(l_scale=L_SCALE_ENC,x_range=X_RANGE,n_x_axis=N_X_AXIS,y_range=Y_RANGE,normalize=True)

print('Encoder grid', Encoder.grid)

'''
STEERABLE CNP
'''

DIM_COV_EST=1
N=4
L_SCALE_OUT=5.

#Set parameters for Steerable Decoder:
if ARCHITECTURE=="small":
    GEOM_KERNEL_SIZES=[7,9,11]
    GEOM_NON_LINEARITY=['NormReLU']
    HIDDEN_FIB_REPS=[[-1,-1],[-1,-1,1,1]]
elif ARCHITECTURE=="middle":
    GEOM_KERNEL_SIZES=[7,9,11,15,17]
    GEOM_NON_LINEARITY=['NormReLU']
    HIDDEN_FIB_REPS=[[-1,1],[-1,-1,-1,-1,1],[-1,-1,-1,-1,1],[-1,-1,1,1]]
elif ARCHITECTURE=="big":
    GEOM_KERNEL_SIZES=[7,9,11,15,17,33,65]
    GEOM_NON_LINEARITY=['NormReLU']
    HIDDEN_FIB_REPS=[[-1,1],[-1,-1,-1,-1,1],[-1,-1,-1,-1,1],[-1,-1,-1,-1,1,1,1,1],[-1,-1,-1,-1,1,1,1,1],[-1,-1,-1,1,1,1]]
else:
    sys.exit("Unknown architecture")

geom_decoder=My_Models.Cyclic_Decoder(hidden_fib_reps=HIDDEN_FIB_REPS,kernel_sizes=GEOM_KERNEL_SIZES,dim_cov_est=DIM_COV_EST,non_linearity=GEOM_NON_LINEARITY,N=N)
geom_cnp=My_Models.Steerable_CNP(encoder=Encoder,decoder=geom_decoder,dim_cov_est=DIM_COV_EST,l_scale=L_SCALE_OUT)

'''
TRAINING PARAMETERS
'''

N_EPOCHS=int(sys.argv[2])
N_ITERAT_PER_EPOCH=int(sys.argv[3])
LEARNING_RATE=1e-5
WEIGHT_DECAY=float(sys.argv[4])
SHAPE_REG=None
N_PLOTS=None
<<<<<<< HEAD
<<<<<<< HEAD
N_VAL_SAMPLES=20
=======
N_VAL_SAMPLES=10
>>>>>>> b71a2d0ffd467eca72fe6336fc125e4a538c8aaf
=======
N_VAL_SAMPLES=None#10
>>>>>>> a7bc8330271335acb478c9ba4cf2fc3f9792d71a

#File path to save models:
FOLDER="Trained_Models/ERA5_Wind_Test/"

'''
Train Steerable CNP with non-div-free kernel:
'''

geom_n_param=My_Tools.count_parameters(geom_decoder,print_table=True)


print("---------Train EquivCNP--------")

FILENAME=FOLDER+"ERA5_Wind_"+str(ARCHITECTURE)


_,_,geom_file_loc=Training.train_CNP(
Steerable_CNP=geom_cnp, 
train_dataset=ERA5_WIND_DATA,
val_dataset=ERA5_WIND_DATA, 
data_identifier=DATA_IDENTIFIER,
device=DEVICE,
n_epochs=N_EPOCHS, 
n_iterat_per_epoch=N_ITERAT_PER_EPOCH,
learning_rate=LEARNING_RATE, 
weight_decay=WEIGHT_DECAY,
shape_reg=SHAPE_REG,
n_plots=N_PLOTS,
n_val_samples=N_VAL_SAMPLES,
filename=FILENAME)


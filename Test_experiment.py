#LIBRARIES:
#Tensors:

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

#E(2)-steerable CNNs - librar"y:
from e2cnn import gspaces    
from e2cnn import nn as G_CNN   
import e2cnn

#Plotting in 2d/3d:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
import matplotlib.cm as cm

#Tools:
import datetime
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Own files:
import Kernel_and_GP_tools as GP
import My_Tools
import EquivDeepSets 
import Training
from Cov_Converter import cov_converter
import Decoder_Models as models
import Architectures
import EquivCNP
import Tasks.GP_Data.GP_div_free_circle.loader as DataLoader

#HYPERPARAMETERS and set seed:
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

DIM_COV_EST=int(sys.argv[1])
N_EPOCHS=int(sys.argv[2])
N_ITERAT_PER_EPOCH=int(sys.argv[3])
X_RANGE=[-10,10]
N_X_AXIS=40
BATCH_SIZE=3
LEARNING_RATE=float(sys.argv[4])
N=int(sys.argv[5])
name='regular_little'

encoder=EquivDeepSets.EquivDeepSets(x_range=X_RANGE,n_x_axis=N_X_AXIS)
decoder=models.get_EquivDecoder(name,dim_cov_est=DIM_COV_EST,context_rep_ids=[1],flip=False,N=N)

equivcnp=EquivCNP.EquivCNP(encoder,decoder,DIM_COV_EST,dim_context_feat=2)

FILEPATH="Tasks/GP_Data/GP_div_free_circle/"
train_dataset=DataLoader.give_GP_div_free_data_set(5,50,'train',file_path=FILEPATH)
val_dataset=DataLoader.give_GP_div_free_data_set(5,50,'valid',file_path=FILEPATH)
data_identifier="GP_div_free_circle"

Training.train_CNP(equivcnp,train_dataset,val_dataset,data_identifier,DEVICE,BATCH_SIZE,N_EPOCHS,N_ITERAT_PER_EPOCH,LEARNING_RATE,n_val_samples=None)

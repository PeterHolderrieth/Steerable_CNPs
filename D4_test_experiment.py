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
LIST_NAMES=["regular_little",
        "regular_small",
        "regular_middle",
        "regular_big",
        "regular_huge",
        "irrep_little",
        "irrep_small",
        "irrep_middle",
        "irrep_big",
        "irrep_huge"
        ]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")

DIM_COV_EST=int(sys.argv[1])
N_EPOCHS=int(sys.argv[2])
N_ITERAT_PER_EPOCH=int(sys.argv[3])
LEARNING_RATE=float(sys.argv[4])
name=sys.argv[5]

X_RANGE=[-10,10]
N_X_AXIS=30
BATCH_SIZE=30
N_VAL_SAMPLES=None
PRINT_PROGRESS=False
N_EVAL_SAMPLES=10000
FILEPATH="Tasks/GP_Data/GP_div_free_circle/"                                                       
data_identifier="GP_div_free_circle"
train_dataset=DataLoader.give_GP_div_free_data_set(5,50,'train',file_path=FILEPATH)                 
val_dataset=DataLoader.give_GP_div_free_data_set(5,50,'valid',file_path=FILEPATH)

print()
print("Group: D4")
print('Model type:')
print(name)
print("Learning rate: ", LEARNING_RATE)
encoder=EquivDeepSets.EquivDeepSets(x_range=X_RANGE,n_x_axis=N_X_AXIS)
decoder=models.get_D4_Decoder(name,dim_cov_est=DIM_COV_EST,context_rep_ids=[[1,1]])

#My_Tools.count_parameters(decoder,print_table=True)
equivcnp=EquivCNP.EquivCNP(encoder,decoder,DIM_COV_EST,dim_context_feat=2)
CNP,_,_=Training.train_CNP(equivcnp,train_dataset,val_dataset,data_identifier,DEVICE,BATCH_SIZE,N_EPOCHS,N_ITERAT_PER_EPOCH,LEARNING_RATE,n_val_samples=N_VAL_SAMPLES,print_progress=PRINT_PROGRESS)
eval_log_ll=Training.test_CNP(CNP,val_dataset,DEVICE,n_samples=N_EVAL_SAMPLES,batch_size=BATCH_SIZE)
print("Final log ll:", eval_log_ll)
print()

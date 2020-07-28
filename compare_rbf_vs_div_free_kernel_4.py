#%%
#LIBRARIES:
#Tensors:
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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
import Steerable_CNP_Models as My_Models
import Training
import Evaluation
import Tasks.GP_div_free_small.loader as GP_loader

#Set hyperparameters:
torch.set_default_dtype(torch.float)
torch.manual_seed(3012)
np.random.seed(3012)

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
#Loading Data to data loaders:
BATCH_SIZE=16
TRAIN_DATASET=GP_loader.give_GP_div_free_data_set(Min_n_cont=5,Max_n_cont=50,n_total=None,data_set='train')
VAL_DATASET=GP_loader.give_GP_div_free_data_set(Min_n_cont=5,Max_n_cont=50,n_total=None,data_set='valid')
DATA_IDENTIFIER="GP_data_small"
GP_PARAMETERS={'l_scale':0.5,
'sigma_var': 2., 
'kernel_type':"div_free",
'obs_noise':1e-4}

'''
ENCODER
'''
#Set hyperparamters:
X_RANGE=[-4,4]
N_X_AXIS=40
L_SCALE_ENC=0.2
#Define the model:
Encoder=My_Models.Steerable_Encoder(l_scale=L_SCALE_ENC,x_range=X_RANGE,n_x_axis=N_X_AXIS)

'''
STEERABLE CNP
'''
#Set parameters for Steerable Decoder:
DIM_COV_EST=3
N=4
CONV_KERNEL_SIZES=[5,7,9,11,17,19,25,29]
LIST_HID_CHANNELS=[6,9,11,13,11,9,6]
CONV_NON_LINEARITY=['ReLU']

conv_decoder=My_Models.CNN_Decoder(list_hid_channels=LIST_HID_CHANNELS,kernel_sizes=CONV_KERNEL_SIZES,dim_cov_est=DIM_COV_EST,non_linearity=CONV_NON_LINEARITY)

#Set parameters for kernel dict out:
KERNEL_DICT_OUT_RBF={'kernel_type': 'rbf'}
KERNEL_DICT_OUT_DIV_FREE={'kernel_type': 'div_free'}

#Define two different cnps:
rbf_cnp=My_Models.Steerable_CNP(encoder=Encoder,decoder=conv_decoder,dim_cov_est=DIM_COV_EST,kernel_dict_out=KERNEL_DICT_OUT_RBF)

div_free_cnp=My_Models.Steerable_CNP(encoder=Encoder,decoder=conv_decoder,dim_cov_est=DIM_COV_EST,
                                        kernel_dict_out=KERNEL_DICT_OUT_DIV_FREE, normalize_output=False)


'''
TRAINING PARAMETERS
'''

N_EPOCHS=30
N_ITERAT_PER_EPOCH=1000
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.
SHAPE_REG=None
N_PLOTS=None
N_VAL_SAMPLES=200

#File path to save models:
FOLDER="Trained_Models/Comp_experiments/Rbf_vs_div_free_kernel/"

'''
Train Steerable CNP with non-div-free kernel:
'''
geom_n_param=My_Tools.count_parameters(conv_decoder,print_table=True)


print("---------Train rbf ConvCNP--------")

RBF_FILENAME=FOLDER+"Exp_3_rbf.py"


_,_,geom_file_loc=Training.train_CNP(
Steerable_CNP=rbf_cnp, 
train_dataset=TRAIN_DATASET,
val_dataset=VAL_DATASET, 
data_identifier=DATA_IDENTIFIER,
device=DEVICE,
n_epochs=N_EPOCHS, 
n_iterat_per_epoch=N_ITERAT_PER_EPOCH,
learning_rate=LEARNING_RATE, 
weight_decay=WEIGHT_DECAY,
shape_reg=SHAPE_REG,
n_plots=N_PLOTS,
n_val_samples=N_VAL_SAMPLES,
filename=RBF_FILENAME)


'''
Train Steerable CNP with div-free kernel:
'''

print("---------Train div free ConvCNP--------")

DIV_FREE_FILENAME=FOLDER+"Exp_3_div_free.py"

_,_,conv_file_loc=Training.train_CNP(
Steerable_CNP=div_free_cnp, 
train_dataset=TRAIN_DATASET,
val_dataset=VAL_DATASET, 
data_identifier=DATA_IDENTIFIER,
device=DEVICE,
n_epochs=N_EPOCHS, 
n_iterat_per_epoch=N_ITERAT_PER_EPOCH,
learning_rate=LEARNING_RATE, 
weight_decay=WEIGHT_DECAY,
shape_reg=SHAPE_REG,
n_plots=N_PLOTS,
n_val_samples=N_VAL_SAMPLES,
filename=DIV_FREE_FILENAME)



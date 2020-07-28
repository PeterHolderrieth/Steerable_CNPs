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

#x_context,y_context,x_target,y_target=VAL_DATASET.get_rand_batch(batch_size=1)
#My_Tools.Plot_Inference_2d(x_context.squeeze(),y_context.squeeze(),x_target.squeeze(),y_target.squeeze(),quiver_scale=30)
x_context,y_context,x_target,y_target=TRAIN_DATASET.get_rand_batch(batch_size=1)
My_Tools.Plot_Inference_2d(x_context.squeeze(),y_context.squeeze(),x_target.squeeze(),y_target.squeeze(),quiver_scale=30)
x_context,y_context,x_target,y_target=TRAIN_DATASET.get_rand_batch(batch_size=1)
My_Tools.Plot_Inference_2d(x_context.squeeze(),y_context.squeeze(),x_target.squeeze(),y_target.squeeze(),quiver_scale=30)
x_context,y_context,x_target,y_target=TRAIN_DATASET.get_rand_batch(batch_size=1)
My_Tools.Plot_Inference_2d(x_context.squeeze(),y_context.squeeze(),x_target.squeeze(),y_target.squeeze(),quiver_scale=30)
x_context,y_context,x_target,y_target=TRAIN_DATASET.get_rand_batch(batch_size=1)
My_Tools.Plot_Inference_2d(x_context.squeeze(),y_context.squeeze(),x_target.squeeze(),y_target.squeeze(),quiver_scale=30)

# %%

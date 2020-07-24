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

GEOM_PATH="Trained_Models/Comp_experiments/Equal_par_exp_Conv_CNP_2020_07_23_23_32"
CONV_PATH="Trained_Models/Comp_experiments/Equal_par_exp_Steerable_CNP_2020_07_23_23_19"

geom_dict=torch.load(GEOM_PATH,map_location=torch.device('cpu'))
conv_dict=torch.load(CONV_PATH,map_location=torch.device('cpu'))

N=geom_dict['CNP_dict']['decoder_dict']['N']
G_act=gspaces.Rot2dOnR2(N=N)

geom_eval=Evaluation.Steerable_CNP_Evaluater(geom_dict,G_act,G_act.irrep(1))
geom_eval.plot_loss_memory()
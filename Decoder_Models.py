#LIBRARIES:
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

#E(2)-steerable CNNs - librar"y:
from e2cnn import gspaces    
from e2cnn import nn as G_CNN   
from e2cnn import group  
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
import Cov_Converter
import Architectures as AC 

#HYPERPARAMETERS and set seed:
torch.set_default_dtype(torch.float)

'''
Number of parameters:
little - ca 1 000
small - ca 20 000
middle - ca 100 000
big - ca 500 000
huge - ca 2M
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

def get_EquivDecoder(name,dim_cov_est,context_rep_ids,N,flip,max_frequency=30):
    
    #Family of decoders using purely regular fiber representations:
    if name=="regular_little":
        hidden_reps_ids=2*[3*[-1]]
        kernel_sizes=[j for j in range(7,20,6)]
        non_linearity=['ReLU']

    elif name=="regular_small":
        hidden_reps_ids=4*[4*[-1]]
        kernel_sizes=[j for j in range(3,20,4)]
        non_linearity=['ReLU']

    elif name=="regular_middle":
        hidden_reps_ids=6*[12*[-1]]
        kernel_sizes=[j for j in range(3,28,4)]
        non_linearity=['ReLU']

    elif name=="regular_big":
        hidden_reps_ids=6*[24*[-1]]
        kernel_sizes=[5,5,5,7,7,11,11]
        non_linearity=['ReLU']
    
    elif name=="regular_huge":
        hidden_reps_ids=8*[24*[-1]]
        kernel_sizes=[5,5,5,5,7,7,9,15,21]
        non_linearity=['ReLU']

    #Family of decoders using irreps and regular representations:

    elif name=="irrep_little":
        if flip:
            hidden_reps_ids=2*[8*[[1,1]]]
        else:
            hidden_reps_ids=2*[8*[1]]
        kernel_sizes=[j for j in range(7,20,6)]
        non_linearity=['NormReLU']

    elif name=="irrep_small":
        if flip:
            hidden_reps_ids=5*[12*[[1,1]]]
        else:
            hidden_reps_ids=5*[12*[1]]      
        kernel_sizes=[j for j in range(3,24,4)]
        non_linearity=['NormReLU']

    elif name=="irrep_middle":
        if flip:
            hidden_reps_ids=7*[36*[[1,1]]]
        else:
            hidden_reps_ids=7*[36*[1]]  
        kernel_sizes=[3,3,5,5,11,11,13,13]
        non_linearity=['NormReLU']

    elif name=="irrep_big":
        if flip:
            hidden_reps_ids=10*[64*[[1,1]]]
        else:
            hidden_reps_ids=10*[64*[1]]
        kernel_sizes=[5,5,5,7,7,11,11,13,15,17,19]
        non_linearity=['NormReLU']

    elif name=="irrep_huge":
        if flip:
            hidden_reps_ids=16*[80*[[1,1]]]
        else:
            hidden_reps_ids=16*[80*[1]]
        kernel_sizes=[5,5,5,5,7,7,7,7,11,11,11,11,13,15,17,19,21]
        non_linearity=['NormReLU']    
    else:
        sys.exit("Unkown architecture name.")

    return(AC.EquivDecoder(hidden_reps_ids=hidden_reps_ids,
                           kernel_sizes=kernel_sizes,
                            dim_cov_est=dim_cov_est,
                            context_rep_ids=context_rep_ids,
                            N=N,
                            flip=flip,
                            non_linearity=non_linearity,
                            max_frequency=max_frequency))


def get_CNNDecoder(name,dim_cov_est,dim_features_inp=2):
    if name=="little":
        list_hid_channels=[4,5]
        kernel_sizes=[5,7,5]
        non_linearity=["ReLU"]

    elif name=="small":
        list_hid_channels=[8,12,12]
        kernel_sizes=[5,9,11,5]
        non_linearity=["ReLU"]
    
    elif name=="middle":
        list_hid_channels=4*[24]
        kernel_sizes=5*[7]
        non_linearity=["ReLU"]
    
    elif name=="big":
        list_hid_channels=6*[40]
        kernel_sizes=7*[7]
        non_linearity=["ReLU"]

    elif name=="huge":
        list_hid_channels=8*[52]
        kernel_sizes=9*[9]
        non_linearity=["ReLU"]
    else:
        sys.exit("Unknown decoder architecture name.")
    

    return(AC.CNNDecoder(list_hid_channels=list_hid_channels,
                        kernel_sizes=kernel_sizes, 
                        non_linearity=non_linearity,
                        dim_cov_est=dim_cov_est,
                        dim_features_inp=dim_features_inp
                        ))

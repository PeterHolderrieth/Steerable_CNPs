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

LIST_NAMES=["regular_little",
"regular_small",
"regular_middle",
"regular_big",
"regular_huge",
"irrep_little",
"irrep_small",
"irrep_middle"
"irrep_big",
"irrep_huge"
]

def get_EquivCNP(name,dim_cov_est,context_rep_ids,N,flip,max_frequency=30):
    
    #Family of decoders using purely regular fiber representations:
    if name=="regular_little":
        hidden_reps_ids=2*[2*[-1]]
        kernel_sizes=3*[7]
        non_linearity=['ReLU']

    elif name=="regular_small":
        hidden_reps_ids=4*[4*[-1]]
        kernel_sizes=5*[11]
        non_linearity=['ReLU']

    elif name=="regular_middle":
        hidden_reps_ids=6*[12*[-1]]
        kernel_sizes=7*[7]
        non_linearity=['ReLU']

    elif name=="regular_big":
        hidden_reps_ids=6*[24*[-1]]
        kernel_sizes=7*[7]
        non_linearity=['ReLU']
    
    elif name=="regular_huge":
        hidden_reps_ids=8*[24*[-1]]
        kernel_sizes=9*[11]
        non_linearity=['ReLU']

    #Family of decoders using irreps and regular representations:

    elif name=="irrep_little":
        if flip:
            hidden_reps_ids=2*[2*[[1,1]]]
        else:
            hidden_reps_ids=2*[2*[1]]
        kernel_sizes=3*[11]
        non_linearity=['NormReLU']

    elif name=="irrep_small":
        if flip:
            hidden_reps_ids=4*[4*[[1,1]]]
        else:
            hidden_reps_ids=4*[4*[1]]      
        kernel_sizes=5*[9]
        non_linearity=['NormReLU']

    elif name=="irrep_middle":
        if flip:
            hidden_reps_ids=6*[12*[[1,1]]]
        else:
            hidden_reps_ids=6*[12*[1]]  
        kernel_sizes=7*[7]
        non_linearity=['NormReLU']

    elif name=="irrep_big":
        if flip:
            hidden_reps_ids=6*[24*[[1,1]]]
        else:
            hidden_reps_ids=6*[24*[[1,1]]]
        kernel_sizes=7*[7]
        non_linearity=['NormReLU']

    elif name=="irrep_huge":
        if flip:
            hidden_reps_ids=8*[2*[[1,1]]]
        kernel_sizes=9*[7]
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

for name in LIST_NAMES:
    print(name)
    decoder=get_EquivCNP(name,dim_cov_est=4,context_rep_ids=[1],N=10,flip=False,max_frequency=30)

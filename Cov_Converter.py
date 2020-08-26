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

#HYPERPARAMETERS and set seed:
torch.set_default_dtype(torch.float)

#The following function gives the fiber representation for the eigen value covariance converter:
def get_eig_val_cov_conv_rep(G_act):
    '''
        Input:
            G_act - instance of e2cnn.gspaces.r2.rot2d_on_r2.Rot2dOnR2 - underlying group
            
        Output:
            psd_rep - instance of e2cnn.group.Representation - group representation of the group representation before the covariance 
            feat_type_pre_rep - instance of G_CNN.FieldType - corresponding field type
    '''
    #Change of basis matrix:
    change_of_basis=np.array([[1,1.,0.],
                          [0.,0.,1.],
                          [1,-1.,0.]])
    
    #Get group order and control:
    N=G_act.fibergroup.order()
    if N<=3 and N!=-1: sys.exit("Group order is not valid.")

    if isinstance(G_act,gspaces.FlipRot2dOnR2):
        irreps=['irrep_0,0','irrep_1,2'] if N>4 else ['irrep_0,0','irrep_1,2','irrep_1,2']
    elif isinstance(G_act,gspaces.Rot2dOnR2):
        irreps=['irrep_0','irrep_2'] if N>4 else ['irrep_0','irrep_2','irrep_2']
    else:
        sys.exit("Error: Unknown group.")

    psd_rep=e2cnn.group.Representation(group=G_act.fibergroup,name="eig_val_rep",irreps=irreps,
                                   change_of_basis=change_of_basis,
                                   supported_nonlinearities=['n_relu'])

    return(psd_rep)

#The following function gives the fiber representation for the different of covariance converters:
def get_pre_cov_rep(G_act,dim_cov_est):
    '''
        G_act - instance of e2cnn.gspaces.r2.rot2d_on_r2.Rot2dOnR2 - underlying group
        dim_cov_est - int - either 1,2,3 or 4  - gives dimension and also the type of the covariance converter
    '''
    if dim_cov_est==1:
        return(G_act.trivial_repr)
    elif dim_cov_est==2:
        return(group.directsum(2*[G_act.trivial_repr]))
    elif dim_cov_est==3:
        return(get_eig_val_cov_conv_rep(G_act))
    elif dim_cov_est==4:
        if isinstance(G_act,gspaces.FlipRot2dOnR2):
            vec_rep=G_act.irrep(1,1)
        elif isinstance(G_act,gspaces.Rot2dOnR2):
            vec_rep=G_act.irrep(1)
        else:
            sys.exit('Error: unknown group.')  
        return(group.directsum(2*[vec_rep]))
    else:
        sys.exit('Error when loading pre covariance representation: dim_cov_est can only be 1,2,3 or 4')
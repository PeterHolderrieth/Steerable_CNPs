#!/usr/bin/env python
# coding: utf-8

# To do:
# 1. Structure the class encoder and ConvCNP better: Allow for variable CNN to be defined
# (Is it necessary that the height and width of output feature map is the same the input height and width? Otherwise,
# it gets a mess)
# 2. Change the architecture su|ch that it allows for minibatches of data sets (so far only: minibatch size is one)
# 3. Show in an example with plot that equivariance is not fulfilled (maybe one before training, one after traing)
# |

# In[1]:

#LIBRARIES:
#Tensors:
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
#Sparse matrices:
from scipy.sparse import coo_matrix

#E(2)-steerable CNNs - librar"y:
from e2cnn import gspaces    
import e2cnn
from e2cnn import nn as G_CNN   

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

# In[2]:


#HYPERPARAMETERS:
#Set default as double:
torch.set_default_dtype(torch.float)
#Scale for plotting with plt quiver
quiver_scale=15

# In[13]:

#Define the group and the group actions:        
G_act = gspaces.Rot2dOnR2(N=8)
#%%
#G_act.irreps
change_of_basis=np.array([[0.5,0.,0.5],
                          [0.5,0.,-0.5],
                          [0.,1.,0.]])
psd_rep=e2cnn.group.Representation(group=G_act.fibergroup,name="psd_rep",irreps=['irrep_2'],change_of_basis=change_of_basis,
                           supported_nonlinearities=['n_relu'])

feat_type_pre_cov=G_CNN.FieldType(G_act, [psd_rep])

def cov_softplus(X):
    '''
    X- torch.tensor - shape (n,3)
    '''
    n=X.size(0)
    M=torch.stack([X[:,0],X[:,1],X[:,1],X[:,2]],dim=1).view(n,2,2)
    return(M)  

#X=torch.randn((5,3))
torch.symeig(cov_softplus(X),eigenvectors=True)    




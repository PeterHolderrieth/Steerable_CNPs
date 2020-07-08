#!/usr/bin/env python
# coding: utf-8

# To do:
# 1. Structure the class encoder and ConvCNP better: Allow for variable CNN to be defined
# (Is it necessary that the height and width of output feature map is the same the input height and width? Otherwise,
# it gets a mess)
# 2. Change the architecture su|ch that it allows for minibatches of data sets (so far only: minibatch size is one)
# 3. Show in an example with plot that equivariance is not fulfilled (maybe one before training, one after traing)
# |

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


#HYPERPARAMETERS:
#Set default as double:
torch.set_default_dtype(torch.float)
#Scale for plotting with plt quiver
quiver_scale=15


#Define the group and the group actions:        
G_act = gspaces.Rot2dOnR2(N=8)

#G_act.irreps
'''change_of_basis=np.array([[0.5,0.,0.5],
                          [0.5,0.,-0.5],
                          [0.,1.,0.]])'''
change_of_basis=np.array([[1,1.,0.],
                          [0.,0.,1.],
                          [1,-1.,0.]])
psd_rep=e2cnn.group.Representation(group=G_act.fibergroup,name="psd_rep",irreps=['irrep_0','irrep_2'],
                                   change_of_basis=change_of_basis,
                                   supported_nonlinearities=['n_relu'])

feat_type_pre_cov=G_CNN.FieldType(G_act, [psd_rep])
feat_type_cov=G_CNN.FieldType(G_act, [G_act.irrep(1)])

change_of_basis=psd_rep.change_of_basis
change_of_basis_inv=psd_rep.change_of_basis_inv
#%%
X=torch.randn((100,3))
size_scale=2
ellip_scale=1
f = plt.figure(figsize=(10,3))
Cov=My_Tools.cov_activation_function(X)
v_1=torch.tensor([1.,0.,1.])
v_2=torch.tensor([1.,0.,-1.])
v_3=torch.tensor([0,1.,0])
for g in G_act.testing_elements:
    M_cov=torch.tensor(psd_rep(g),dtype=torch.get_default_dtype())
    #print(M_cov)
    X_trans=torch.matmul(M_cov,X.t()).t()
    M_std=torch.tensor(G_act.irrep(1)(g),dtype=torch.get_default_dtype())
    #print(M_std)
    #print(torch.mv(M_cov,v_1))
    #print(torch.mv(M_cov,v_2))
    #print(torch.mv(M_cov,v_3))
    #print(X_trans)
    Cov_trans=My_Tools.cov_activation_function(X_trans)
    #print(Cov_trans)
    trans_Cov=torch.matmul(torch.matmul(M_std,Cov),M_std.t())
    #print(Cov_trans)
    #print(trans_Cov)
    print(torch.norm(Cov_trans-trans_Cov))
    '''
    trans_Cov=torch.matmul(Cov
    #Decompose A:
    eigen_decomp=torch.eig(Cov_trans,eigenvectors=True)
    print(eigen_decomp)
    #Get the eigenvector corresponding corresponding to the largest eigenvalue:
    u=eigen_decomp[1][:,0]

    #Get the angle of the ellipse in degrees:
    alpha=360*torch.atan(u[1]/u[0])/(2*math.pi)

    #Get the width and height of the ellipses (eigenvalues of A):
    D=torch.sqrt(eigen_decomp[0][:,0])
    print(D)
    #Plot the Ellipse:
    E=Ellipse(xy=np.array([0,0]),width=ellip_scale*D[0],height=ellip_scale*D[1],angle=alpha)
    print(E)
    '''

#!/usr/bin/env python
# coding: utf-8

# To do:
# 1. Structure the class encoder and ConvCNP better: Allow for variable CNN to be defined
# (Is it necessary that the height and width of output feature map is the same the input height and width? Otherwise,
# it gets a mess)
# 2. Change the architecture su|ch that it allows for minibatches of data sets (so far only: minibatch size is one)
# 3. Show in an example with plot that equivariance is not fulfilled (maybe one before training, one after traing)

#LIBRARIES:
#Tensors:
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

#E(2)-steerable CNNs - librar"y:
from e2cnn import gspaces                                          
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

def SETUP_EXP_1_Cyclic_GP_div_free(Training_par,N=8,batch_size=3):
    
    G_act = gspaces.Rot2dOnR2(N=N)
    feat_type_in=G_CNN.FieldType(G_act, [G_act.irrep(1)])
    GP_train_data_loader,GP_test_data_loader=GP.load_2d_GP_data(Id="37845",batch_size=batch_size)
    GP_parameters={'l_scale':1,'sigma_var':1, 'kernel_type':"div_free",'obs_noise':1e-4,'B':None,'Ker_project':False}
    Operator_par={'train_data_loader': GP_train_data_loader,'test_data_loader': GP_test_data_loader}
    
    #Define the grid, the kernel parameters and the encoder:
    grid_dict={'x_range':[-3,3],'y_range':[-3,3],'n_x_axis':20,'n_y_axis':20}
    kernel_dict_emb={'sigma_var':1,'kernel_type':"rbf",'Ker_project':False}
    encoder=My_Models.Steerable_Encoder(**grid_dict,kernel_dict=kernel_dict_emb,normalize=True)
    
    #Define the kernel parameters for the kernel smoother:
    kernel_dict_out={'sigma_var':1,'kernel_type':"rbf",'B':None,'Ker_project':False}
    #---------------------Conv CNP decoder-----------------------------
    conv_decoder=nn.Sequential(nn.Conv2d(3,16,kernel_size=5,stride=1,padding=2),
              nn.ReLU(),
              nn.Conv2d(16,16,kernel_size=7,stride=1,padding=3),
              nn.ReLU(),
              nn.Conv2d(16,16,kernel_size=5,stride=1,padding=2),
              nn.ReLU(),
              nn.Conv2d(16,12,kernel_size=7,stride=1,padding=3),
              nn.ReLU(),
              nn.Conv2d(12,5,kernel_size=5,stride=1,padding=2))
    #---------------------Steerable CNP decoder-----------------------------
    #Define the f||eature types:
    psd_rep,_=My_Tools.get_pre_psd_rep(G_act)
    feat_type_out=G_CNN.FieldType(G_act,[G_act.irrep(1),psd_rep])
    feat_types=[G_CNN.FieldType(G_act, [G_act.trivial_repr,G_act.irrep(1)]),
                G_CNN.FieldType(G_act, 2*[G_act.regular_repr]),
                G_CNN.FieldType(G_act,2*[G_act.regular_repr]),
                G_CNN.FieldType(G_act,2*[G_act.regular_repr]),
                G_CNN.FieldType(G_act,[G_act.regular_repr]),
                feat_type_out]

    #Define the kernel sizes:
    kernel_sizes=[5,7,5,7,5]
    geom_decoder=My_Models.Steerable_Decoder(feat_types,kernel_sizes)
    
    #Get the convcnp:
    conv_cnp=My_Models.Steerable_CNP(feature_in=feat_type_in,dim_cov_est=3,G_act=G_act,encoder=encoder,decoder=conv_decoder,kernel_dict_out=kernel_dict_out)    
    geom_cnp=My_Models.Steerable_CNP(feature_in=feat_type_in,dim_cov_est=3,G_act=G_act,encoder=encoder,decoder=geom_decoder,kernel_dict_out=kernel_dict_out)
    
    Conv_CNP_Operator=My_Models.Steerable_CNP_Operator(conv_cnp,**Training_par,**Operator_par)
    Geom_CNP_Operator=My_Models.Steerable_CNP_Operator(geom_cnp,**Training_par,**Operator_par)
    
    return(Conv_CNP_Operator,Geom_CNP_Operator,GP_parameters)    
 
  #%%-------------------------------------
#-----Experiment 1
#----------------------------------------  
Training_par={'Max_n_context_points':50,'n_epochs':3,'n_plots':None,'n_iterat_per_epoch':1,
            'learning_rate':1e-4}    
Conv_CNP,Geom_CNP,GP_parameters=SETUP_EXP_1_Cyclic_GP_div_free(Training_par,N=8,batch_size=3)
loss_Geom_CNP=Geom_CNP.train(filename=None)#filename_1+"_Steerable_CNP_")
#loss_ConvCNP=Conv_CNP.train(filename=None)#filename_1+"_Conv_CNP_")



Geom_CNP.plot_log_ll_memory()
#Conv_CNP.plot_log_ll_memory()

Geom_CNP.plot_test_random(GP_parameters=GP_parameters)
#Conv_CNP.plot_test_random(GP_parameters=GP_parameters)

#Geom_CNP.plot_test_random(GP_parameters=GP_parameters)
#Conv_CNP.plot_test_random(GP_parameters=GP_parameters)

#%%
#That is how to load the model again:
#Geom_CNP_Operator.load_state_dict(torch.load("Trained_Models/Initial_comparison_experiment_Steerable_CNP__2020_07_02_12_39"))
#Conv_CNP_Operator.load_state_dict(torch.load("Trained_Models/Initial_comparison_experiment_ConvCNP__2020_07_02_12_39"))


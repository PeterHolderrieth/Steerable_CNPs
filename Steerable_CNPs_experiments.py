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

# In[2]:


#HYPERPARAMETERS:
#Set default as double:
torch.set_default_dtype(torch.float)
#Scale for plotting with plt quiver
quiver_scale=15

# In[13]:

#Define the group and the group actions:        
G_act = gspaces.Rot2dOnR2(N=8)
feat_type_in=G_CNN.FieldType(G_act, [G_act.irrep(1)])

#Load the data and set the parameters for the Operator:
GP_train_data_loader,GP_test_data_loader=GP.load_2d_GP_data(Id="37845",batch_size=3)
GP_parameters={'l_scale':1,'sigma_var':1, 'kernel_type':"div_free",'obs_noise':1e-4,'B':None,'Ker_project':False}
Operator_par={'train_data_loader': GP_train_data_loader,'test_data_loader': GP_test_data_loader}

#Define the grid, the kernel parameters and the encoder:
grid_dict={'x_range':[-3,3],'y_range':[-3,3],'n_x_axis':20,'n_y_axis':20}
kernel_dict_emb={'sigma_var':1,'kernel_type':"rbf",'Ker_project':False}
encoder=My_Models.Steerable_Encoder(**grid_dict,kernel_dict=kernel_dict_emb,normalize=True)

#Define the kernel parameters for the kernel smoother:
kernel_dict_out={'sigma_var':1,'kernel_type':"rbf",'B':None,'Ker_project':False}


#---------------------ConvCNP-----------------------------
#Get CNN as a decoder:
conv_decoder=nn.Sequential(nn.Conv2d(3,16,kernel_size=5,stride=1,padding=2),
              nn.ReLU(),
              nn.Conv2d(16,16,kernel_size=7,stride=1,padding=3),
              nn.ReLU(),
              nn.Conv2d(16,16,kernel_size=5,stride=1,padding=2),
              nn.ReLU(),
              nn.Conv2d(16,12,kernel_size=7,stride=1,padding=3),
              nn.ReLU(),
              nn.Conv2d(12,4,kernel_size=5,stride=1,padding=2))

#Get the ConvCNP and the corresponding operator:
Conv_CNP=My_Models.Steerable_CNP(G_act=G_act,feature_in=feat_type_in,encoder=encoder,decoder=conv_decoder,kernel_dict_out=kernel_dict_out)

#---------------------Steerable CNP-----------------------------
#Define the f||eature types:

feat_types=[G_CNN.FieldType(G_act, [G_act.trivial_repr,G_act.irrep(1)]),
            G_CNN.FieldType(G_act, 2*[G_act.regular_repr]),
            G_CNN.FieldType(G_act,2*[G_act.regular_repr]),
            G_CNN.FieldType(G_act,2*[G_act.regular_repr]),
            G_CNN.FieldType(G_act,[G_act.regular_repr]),
            G_CNN.FieldType(G_act, [G_act.irrep(1),G_act.trivial_repr,G_act.trivial_repr])]
#Define the kernel sizes:
kernel_sizes=[5,7,5,7,5]

geom_decoder=My_Models.Steerable_Decoder(feat_types,kernel_sizes)
geom_cnp=My_Models.Steerable_CNP(feature_in=feat_type_in,G_act=G_act,encoder=encoder,decoder=geom_decoder,kernel_dict_out=kernel_dict_out)

#%%
G_act.representations


#%%-------------------------------------
#-----Experiment 1
#----------------------------------------
filename_1="Pre_hyperparameter_tuning_1"
Training_par_1={'Max_n_context_points':50,'n_epochs':20,'n_plots':None,'n_iterat_per_epoch':200,
            'learning_rate':1e-4}

Conv_CNP_Operator_1=My_Models.Steerable_CNP_Operator(Conv_CNP,**Training_par_1,**Operator_par)
Geom_CNP_Operator_1=My_Models.Steerable_CNP_Operator(geom_cnp,**Training_par_1,**Operator_par)

loss_ConvCNP=Conv_CNP_Operator_1.train(filename=filename_1+"_Conv_CNP_")
loss_Geom_CNP=Geom_CNP_Operator_1.train(filename=filename_1+"_Steerable_CNP_")

Geom_CNP_Operator_1.plot_log_ll_memory()
Conv_CNP_Operator_1.plot_log_ll_memory()
#%%
Geom_CNP_Operator_1.plot_test_random(GP_parameters=GP_parameters)
#Conv_CNP_Operator_1.plot_test_random(GP_parameters=GP_parameters)
#%%-------------------------------------
#-----Experiment 2
#----------------------------------------

filename_2="Pre_hyperparameter_tuning_2"
Training_par_2={'Max_n_context_points':30,'n_epochs':40,'n_plots':None,'n_iterat_per_epoch':400,
                'learning_rate':1e-4}

Conv_CNP_Operator_2=My_Models.Steerable_CNP_Operator(Conv_CNP,**Training_par_2,**Operator_par)
Geom_CNP_Operator_2=My_Models.Steerable_CNP_Operator(geom_cnp,**Training_par_2,**Operator_par)

loss_Geom_CNP=Geom_CNP_Operator_2.train(filename=filename_2+"_Steerable_CNP_")
loss_ConvCNP=Conv_CNP_Operator_2.train(filename=filename_2+"_Conv_CNP_")

#Geom_CNP_Operator_2.plot_log_ll_memory()
Conv_CNP_Operator_2.plot_log_ll_memory()

#%%-------------------------------------
#-----Experiment 3
#----------------------------------------

filename_3="Pre_hyperparameter_tuning_3"
Training_par_3={'Max_n_context_points':50,'n_epochs':40,'n_plots':None,'n_iterat_per_epoch':400,
                'learning_rate':1e-5}

Conv_CNP_Operator_3=My_Models.Steerable_CNP_Operator(Conv_CNP,**Training_par_2,**Operator_par)
Geom_CNP_Operator_3=My_Models.Steerable_CNP_Operator(geom_cnp,**Training_par_2,**Operator_par)

loss_ConvCNP=Conv_CNP_Operator_3.train(filename=filename_3+"_Conv_CNP_")
loss_Geom_CNP=Geom_CNP_Operator_3.train(filename=filename_3+"_Steerable_CNP_")

Conv_CNP_Operator_3.plot_log_ll_memory()
Geom_CNP_Operator_3.plot_log_ll_memory()



#%%
#That is how to load the model again:
#Geom_CNP_Operator.load_state_dict(torch.load("Trained_Models/Initial_comparison_experiment_Steerable_CNP__2020_07_02_12_39"))
#Conv_CNP_Operator.load_state_dict(torch.load("Trained_Models/Initial_comparison_experiment_ConvCNP__2020_07_02_12_39"))


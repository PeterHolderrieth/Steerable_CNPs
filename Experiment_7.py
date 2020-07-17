#%%
#THIS FILE PERFORMS BUILDS A STEERABLE AND CONV CNP ARCHITECTURE 
#HYPERPARAMETERS:
#1. Dimension of covariance estimation: 1 - diagonal covariance estimation
#2. Number of layers: 5
#3. Here we use a shape regularizer

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
if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


def SETUP_EXP_7_Cyclic_GP_div_free(Training_par,N=8,batch_size=3):
    
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
              nn.Conv2d(12,3,kernel_size=5,stride=1,padding=2))
    #---------------------Steerable CNP decoder-----------------------------
    #Define the f||eature types:
    #psd_rep,_=My_Tools.get_pre_psd_rep(G_act)
    feat_type_out=G_CNN.FieldType(G_act,[G_act.irrep(1),G_act.trivial_repr])
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
    conv_cnp=My_Models.Steerable_CNP(feature_in=feat_type_in,dim_cov_est=1,G_act=G_act,encoder=encoder,decoder=conv_decoder,kernel_dict_out=kernel_dict_out)    
    geom_cnp=My_Models.Steerable_CNP(feature_in=feat_type_in,dim_cov_est=1,G_act=G_act,encoder=encoder,decoder=geom_decoder,kernel_dict_out=kernel_dict_out)
    
    #Send the models to the correct devices:
    conv_cnp=conv_cnp.to(device)
    geom_cnp=geom_cnp.to(device)
    
    #Create Operator model:
    Conv_CNP_Operator=My_Models.Steerable_CNP_Operator(conv_cnp,**Training_par,**Operator_par)
    Geom_CNP_Operator=My_Models.Steerable_CNP_Operator(geom_cnp,**Training_par,**Operator_par)
    
    return(Conv_CNP_Operator,Geom_CNP_Operator,GP_parameters)    
 

n_epochs=3
n_iterat=1
train=True
evaluate=False
n_tests=400

#------------------------------------
#-----Experiment 1.1:
#----------------------------------------  
Training_par={'Max_n_context_points':50,'n_epochs':n_epochs,'n_plots':None,'n_iterat_per_epoch':n_iterat,
            'learning_rate':1e-4,'shape_reg':0.01}    
Conv_CNP,Geom_CNP,GP_parameters=SETUP_EXP_7_Cyclic_GP_div_free(Training_par,N=8,batch_size=3)
if train:
    filename_71="Exp_7_1"
    starttime=datetime.datetime.today()
    print("Start training experiment 1.1: ", starttime)
    loss_Geom_CNP=Geom_CNP.train(filename="Initial_ziz_exp_1507/Exp_7/"+filename_71+"_Steerable_CNP_")
    loss_ConvCNP=Conv_CNP.train(filename="Initial_ziz_exp_1507/Exp_7/"+filename_71+"_Conv_CNP_")
    endtime=datetime.datetime.today()
    print("Duration of training on device: ",device,": ",endtime-starttime)

if evaluate:
    Conv_CNP.load_state_dict(torch.load("Trained_Models/Initial_ziz_exp_1507/Exp_7Exp_7_1_Conv_CNP__2020_07_17_14_47",map_location=torch.device('cpu')),strict=False)
    Geom_CNP.load_state_dict(torch.load("Trained_Models/Initial_ziz_exp_1507/Exp_7Exp_7_1_Steerable_CNP__2020_07_17_14_34",map_location=torch.device('cpu')),strict=False)
    X,Y=next(iter(Conv_CNP.test_data_loader))
    n_context_points=torch.randint(size=[],low=2,high=Conv_CNP.Max_n_context_points)
    x_context,y_context,x_target,y_target=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
    Conv_CNP.plot_test(x_context,y_context,x_target,y_target,GP_parameters=None,title="Exp7: ConvCNP")
    Geom_CNP.plot_test(x_context,y_context,x_target,y_target,GP_parameters=GP_parameters,title="Exp7: SteerCNP")
    print("Exp. 7: Log-LL Steer.: ",Geom_CNP.test(n_tests))
    print("Exp. 7: Log-LL Conv.: ",Conv_CNP.test(n_tests))


#------------------------------------
#-----Experiment 7.2:
#----------------------------------------  
Training_par={'Max_n_context_points':50,'n_epochs':n_epochs,'n_plots':None,'n_iterat_per_epoch':n_iterat,
            'learning_rate':1e-4,'shape_reg': 0.001}    
Conv_CNP,Geom_CNP,GP_parameters=SETUP_EXP_7_Cyclic_GP_div_free(Training_par,N=4,batch_size=4)
if train:
    filename_72="Exp_1_2"
    starttime=datetime.datetime.today()
    print("Start training experiment 1.2: ", starttime)
    loss_Geom_CNP=Geom_CNP.train(filename="Initial_ziz_exp_1507/Exp_7/"+filename_72+"_Steerable_CNP_")
    loss_ConvCNP=Conv_CNP.train(filename="Initial_ziz_exp_1507/Exp_7/"+filename_72+"_Conv_CNP_")
    endtime=datetime.datetime.today()
    print("Duration of training on device: ",device,": ",endtime-starttime)
if evaluate:
    Conv_CNP.load_state_dict(torch.load("Trained_Models/Initial_ziz_exp_1507/Exp_7/Exp_1_2_Conv_CNP__2020_07_16_23_53",map_location=torch.device('cpu')))
    Geom_CNP.load_state_dict(torch.load("Trained_Models/Initial_ziz_exp_1507/Exp_7/Exp_1_2_Steerable_CNP__2020_07_16_23_18",map_location=torch.device('cpu')))
    X,Y=next(iter(Conv_CNP.test_data_loader))
    n_context_points=torch.randint(size=[],low=2,high=Conv_CNP.Max_n_context_points)
    x_context,y_context,x_target,y_target=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
    Conv_CNP.plot_test(x_context,y_context,x_target,y_target,GP_parameters=None,title="Exp7.2: ConvCNP")
    Geom_CNP.plot_test(x_context,y_context,x_target,y_target,GP_parameters=GP_parameters,title="Exp7.2: SteerCNP")
    print("Exp. 7.2: Log-LL Steer.: ",Geom_CNP.test(n_tests))
    print("Exp. 7.2: Log-LL Conv.: ",Conv_CNP.test(n_tests))


#------------------------------------
#-----Experiment 7.3:
#----------------------------------------  
Training_par={'Max_n_context_points':50,'n_epochs':n_epochs,'n_plots':None,'n_iterat_per_epoch':n_iterat,
            'learning_rate':1e-4,'shape_reg': None}    
Conv_CNP,Geom_CNP,GP_parameters=SETUP_EXP_7_Cyclic_GP_div_free(Training_par,N=4,batch_size=1)
if train:
    filename_73="Exp_1_3"
    starttime=datetime.datetime.today()
    print("Start training experiment 1.3: ", starttime)
    loss_Geom_CNP=Geom_CNP.train(filename="Initial_ziz_exp_1507/Exp_7/"+filename_73+"_Steerable_CNP_")
    loss_ConvCNP=Conv_CNP.train(filename="Initial_ziz_exp_1507/Exp_7/"+filename_73+"_Conv_CNP_")
    endtime=datetime.datetime.today()
    print("Duration of training on device: ",device,": ",endtime-starttime)
if evaluate:
    Conv_CNP.load_state_dict(torch.load("Trained_Models/Initial_ziz_exp_1507/Exp_7/Exp_1_3_Conv_CNP__2020_07_17_00_14",map_location=torch.device('cpu')))
    Geom_CNP.load_state_dict(torch.load("Trained_Models/Initial_ziz_exp_1507/Exp_7/Exp_1_3_Steerable_CNP__2020_07_17_00_04",map_location=torch.device('cpu')))
    X,Y=next(iter(Conv_CNP.test_data_loader))
    n_context_points=torch.randint(size=[],low=2,high=Conv_CNP.Max_n_context_points)
    x_context,y_context,x_target,y_target=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
    Conv_CNP.plot_test(x_context,y_context,x_target,y_target,GP_parameters=None,title="Exp7: ConvCNP")
    Geom_CNP.plot_test(x_context,y_context,x_target,y_target,GP_parameters=GP_parameters,title="Exp7: SteerCNP")
    print("Exp. 7: Log-LL Steer.: ",Geom_CNP.test(n_tests))
    print("Exp. 7: Log-LL Conv.: ",Conv_CNP.test(n_tests))




# %%

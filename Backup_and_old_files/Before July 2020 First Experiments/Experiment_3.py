#%%
#THIS FILE PERFORMS BUILDS A STEERABLE AND CONV CNP ARCHITECTURE 
#HYPERPARAMETERS:
#1. Dimension of covariance estimation: 1 - diagonal covariance estimation
#2. Number of layers: 9

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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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


def SETUP_EXP_3_Cyclic_GP_div_free(Training_par,N=4,batch_size=3):
    
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
    conv_decoder=nn.Sequential(nn.Conv2d(3,6,kernel_size=5,stride=1,padding=2),
              nn.ReLU(),
              nn.Conv2d(6,12,kernel_size=7,stride=1,padding=3),
              nn.ReLU(),
              nn.Conv2d(12,24,kernel_size=9,stride=1,padding=4),
              nn.ReLU(),
              nn.Conv2d(24,48,kernel_size=11,stride=1,padding=5),
              nn.ReLU(),
              nn.Conv2d(48,48,kernel_size=11,stride=1,padding=5),
              nn.ReLU(),
              nn.Conv2d(48,24,kernel_size=9,stride=1,padding=4),
              nn.ReLU(),
              nn.Conv2d(24,12,kernel_size=7,stride=1,padding=3),
              nn.ReLU(),
              nn.Conv2d(12,6,kernel_size=5,stride=1,padding=2),
              nn.ReLU(),
              nn.Conv2d(6,3,kernel_size=7,stride=1,padding=3)
              )
    #---------------------Steerable CNP decoder-----------------------------
    #Define the f||eature types:
    #psd_rep,_=My_Tools.get_pre_psd_rep(G_act)
    feat_type_out=G_CNN.FieldType(G_act,[G_act.irrep(1),G_act.trivial_repr])
    feat_types=[G_CNN.FieldType(G_act, [G_act.trivial_repr,G_act.irrep(1)]),
                G_CNN.FieldType(G_act, 2*[G_act.trivial_repr,G_act.irrep(1)]),
                G_CNN.FieldType(G_act, 4*[G_act.trivial_repr,G_act.irrep(1)]),
                G_CNN.FieldType(G_act, 8*[G_act.trivial_repr,G_act.irrep(1)]),
                G_CNN.FieldType(G_act, 16*[G_act.trivial_repr,G_act.irrep(1)]),
                G_CNN.FieldType(G_act,16*[G_act.irrep(1),G_act.trivial_repr]),
                G_CNN.FieldType(G_act,8*[G_act.irrep(1),G_act.trivial_repr]),
                G_CNN.FieldType(G_act,4*[G_act.irrep(1),G_act.trivial_repr]),
                G_CNN.FieldType(G_act,2*[G_act.irrep(1),G_act.trivial_repr]),
                feat_type_out]

    #Define the kernel sizes:
    kernel_sizes=[5,7,9,11,11,9,7,5,7]
    geom_decoder=My_Models.Steerable_Decoder(feat_types,kernel_sizes,non_linearity="norm")
    
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
 

n_epochs=30
n_iterat=1000
train=False
evaluate=True
n_tests=200
#------------------------------------
#-----Experiment 3.1:
#----------------------------------------  
Training_par={'Max_n_context_points':50,'n_epochs':n_epochs,'n_plots':None,'n_iterat_per_epoch':n_iterat,
            'learning_rate':1e-4}    
Conv_CNP,Geom_CNP,GP_parameters=SETUP_EXP_3_Cyclic_GP_div_free(Training_par,N=8,batch_size=3)
if train:
    filename_31="Exp_3_1"
    starttime=datetime.datetime.today()
    print("Start training experiment 1.1: ", starttime)
    loss_Geom_CNP=Geom_CNP.train(filename="Initial_ziz_exp_1507/"+filename_31+"_Steerable_CNP_")
    loss_ConvCNP=Conv_CNP.train(filename="Initial_ziz_exp_1507/"+filename_31+"_Conv_CNP_")
    endtime=datetime.datetime.today()
    print("Duration of training on device: ",device,": ",endtime-starttime)

if evaluate:
    Conv_CNP.load_state_dict(torch.load("Trained_Models/Initial_ziz_exp_1507/Exp_3/Exp_3_1_Conv_CNP__2020_07_16_23_37",map_location=torch.device('cpu')))
    #Geom_CNP.load_state_dict(torch.load("Trained_Models/Initial_ziz_exp_1507/Exp_3/Exp_3_1_Steerable_CNP__2020_07_16_23_09",map_location=torch.device('cpu')))
    X,Y=next(iter(Conv_CNP.test_data_loader))
    n_context_points=torch.randint(size=[],low=2,high=Conv_CNP.Max_n_context_points)
    x_context,y_context,x_target,y_target=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
    Conv_CNP.plot_test(x_context,y_context,x_target,y_target,GP_parameters=None,title="Exp. 3.1: ConvCNP")
    #Geom_CNP.plot_test(x_context,y_context,x_target,y_target,GP_parameters=GP_parameters,title="Exp. 3.1: SteerCNP")
    #print("Exp. 3.1: Log-LL Steer.: ",Geom_CNP.test(n_tests))
    print("Exp. 3.1: Log-LL Conv.: ",Conv_CNP.test(n_tests))



#------------------------------------
#-----Experiment 3.2:
#----------------------------------------  
Training_par={'Max_n_context_points':50,'n_epochs':n_epochs,'n_plots':None,'n_iterat_per_epoch':n_iterat,
            'learning_rate':1e-3}    
Conv_CNP,Geom_CNP,GP_parameters=SETUP_EXP_3_Cyclic_GP_div_free(Training_par,N=4,batch_size=4)
if train:
    filename_32="Exp_3_2"
    starttime=datetime.datetime.today()
    print("Start training experiment 1.2: ", starttime)
    loss_Geom_CNP=Geom_CNP.train(filename="Initial_ziz_exp_1507/"+filename_32+"_Steerable_CNP_")
    loss_ConvCNP=Conv_CNP.train(filename="Initial_ziz_exp_1507/"+filename_32+"_Conv_CNP_")
    endtime=datetime.datetime.today()
    print("Duration of training on device: ",device,": ",endtime-starttime)
if evaluate:
    Conv_CNP.load_state_dict(torch.load("Trained_Models/Initial_ziz_exp_1507/Exp_3/Exp_3_2_Conv_CNP__2020_07_17_02_04",map_location=torch.device('cpu')))
    #Geom_CNP.load_state_dict(torch.load("Trained_Models/Initial_ziz_exp_1507/Exp_3/Exp_3_2_Steerable_CNP__2020_07_17_01_24",map_location=torch.device('cpu')))
    X,Y=next(iter(Conv_CNP.test_data_loader))
    n_context_points=torch.randint(size=[],low=2,high=Conv_CNP.Max_n_context_points)
    x_context,y_context,x_target,y_target=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
    Conv_CNP.plot_test(x_context,y_context,x_target,y_target,GP_parameters=None,title="Exp. 3.2: ConvCNP")
    #Geom_CNP.plot_test(x_context,y_context,x_target,y_target,GP_parameters=GP_parameters,title="Exp. 3.2: SteerCNP")
    #print("Exp. 3.2: Log-LL Steer.: ",Geom_CNP.test(n_tests))
    print("Exp. 3.2: Log-LL Conv.: ",Conv_CNP.test(n_tests))
#------------------------------------
#-----Experiment 3.3:
#----------------------------------------  
Training_par={'Max_n_context_points':50,'n_epochs':n_epochs,'n_plots':None,'n_iterat_per_epoch':n_iterat,
            'learning_rate':1e-4}    
Conv_CNP,Geom_CNP,GP_parameters=SETUP_EXP_3_Cyclic_GP_div_free(Training_par,N=4,batch_size=1)
if train:
    filename_33="Exp_3_3"
    starttime=datetime.datetime.today()
    print("Start training experiment 1.3: ", starttime)
    loss_Geom_CNP=Geom_CNP.train(filename="Initial_ziz_exp_1507/"+filename_33+"_Steerable_CNP_")
    loss_ConvCNP=Conv_CNP.train(filename="Initial_ziz_exp_1507/"+filename_33+"_Conv_CNP_")
    endtime=datetime.datetime.today()
    print("Duration of training on device: ",device,": ",endtime-starttime)
if evaluate:
    Conv_CNP.load_state_dict(torch.load("Trained_Models/Initial_ziz_exp_1507/Exp_3/Exp_3_3_Conv_CNP__2020_07_17_02_47",map_location=torch.device('cpu')))
    #Geom_CNP.load_state_dict(torch.load("Trained_Models/Initial_ziz_exp_1507/Exp_3/Exp_3_3_Steerable_CNP__2020_07_17_02_35",map_location=torch.device('cpu')))
    X,Y=next(iter(Conv_CNP.test_data_loader))
    n_context_points=torch.randint(size=[],low=2,high=Conv_CNP.Max_n_context_points)
    x_context,y_context,x_target,y_target=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
    Conv_CNP.plot_test(x_context,y_context,x_target,y_target,GP_parameters=None,title="Exp. 3.3: ConvCNP")
    #Geom_CNP.plot_test(x_context,y_context,x_target,y_target,GP_parameters=GP_parameters,title="Exp. 3.3: SteerCNP")
    #print("Exp. 3.3: Log-LL Steer.: ",Geom_CNP.test(n_tests))
    print("Exp. 3.3: Log-LL Conv.: ",Conv_CNP.test(n_tests))




# %%

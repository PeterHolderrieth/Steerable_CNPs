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
#import e2cnn

#Plotting in 2d/3d:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
import matplotlib.cm as cm

#Tools:
import datetime
import sys

sys.path.append("./")


#Own files:
import Kernel_and_GP_tools as GP
import My_Tools

#This functions create n_samples of a GP and saves it as/in "filename"
#We in particular assume that the dimension of the output space is 2, too.
def Data_Sampler_2d(n_samples=10,min_x=-2,max_x=2,n_grid_points=10,l_scale=1,sigma_var=1, 
                        kernel_type="rbf",B=None,Ker_project=False,obs_noise=1e-4,shuffle=True):
    X_data=torch.empty((n_samples,n_grid_points**2,2))
    Y_data=torch.empty((n_samples,n_grid_points**2,2))
    for i in range(n_samples):
        X,Y=GP.vec_GP_sampler_2dim(min_x=min_x,max_x=max_x,n_grid_points=n_grid_points,l_scale=l_scale,sigma_var=sigma_var, 
                        kernel_type=kernel_type,B=B,Ker_project=Ker_project,obs_noise=obs_noise)
        n=X.size(0)
        ind=torch.randperm(n)
        X_data[i,:,:]=X[ind]
        Y_data[i,:,:]=Y[ind]
        if i%100==0:
            print('Iteration: ', i)
    return(X_data,Y_data)

#This functions create samples and saves it in a filename:
def Create_GP_Data_File_2d(filename,n_samples=10,min_x=-2,max_x=2,n_grid_points=10,l_scale=1,sigma_var=1, 
                        kernel_type="rbf",B=None,Ker_project=False,obs_noise=1e-4):
    X_data,Y_data=Data_Sampler_2d(n_samples,min_x,max_x,n_grid_points,l_scale,sigma_var, 
                        kernel_type,B,Ker_project,obs_noise)
    np.save('Tasks/GP_div_free_big/Data/'+filename+'_X', X_data.numpy())
    np.save('Tasks/GP_div_free_big/Data/'+filename+'_Y', Y_data.numpy())

MIN_X=-10
MAX_X=10
N_GRID_POINTS=40
L_SCALE=3.5
SIGMA_VAR=10. 
KERNEL_TYPE="div_free"
OBS_NOISE=1e-3
N_TRAIN_SAMPLES=20000
N_VAL_SAMPLES=5000
N_TEST_SAMPLES=5000
TRAIN_FILENAME='GP_Big_Train'
VAL_FILENAME='GP_Big_Valid'
TEST_FILENAME='GP_Big_Test'

#Create train data:
Create_GP_Data_File_2d(filename=TRAIN_FILENAME,n_samples=N_TRAIN_SAMPLES,min_x=MIN_X,max_x=MAX_X,n_grid_points=N_GRID_POINTS,l_scale=L_SCALE,sigma_var=SIGMA_VAR,kernel_type=KERNEL_TYPE,
                       obs_noise=OBS_NOISE)
#Create validation data:
Create_GP_Data_File_2d(filename=VAL_FILENAME,n_samples=N_VAL_SAMPLES,min_x=MIN_X,max_x=MAX_X,n_grid_points=N_GRID_POINTS,l_scale=L_SCALE,sigma_var=SIGMA_VAR,kernel_type=KERNEL_TYPE,
                       obs_noise=OBS_NOISE)
#Create test data:
Create_GP_Data_File_2d(filename=TEST_FILENAME,n_samples=N_TEST_SAMPLES,min_x=MIN_X,max_x=MAX_X,n_grid_points=N_GRID_POINTS,l_scale=L_SCALE,sigma_var=SIGMA_VAR,kernel_type=KERNEL_TYPE,
                       obs_noise=OBS_NOISE)
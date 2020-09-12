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

sys.path.append("././")

#Own files:
import Kernel_and_GP_tools as GP
import My_Tools

#This functions create n_samples of a GP on a radial grid:
def Cyclic_GP_Sampler(n_samples,min_x,max_x,n_grid_points,l_scale=1.,sigma_var=1., 
                        kernel_type="div_free",obs_noise=1e-2,chol_noise=):
    #Get a radial grid:
    X_Grid=My_Tools.Radial_Grid(min=min_x,max=max_x,n_axis=n_grid_points)
    n=X_Grid.size(0)
    #Create empty data arrays:
    X_data=torch.empty((n_samples,n,2))
    Y_data=torch.empty((n_samples,n,2))
    for i in range(n_samples):
        #Sample a GP:
        Y=GP.Multidim_GP_sampler(X_Grid,kernel_type=kernel_type,B=None,l_scale=l_scale,sigma_var=sigma_var,obs_noise=obs_noise)
        #Shuffle it and add it to the data arrays:
        ind=torch.randperm(n)
        X_data[i,:,:]=X_Grid[ind]
        Y_data[i,:,:]=Y[ind]
        if i%100==0:
            print('Iteration: ', i)
    return(X_data,Y_data)

#This functions create samples and saves it in a filename:
def Create_GP_Data_File_2d(filename,n_samples,min_x,max_x,n_grid_points,l_scale=1,sigma_var=1, 
                        kernel_type="div_free",obs_noise=1e-2):
    #Sample data:
    X_data,Y_data=Cyclic_GP_Sampler(n_samples=n_samples,min_x=min_x,max_x=max_x,n_grid_points=n_grid_points,l_scale=l_scale,sigma_var=sigma_var, 
                        kernel_type=kernel_type,obs_noise=obs_noise)
    #Save the numpy array:
    np.save('Tasks/GP_Data/GP_div_free_circle/Data/'+filename+'_X', X_data.numpy())
    np.save('Tasks/GP_Data/GP_div_free_circle/Data/'+filename+'_Y', Y_data.numpy())

MIN_X=-10
MAX_X=10
N_GRID_POINTS=30
L_SCALE=5
SIGMA_VAR=10. 
KERNEL_TYPE="div_free"
OBS_NOISE=0.02
N_TRAIN_SAMPLES=40000
N_VAL_SAMPLES=10000
N_TEST_SAMPLES=10000
TRAIN_FILENAME='GP_Circle_Train'
VAL_FILENAME='GP_Circle_Valid'
TEST_FILENAME='GP_Circle_Test'

#Create train data:
Create_GP_Data_File_2d(filename=TRAIN_FILENAME,n_samples=N_TRAIN_SAMPLES,min_x=MIN_X,max_x=MAX_X,n_grid_points=N_GRID_POINTS,l_scale=L_SCALE,sigma_var=SIGMA_VAR,kernel_type=KERNEL_TYPE,
                       obs_noise=OBS_NOISE)
#Create validation data:
Create_GP_Data_File_2d(filename=VAL_FILENAME,n_samples=N_VAL_SAMPLES,min_x=MIN_X,max_x=MAX_X,n_grid_points=N_GRID_POINTS,l_scale=L_SCALE,sigma_var=SIGMA_VAR,kernel_type=KERNEL_TYPE,
                       obs_noise=OBS_NOISE)
#Create test data:
Create_GP_Data_File_2d(filename=TEST_FILENAME,n_samples=N_TEST_SAMPLES,min_x=MIN_X,max_x=MAX_X,n_grid_points=N_GRID_POINTS,l_scale=L_SCALE,sigma_var=SIGMA_VAR,kernel_type=KERNEL_TYPE,
                       obs_noise=OBS_NOISE)

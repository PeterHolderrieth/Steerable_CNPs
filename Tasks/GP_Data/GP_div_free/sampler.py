#LIBRARIES:
#Tensors:
import torch
import numpy as np

#Tools:
import datetime
import sys

sys.path.append("./././")

#Own files:
import Kernel_and_GP_tools as GP
import My_Tools
from Tasks.GP_Data.GP_sampler import Cyclic_GP_Sampler

#This functions create samples and saves it in a filename:
def Create_GP_Data_File_2d(filename,n_samples,min_x,max_x,n_grid_points,l_scale=1,sigma_var=1, 
                        kernel_type="div_free",obs_noise=1e-2):
    #Sample data:
    X_data,Y_data=Cyclic_GP_Sampler(n_samples=n_samples,min_x=min_x,max_x=max_x,n_grid_points=n_grid_points,l_scale=l_scale,sigma_var=sigma_var, 
                        kernel_type=kernel_type,obs_noise=obs_noise)
    #Save the numpy array:
    np.save('Tasks/GP_Data/GP_div_free/Data/'+filename+'_X', X_data.numpy())
    np.save('Tasks/GP_Data/GP_div_free/Data/'+filename+'_Y', Y_data.numpy())

MIN_X=-10
MAX_X=10
N_GRID_POINTS=30
L_SCALE=5
SIGMA_VAR=10. 
KERNEL_TYPE="div_free"
OBS_NOISE=0.02
N_TRAIN_SAMPLES=80000
N_VAL_SAMPLES=20000
N_TEST_SAMPLES=20000
TRAIN_FILENAME='GP_div_free_Train'
VAL_FILENAME='GP_div_free_Valid'
TEST_FILENAME='GP_div_free_Test'

#Create train data:
Create_GP_Data_File_2d(filename=TRAIN_FILENAME,n_samples=N_TRAIN_SAMPLES,min_x=MIN_X,max_x=MAX_X,n_grid_points=N_GRID_POINTS,l_scale=L_SCALE,sigma_var=SIGMA_VAR,kernel_type=KERNEL_TYPE,
                       obs_noise=OBS_NOISE)
#Create validation data:
Create_GP_Data_File_2d(filename=VAL_FILENAME,n_samples=N_VAL_SAMPLES,min_x=MIN_X,max_x=MAX_X,n_grid_points=N_GRID_POINTS,l_scale=L_SCALE,sigma_var=SIGMA_VAR,kernel_type=KERNEL_TYPE,
                       obs_noise=OBS_NOISE)
#Create test data:
Create_GP_Data_File_2d(filename=TEST_FILENAME,n_samples=N_TEST_SAMPLES,min_x=MIN_X,max_x=MAX_X,n_grid_points=N_GRID_POINTS,l_scale=L_SCALE,sigma_var=SIGMA_VAR,kernel_type=KERNEL_TYPE,
                       obs_noise=OBS_NOISE)

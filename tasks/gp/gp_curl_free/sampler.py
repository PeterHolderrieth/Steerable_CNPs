#LIBRARIES:
#Tensors:
import torch
import numpy as np

#Tools:
import datetime
import sys

sys.path.append("./././")

#Own files:
import kernel_and_gp_tools as GP
import my_utils
from tasks.gp.gp_sampler import cyclic_gp_sampler

#This functions create samples and saves it in a filename:
def create_gp_file_2d(filename,n_samples,min_x,max_x,n_grid_points,l_scale=1,sigma_var=1, 
                        kernel_type="curl_free",obs_noise=1e-2):
    #Sample data:
    X_data,Y_data=cyclic_gp_sampler(n_samples=n_samples,min_x=min_x,max_x=max_x,n_grid_points=n_grid_points,l_scale=l_scale,sigma_var=sigma_var, 
                        kernel_type=kernel_type,obs_noise=obs_noise)
    #Save the numpy array:
    np.save('tasks/gp/gp_curl_free/data/'+filename+'_X', X_data.numpy())
    np.save('tasks/gp/gp_curl_free/data/'+filename+'_Y', Y_data.numpy())

MIN_X=-10
MAX_X=10
N_GRID_POINTS=30
L_SCALE=5
SIGMA_VAR=10. 
KERNEL_TYPE="curl_free"
OBS_NOISE=0.02
N_TRAIN_SAMPLES=80000
N_VAL_SAMPLES=20000
N_TEST_SAMPLES=20000
TRAIN_FILENAME='gp_curl_free_Train'
VAL_FILENAME='gp_curl_free_Valid'
TEST_FILENAME='gp_curl_free_Test'

#Create train data:
create_gp_file_2d(filename=TRAIN_FILENAME,n_samples=N_TRAIN_SAMPLES,min_x=MIN_X,max_x=MAX_X,n_grid_points=N_GRID_POINTS,l_scale=L_SCALE,sigma_var=SIGMA_VAR,kernel_type=KERNEL_TYPE,
                       obs_noise=OBS_NOISE)
#Create validation data:
create_gp_file_2d(filename=VAL_FILENAME,n_samples=N_VAL_SAMPLES,min_x=MIN_X,max_x=MAX_X,n_grid_points=N_GRID_POINTS,l_scale=L_SCALE,sigma_var=SIGMA_VAR,kernel_type=KERNEL_TYPE,
                       obs_noise=OBS_NOISE)
#Create test data:
create_gp_file_2d(filename=TEST_FILENAME,n_samples=N_TEST_SAMPLES,min_x=MIN_X,max_x=MAX_X,n_grid_points=N_GRID_POINTS,l_scale=L_SCALE,sigma_var=SIGMA_VAR,kernel_type=KERNEL_TYPE,
                       obs_noise=OBS_NOISE)

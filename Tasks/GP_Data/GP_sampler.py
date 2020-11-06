#LIBRARIES:
#Tensors:
import torch
import numpy as np

#Tools:
import datetime
import sys

#sys.path.append("././")

#Own files:
import Kernel_and_GP_tools as GP
import My_Tools

#This functions create n_samples of a GP on a radial grid:
def Cyclic_GP_Sampler(n_samples,min_x,max_x,n_grid_points,l_scale=1.,sigma_var=1., 
                        kernel_type="div_free",obs_noise=1e-2):
    '''
    Input:
    n_samples - int - number of samples to compute 
    min_x,max_x - minimum und maximum x-axis section to compute radial grid 
    n_grid_points - number of grid points per axis (full grid)
    l_scale - length scale for kernel
    sigma_var - sigma parameter for kernel
    kernel_type - type of kernel considered (see Kernel_and_GP_tools.py)
    obs_noise - observation noise to be added 
    Output:
    X_data - torch.Tensor - shape (n_samples,number of points per sample,2) - n_samples samples of a GP 
                                                                              sampled on a circle with kernel 
                                                                              of type specified in kernel_type

    '''
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
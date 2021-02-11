
#Tensors:
import numpy as np
import math
import xarray
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import sys
from datetime import datetime
from datetime import timedelta

sys.path.append("../../")

import my_utils

#Set default as double:
torch.set_default_dtype(torch.float)

'''
A data set class to deal with the ERA5 weather data set.
'''
class ERA5Dataset(utils.dataset):
    def __init__(self, path_to_nc_file,Min_n_cont,Max_n_cont,circular=True, normalize=True,place='US'):
        '''
        path_to_nc_file - string - gives filepath to a netCDF file which can be loaded as an xarray dataset
                                   having index "datetime","Longitude","Latitude" and the data variables
                                   ['sp_in_kPa','t_in_Cels','wind_10m_east','wind_10m_north']
        Min_n_cont,Max_n_cont,n_total - int - minimum and maximum number of context points and the number of total points per sample
        var_names - list of strings - gives names of variables which are supposed to be in the dataset, if None then all variables are used
        '''
        super(ERA5Dataset, self).__init__()
        #Load the data as an xarray:
        self.Y_data=xarray.open_dataset(path_to_nc_file).to_array()

        #Save variables list:
        self.variables=list(self.Y_data.coords['variable'].values)
        #Save the number of variables:
        self.n_variables=len(self.variables)
        #Save the indices for the wind variables if they are in the list of variables:
        self.ind_wind_10=self.give_index_for_var(['wind_10m_east','wind_10m_north'])

        #Transpose the data and get the number of observations:
        self.Y_data=self.Y_data.transpose("datetime","Longitude","Latitude","variable")
        self.n_obs=self.Y_data.shape[0]

        self.Longitude=torch.tensor(self.Y_data.coords['Longitude'].values,dtype=torch.get_default_dtype())
        self.Latitude=torch.tensor(self.Y_data.coords['Latitude'].values,dtype=torch.get_default_dtype())

        if self.Latitude.size(0)!=self.Longitude.size(0):
            sys.exit("The number of grid values are not the same for Longitude and Latitude.")
        else:
            self.n_per_axis=self.Longitude.size(0)
            self.n_points_per_obs=self.n_per_axis**2
        
        self.X_tensor=torch.stack([self.Longitude.repeat_interleave(self.n_per_axis),self.Latitude.repeat(self.n_per_axis)],dim=1)#.view(self.n_per_axis,self.n_per_axis,2)

        self.variables=list(self.Y_data.coords['variable'].values)

        self.translater=ERA5_translater(place=place)

        self.Min_n_cont=Min_n_cont
        self.Max_n_cont=Max_n_cont
        self.normalize=normalize
        self.circular=circular

        if self.circular:
            self.circular_indices=utils.get_inner_circle_indices(self.n_per_axis,flat=True)
        
        
        #Control inputs:    
        if not isinstance(self.Min_n_cont,int) or not isinstance(self.Max_n_cont,int) or self.Min_n_cont>self.Max_n_cont\
            or self.Min_n_cont<2:
            print("Error: Combination of minimum and maximum number of context points and number of total points not compatible.")

        if self.n_obs!=len(self.Y_data.coords['datetime'].values):
            sys.exit("Error: Coordinates of datetime do not match.")
        
        self.print_report()

    def compute_normalization(self):
        #Compute the mean for X:
        X_mean=self.X_tensor.mean(dim=0)

        #Compute the mean for the various components (not correct for wind components):
        Y_mean=torch.tensor(self.Y_data.mean(dim=['datetime','Longitude','Latitude']).values,dtype=torch.get_default_dtype())

        #Compute the standard deviation for the data (not correct for wind components!):
        Y_std=torch.tensor(self.Y_data.std(dim=['datetime','Longitude','Latitude']).values,dtype=torch.get_default_dtype())
        
        #Correct normalizing for the wind components:
        Y_mean[self.ind_wind_10]=torch.tensor([0.,0.],dtype=torch.get_default_dtype())
        mean_norm_10m=np.linalg.norm(self.Y_data.loc[:,:,:,['wind_10m_east','wind_10m_north']].values,axis=3).mean()
        Y_std[self.ind_wind_10]=torch.tensor(mean_norm_10m,dtype=torch.get_default_dtype())

        return(X_mean,Y_mean,Y_std)

    def print_report(self):
        print()
        print("_______________________")
        print("Loaded weather data: ")
        print("-----")
        print("Shape: ", self.Y_data.shape)
        print("Variables: ", self.variables)
        print("West-East Longtitude: ", self.Longitude[0].item(),self.Longitude[-1].item())
        print("South-North Latitude: ",self.Latitude[0].item(),self.Latitude[-1].item())
        print("Grid points per map: ", self.n_points_per_obs)
        print("________________________")
        print()

    def give_index_for_var(self,var_names):
        '''
        Input: var_names - list of strings - names of variables whose index is to return
        Output: list of ints - giving index of variables in self.variables
        '''
        return([self.variables.index(name) for name in var_names])
    
    def rand_rot_mat(self):
        '''
        Output: torch.Tensor - shape (2,2) - a random rotation matrix
        '''
        alpha=2*math.pi*np.random.uniform()
        R=torch.tensor([[math.cos(alpha),-math.sin(alpha)],[math.sin(alpha),math.cos(alpha)]])
        return(R)

    def rand_transform(self,X,Y):
        '''
        Input: X,Y - torch.Tensor - shape (n,2), (n,4)
        Output: X,Y - torch.Tensor - shape (n,2), (n,4) - randomly rotated X and rotated Y 
        (only the wind components of Y are transformed, the scalar values )
        '''
        #Sample a random rotation matrix:
        R=self.rand_rot_mat()
        
        #Rotate coordinates:
        X=torch.matmul(X-self.X_mean[None,:],R.t())+self.X_mean[None,:]

        #Rotate wind components:
        Y[:,self.ind_wind_10]=torch.matmul(Y[:,self.ind_wind_10],R.t())
        return(X,Y)

    #This is the basis function returning maps to plotting and purposes which do not include training the pytorch model:
    def get_map(self,ind,transform=False):
        '''
        Input:  ind - int - index to get
                transform - Boolean - indicates whether a random transformation is performed
        Output: X,Y - torch.Tensor - shape (n,2),(n,self.n_variables)
        '''
        Y=torch.tensor(self.Y_data[ind].values,dtype=torch.get_default_dtype())
        Y=Y.view(-1,self.n_variables)
        if self.circular:
            X=self.X_tensor[self.circular_indices]
            Y=Y[self.circular_indices]
        shuffle_ind=torch.randperm(n=X.size(0))
        X=X[shuffle_ind]
        Y=Y[shuffle_ind]
        if transform:
            X,Y=self.rand_transform(X,Y)
        return(X,Y)

    def get_rand_map(self,transform=False):
        ind=torch.randint(low=0,high=self.n_obs,size=[1]).item()
        return(self.get_map(ind,transform=transform))

    
    #Function which returns random batches for training:
    def get_batch(self,inds,transform=False,n_context_points=None,cont_in_target=False):
        '''
        Input: `inds - torch.Tensor of ints - indices to choose batch from
                transform - Boolean - indicates whether a random transformation is performed
        Output: X_c,Y_c - torch.Tensor - shape (batch_size,n_context_points,2/self.n_variables)
                X_t,Y_t - torch.Tensor - shape (batch_size,n_target_points,2/self.n_variables) if cont_in_target is False
                                               (batch_size,n_target_points+n_context_points,2/self.n_variables) if cont_in_target is True

        '''
        X_list,Y_list=zip(*[self.get_map(ind=ind,transform=transform) for ind in inds])
        X=torch.stack(X_list,dim=0)
        Y=torch.stack(Y_list,dim=0)
        if self.normalize:
            X,Y=self.translater.translate_to_normalized_scale(X,Y)
        if n_context_points is None:
            n_context_points=np.random.randint(low=self.Min_n_cont,high=self.Max_n_cont)    
        if cont_in_target:
            return(X[:,:n_context_points],Y[:,:n_context_points],X,Y[:,:,[2,3]])
        else:
            return(X[:,:n_context_points],Y[:,:n_context_points],X[:,n_context_points:],Y[:,n_context_points:,[2,3]])
    
    def get_rand_batch(self,batch_size,transform=False,n_context_points=None,cont_in_target=False):
        '''
        Returns self.get_batch with random number of indices with length=batch_size and random number of context points
        in range [self.Min_n_cont,high=self.Max_n_cont]
        If n_context_points is None, it is randomly sampled.
        '''
        inds=torch.randperm(self.n_obs)[:batch_size]
        return(self.get_batch(inds=inds,transform=transform,n_context_points=n_context_points,cont_in_target=cont_in_target))

'''
We write a class which get an input X,Y and translates it normalized values
By normalized, we mean here:
    - the mean of X is 0 and the radius 10, there are two types to convert X, first
    from the US grid to [-10,10] or from the China grid to [-10,10]
    - the values of Y are also transformed (scalar values are centered and rescaled while vectors 
    are only rescaled)
'''
class ERA5_translater(object):
    def __init__(self, place='US'):
        self.place=place
        if self.place=='US':
            self.X_mean=torch.tensor([-91.,35.],dtype=torch.get_default_dtype())
        elif self.place=='China':
            self.X_mean=torch.tensor([110.,30.],dtype=torch.get_default_dtype())
        else:
            sys.exit("Unknown place.")
        
        self.Y_mean=torch.tensor([100.1209,   7.4628,   0.0000,   0.0000],dtype=torch.get_default_dtype())
        self.Y_std=torch.tensor([1.4738, 8.5286, 3.4162, 3.4162],dtype=torch.get_default_dtype())
        self.Y_mean_out=torch.tensor([0.0000,   0.0000],dtype=torch.get_default_dtype())
        self.Y_std_out=torch.tensor([3.4162, 3.4162],dtype=torch.get_default_dtype())
        
    def norm_X(self,X):
        '''
        X - torch.Tensor - shape (*,2)
        --> returns torch.Tensor with same shape translated and scaled with mean and std for self.X_Tensor
        '''
        return(2*X.sub(self.X_mean[None,:]))

    def denorm_X(self,X):
        '''
        X - torch.Tensor - shape (*,2)
        --> Inverse of self.norm_X
        '''
        return(X.div(2).add(self.X_mean[None,:]))
    
    def norm_Y(self,Y):
        '''
        Y - torch.Tensor - shape (*,4)
        --> returns torch.Tensor with same shape translated and scaled with mean and std for self.Y_data
        '''
        return(Y.sub(self.Y_mean[None,:]).div(self.Y_std[None,:]))

    def denorm_Y(self,Y):
        '''
        Y - torch.Tensor - shape (*,4)
        --> Inverse of self.norm_Y
        '''
        return(Y.mul(self.Y_std[None,:]).add(self.Y_mean[None,:]))
    
    def denorm_Y_out(self,Y):
        '''
        Y - torch.Tensor - shape (*,4)
        --> Inverse of self.norm_Y
        '''
        return(Y.mul(self.Y_std_out[None,:]).add(self.Y_mean_out[None,:]))
    
    def translate_to_normalized_scale(self,X,Y):
        '''
        X - torch.Tensor - shape (*,2)
        Y - torch.Tensor - shape (*,4)
        --> returns torch.Tensor with same shape translated and scaled with mean and std
        '''
        return(self.norm_X(X),self.norm_Y(Y))
    
    def translate_to_original_scale(self,X,Y):
        '''
        X - torch.Tensor - shape (*,2)
        Y - torch.Tensor - shape (*,4)
        --> returns torch.Tensor with same shape translated and scaled with mean and std
        '''
        return(self.denorm_X(X),self.denorm_Y(Y))

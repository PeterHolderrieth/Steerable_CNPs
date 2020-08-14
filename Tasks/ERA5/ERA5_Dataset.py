
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

#Set default as double:
torch.set_default_dtype(torch.float)

class ERA5Dataset(utils.IterableDataset):
    def __init__(self, path_to_nc_file,Min_n_cont,Max_n_cont,n_total=None,var_names=None, normalize=False):
        '''
        path_to_nc_file - string - gives filepath to a netCDF file which can be loaded as an xarray dataset
                                   having index "datetime","Longitude","Latitude" and the data variables
                                   ['height_in_m','sp_in_kPa','t_in_Cels','wind_10m_east','wind_10m_north',
                                    'wind_100m_east', 'wind_100m_north']
        Min_n_cont,Max_n_cont,n_total - int - minimum and maximum number of context points and the number of total points per sample
        var_names - list of strings - gives names of variables which are supposed to be in the dataset, if None then all variables are used
        '''
        #Load the data as an xarray:
        self.Y_data=xarray.open_dataset(path_to_nc_file).to_array()
        #If only certain variable names are wanted, extract them:
        if var_names is not None:
            #Save list of current variables:
            self.variables=list(self.Y_data.coords['variable'].values)
            #Get index of new variables in old variable array:
            ind=self.give_index_for_var(var_names)
            #Extract:
            self.Y_data=self.Y_data[ind]
        
        #Save variables list:
        self.variables=list(self.Y_data.coords['variable'].values)
        #Save the number of variables:
        self.n_variables=len(self.variables)
        #Save the indices for the wind variables if they are in the list of variables:
        try:
            self.ind_wind_10=self.give_index_for_var(['wind_10m_east','wind_10m_north'])
        except ValueError:
            self.ind_wind_10=None
        try:
            self.ind_wind_100=self.give_index_for_var(['wind_100m_east','wind_100m_north'])
        except ValueError:
            self.ind_wind_100=None

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
        
        self.X_tensor=torch.stack([self.Longitude.repeat_interleave(self.n_per_axis),self.Latitude.repeat(self.n_per_axis)],dim=1)

        self.variables=list(self.Y_data.coords['variable'].values)

        self.n_total=self.n_points_per_obs if n_total is None else n_total
        self.Min_n_cont=Min_n_cont
        self.Max_n_cont=Max_n_cont
        self.normalize=normalize

        #Compute the mean and the standard deviation for X
        #self.X_std=self.X_tensor.std(dim=0,unbiased=False)
        self.X_mean=self.X_tensor.mean(dim=0)

        #Compute the mean for the various components (not correct for wind components):
        self.Y_mean=torch.tensor(self.Y_data.mean(dim=['datetime','Longitude','Latitude']).values,dtype=torch.get_default_dtype())
        #Compute the standard deviation for the data (not correct for wind components!):
        self.Y_std=torch.tensor(self.Y_data.std(dim=['datetime','Longitude','Latitude']).values,dtype=torch.get_default_dtype())
        #Correct normalizing for the wind components:
        if self.ind_wind_10 is not None:
            self.Y_mean[self.ind_wind_10]=torch.tensor([0.,0.],dtype=torch.get_default_dtype())
            mean_norm_10m=np.linalg.norm(self.Y_data.loc[:,:,:,['wind_10m_east','wind_10m_north']].values,axis=3).mean()
            self.Y_std[self.ind_wind_10]=torch.tensor(mean_norm_10m,dtype=torch.get_default_dtype())
        
        if self.ind_wind_100 is not None:
            self.Y_mean[self.ind_wind_100]=torch.tensor([0.,0.],dtype=torch.get_default_dtype())
            mean_norm_100m=np.linalg.norm(self.Y_data.loc[:,:,:,['wind_100m_east','wind_100m_north']].values,axis=3).mean()
            self.Y_std[self.ind_wind_100]=torch.tensor(mean_norm_100m,dtype=torch.get_default_dtype())

        if not isinstance(self.Min_n_cont,int) or not isinstance(self.Max_n_cont,int) or self.Min_n_cont>self.Max_n_cont\
            or self.Min_n_cont<2 or self.Max_n_cont>self.n_total or self.n_total>self.n_points_per_obs:
            print("Error: Combination of minimum and maximum number of context points and number of total points not compatible.")

        if self.n_obs!=len(self.Y_data.coords['datetime'].values):
            sys.exit("Error: Coordinates of datetime do not match.")
        
        self.print_report()

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
    def norm_X(self,X):
        '''
        X - torch.Tensor - shape (*,2)
        --> returns torch.Tensor with same shape translated and scaled with mean and std for self.X_Tensor
        '''
        return(X.sub(self.X_mean[None,:]).div(self.X_std[None,:]))

    def denorm_X(self,X):
        '''
        X - torch.Tensor - shape (*,2)
        --> Inverse of self.norm_X
        '''
        return(X.mul(self.X_std[None,:]).add(self.X_mean[None,:]))
    
    def norm_Y(self,Y):
        '''
        Y - torch.Tensor - shape (*,2)
        --> returns torch.Tensor with same shape translated and scaled with mean and std for self.Y_data
        '''
        return(Y.sub(self.Y_mean[None,:]).div(self.Y_std[None,:]))

    def denorm_Y(self,Y):
        '''
        Y - torch.Tensor - shape (*,2)
        --> Inverse of self.norm_Y
        '''
        return(Y.mul(self.Y_std[None,:]).add(self.Y_mean[None,:]))

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
        Input: X,Y - torch.Tensor - shape (n,2), (n,7)
        Output: X,Y - torch.Tensor - shape (n,2), (n,7) - randomly rotated X and rotated Y 
        (only the wind components of Y are transformed, the scalar values )
        '''
        #Sample a random rotation matrix:
        R=self.rand_rot_mat()
        
        #Rotate coordinates:
        X=torch.matmul(X-self.X_mean[None,:],R.t())+self.X_mean[None,:]

        #Rotate wind components:
        if self.ind_wind_10 is not None:
            Y[:,self.ind_wind_10]=torch.matmul(Y[:,self.ind_wind_10],R.t())
        if self.ind_wind_100 is not None:
            Y[:,self.ind_wind_100]=torch.matmul(Y[:,self.ind_wind_100],R.t())
        return(X,Y)

    #This is the basis function returning maps to plotting and purposes which do not include training the pytorch model:
    def get_rand_map(self,transform=False):
        '''
        Input: transform - Boolean - indicates whether a random transformation is performed
        Output: X,Y - torch.Tensor - shape (n,2),(n,self.n_variables)
        '''
        ind=torch.randint(low=0,high=self.n_obs,size=[1]).item()
        Y=torch.tensor(self.Y_data[ind].values,dtype=torch.get_default_dtype())
        Y=Y.view(-1,self.n_variables)
        shuffle_ind=torch.randperm(n=self.n_points_per_obs)
        X=self.X_tensor[shuffle_ind]
        Y=Y[shuffle_ind]
        if transform:
            X,Y=self.rand_transform(X,Y)
        return(X,Y)
    #Function which returns random batches for training:
    def get_rand_batch(self,batch_size,transform=False,n_context_points=None,cont_in_target=False):
        '''
        Input: transform - Boolean - indicates whether a random transformation is performed
        Output: X_c,Y_c - torch.Tensor - shape (batch_size,n_context_points,2/self.n_variables)
                X_t,Y_t - torch.Tensor - shape (batch_size,n_target_points,2/self.n_variables) if cont_in_target is False
                                               (batch_size,n_target_points+n_context_points,2/self.n_variables) if cont_in_target is True

        '''
        X_list,Y_list=zip(*[self.get_rand_map(transform=transform) for i in range(batch_size)])
        X=torch.stack(X_list,dim=0)
        Y=torch.stack(Y_list,dim=0)
        if self.normalize:
            Y=self.norm_Y(Y)
        if n_context_points is None:
            n_context_points=np.random.randint(low=self.Min_n_cont,high=self.Max_n_cont)    
        if cont_in_target:
            return(X[:,:n_context_points],Y[:,:n_context_points],X,Y)
        else:
            return(X[:,:n_context_points],Y[:,:n_context_points],X[:,n_context_points:],Y[:,n_context_points:])

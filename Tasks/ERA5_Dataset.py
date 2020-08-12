
#Tensors:
import numpy as np
import xarray
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import sys
from datetime import datetime
from datetime import timedelta


class ERA5Dataset(utils.IterableDataset):
    def __init__(self, path_to_nc_file,Min_n_cont,Max_n_cont,n_total=None):
        #Load the data and permute columns:
        self.Y_data=xarray.open_dataset(path_to_nc_file).to_array()
        print("Start transpose.")
        self.Y_data=self.Y_data.transpose("datetime","Longitude","Latitude","variable")
        print("Finished transpose.")
        self.n_obs=self.Y_data.shape[0]
        print(self.Y_data.shape)
        if self.n_obs!=len(self.Y_data.coords['datetime'].values):
            sys.exit("Error: Coordinates of datetime do not match.")

        self.Longitude=torch.tensor(self.Y_data.coords['Longitude'].values,dtype=torch.get_default_dtype())
        self.Latitude=torch.tensor(self.Y_data.coords['Latitude'].values,dtype=torch.get_default_dtype())
        self.n_grid_val=self.Longitude.size(0)**2
        if self.Latitude.size(0)**2!=self.n_grid_val:
            sys.exit("The number of grid values are not the same for Longitude and Latitude.")
    
        self.X_tensor=torch.stack([self.Longitude.repeat_interleave(self.n_grid_val),self.Latitude.repeat(self.n_grid_val)],dim=1)

        self.n_total=self.n_grid_val if n_total is None else n_total
        self.Min_n_cont=Min_n_cont
        self.Max_n_cont=Max_n_cont



    def get_rand_pair(self,transform=True):
        ind=torch.randint(low=0,high=self.n_obs,size=[1]).item()
        Y=torch.tensor(self.Y_data[ind].values,dtype=torch.get_default_dtype())
        Y=Y.view(-1,7)[:,[3,4]]
        shuffle_ind=torch.randperm(n=self.n_grid_val)
        X=self.X_tensor[shuffle_ind]
        Y=Y[shuffle_ind]
        if transform:
            pass
        return(X,Y)
    
    def get_rand_batch(self,batch_size,transform=True,n_context_points=None,cont_in_target=False):
        X_list,Y_list=zip(*[self.get_rand_pair(transform=transform) for i in range(batch_size)])
        X=torch.stack(X_list,dim=0)
        Y=torch.stack(Y_list,dim=0)
        if n_context_points is None:
            n_context_points=np.random.randint(low=self.Min_n_cont,high=self.Max_n_cont)    
        if cont_in_target:
            return(X[:,:n_context_points],Y[:,:n_context_points],X,Y)
        else:
            return(X[:,:n_context_points],Y[:,:n_context_points],X[:,n_context_points:],Y[:,n_context_points:])
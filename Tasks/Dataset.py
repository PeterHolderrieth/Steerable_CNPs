
#Tensors:
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from datetime import datetime
from datetime import timedelta

class CNPDataset(utils.IterableDataset):
    def __init__(self, X,Y,Min_n_cont,Max_n_cont,n_total):
        '''
        X - torch.Tensor - shape (N,n,d) - N...number of observations (size of data set), 
                                           n...number of data pairs per observations
                                           d...dimension of input space
        Y - torch.Tensor -shape (N,n,2) - D...dimension of output space
        Min_n_cont,Max_n_cont - int - minimum and maximum number of context points
        n_total - int - total number of points per sample (target+context)
        '''
        self.X_data=X
        self.Y_data=Y

        self.dim_0,self.dim_1,self.dim_2_X=X.size()
        self.dim_2_Y=Y.size(2)

        self.init_shuffle()

        self.Min_n_cont=Min_n_cont
        self.Max_n_cont=Max_n_cont
        self.n_total=n_total if n_total is not None else self.dim_1
        
        if not isinstance(self.X_data,torch.Tensor) or not isinstance(self.Y_data,torch.Tensor):
            sys.exit("Input is not a tensor.")
        if len(self.X_data.shape)!=3 or len(self.X_data.shape)!=3:
            sys.exit("Input has wrong number dimension - need to be of shape (N,n,d).")
        if self.X_data.size(0)!=self.Y_data.size(0) or self.X_data.size(1)!=self.Y_data.size(1):
            sys.exit("X and Y are not compatible - have different shape.")

    def init_shuffle(self):
        '''
        Permutes/shuffles the observations (so dimension 0)
        and permute/shuffles the order of data pairs per observations (so dimension 1)
        '''
        shuffle=torch.randperm(self.__len__())
        self.X_data=self.X_data[shuffle]
        self.Y_data=self.Y_data[shuffle]
                
        for it in range(self.__len__()):
            shuffle=torch.randperm(self.dim_1)
            self.X_data[it]=self.X_data[it][shuffle]
            self.Y_data[it]=self.Y_data[it][shuffle]
            
    def __len__(self):
            return len(self.X_data)
    
    def get_batch(self,inds,n_context_points,cont_in_target=False):
        '''
        Input: inds - list of ints - gives indices of which observations to choose for minibatch
               n_context_points - int - gives number of context points
               cont_in_target -Boolean - if True, the target set includes the context set
        Ouput: X_context,Y_context - torch.Tensor - shape (len(inds),n_context_points,self.dim_2_X/self.dim_2_Y)
               X_target, Y_target - torch.Tensor - shape (len(inds),n_total-n_context_points,self.dim_2_X/self.dim_2_Y)
        '''
        shuffle=torch.randperm(self.dim_1)
        X=self.X_data[inds][:,shuffle[:self.n_total]]
        Y=self.Y_data[inds][:,shuffle[:self.n_total]]
        if cont_in_target:
            return(X[:,:n_context_points],Y[:,:n_context_points],X,Y)
        else:
            return(X[:,:n_context_points],Y[:,:n_context_points],X[:,n_context_points:],Y[:,n_context_points:])
    
    def get_rand_batch(self,batch_size,n_context_points=None,cont_in_target=False):
        '''
        Returns self.get_batch with random number of indices with length=batch_size and random number of context points
        in range [self.Min_n_cont,high=self.Max_n_cont]
        If n_context_points is None, it is randomly sampled.
        '''
        inds=torch.randperm(self.dim_0)[:batch_size]
        if n_context_points is None:
            n_context_points=torch.randint(low=self.Min_n_cont,high=self.Max_n_cont,size=[1])
        return(self.get_batch(inds,n_context_points,cont_in_target=cont_in_target))

class ERA5WindDataset(utils.IterableDataset):
    def __init__(self, path_to_folder,grid_file,file_without_time,Min_n_cont,Max_n_cont,n_total,min_year,max_year,months=[1,2,12]):
        '''
        path_to_folder - string - file path to the folder containing the grid file and the data files
        grid_file - string ending with .pickle - filename of the grid file - this must be a pickle file
        file_without_time - string - "%Y_%m_%d_%H"+file_without_time should give files containing the features of the grid
        Min_n_cont,Max_n_cont, n_total - int - minimum and maximum number of context points, total number of points per sample
        min_year,max_year - int - range of years to sample from
        months - list of ints - containing the months 
        '''

        self.path_to_folder=path_to_folder
        self.grid_file=grid_file
        self.file_without_time=file_without_time
        self.min_year=min_year
        self.max_year=max_year
        self.Min_n_cont=Min_n_cont
        self.Max_n_cont=Max_n_cont
        self.n_total=n_total 
        self.months=months

        #Load the grid:
        self.grid_df=pd.read_pickle(self.path_to_folder+self.grid_file)
        self.grid_tensor=torch.tensor(self.grid_df.values,dtype=torch.get_default_dtype())

        self.n_grid=self.grid_tensor.size(0)
    
    def get_string_from_time_object(self,time):
        return(time.strftime(format=("%Y_%m_%d_%H")))
    
    def sample_rand_time(self):
        year=np.random.randint(low=self.min_year,high=self.max_year+1)
        month=self.months[np.random.randint(low=0,high=len(self.months))]
        day=np.random.randint(low=1,high=29)
        hour=np.random.randint(low=0,high=24)
        return(datetime(year,month,day,hour))
    
    def get_item(self,time):
        filename=self.path_to_folder+self.get_string_from_time_object(time)+\
                    self.file_without_time
        return(pd.read_pickle(filename))

    def rand_get_item(self):
        time=self.sample_rand_time()
        return(self.get_item(time))
    
    def get_rand_pair(self,transform=True):
        Y=torch.tensor(self.rand_get_item().values,dtype=torch.get_default_dtype())
        shuffle_ind=torch.randperm(n=self.n_grid)
        X=self.grid_tensor[shuffle_ind]
        Y=Y[shuffle_ind]
        if transform:
            pass
        return(X,Y)

    def get_rand_batch(self,batch_size,n_context_points=None,cont_in_target=False):
        '''
        times - list of datetime.datetime objects - times to sample
        '''
        X_list,Y_list=zip(*[self.get_rand_pair() for i in range(batch_size)])
        X=torch.stack(X_list,dim=0)
        Y=torch.stack(Y_list,dim=0)
        if n_context_points is None:
            n_context_points=np.random.randint(low=0,high=X.size(1))    
        if cont_in_target:
            return(X[:,:n_context_points],Y[:,:n_context_points],X,Y)
        else:
            return(X[:,:n_context_points],Y[:,:n_context_points],X[:,n_context_points:],Y[:,n_context_points:])
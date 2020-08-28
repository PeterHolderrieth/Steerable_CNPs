#Tensors:
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from datetime import datetime
from datetime import timedelta

class GPDataset(utils.IterableDataset):
    def __init__(self, X,Y,Min_n_cont,Max_n_cont,n_total,transform=True):
        '''
        X - torch.Tensor - shape (N,n,d) - N...number of observations (size of data set), 
                                           n...number of data pairs per observations
                                           d...dimension of input space
        Y - torch.Tensor -shape (N,n,2) - D...dimension of output space
        Min_n_cont,Max_n_cont - int - minimum and maximum number of context points
        n_total - int - total number of points per sample (target+context)
        transform - Bool - indicates whether random rotation is applied
        '''
        self.X_data=X
        self.Y_data=Y

        self.dim_0,self.dim_1,self.dim_2_X=X.size()
        self.dim_2_Y=Y.size(2)

        self.init_shuffle()

        self.Min_n_cont=Min_n_cont
        self.Max_n_cont=Max_n_cont
        self.n_total=n_total if n_total is not None else self.dim_1
        self.transform=transform
        
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
        
    def rand_orthog_mat(self):
        '''
        Output: torch.Tensor - shape (2,2) - a random orthogonal matrix
        '''
        alpha=2*math.pi*np.random.uniform()
        s=np.random.choice([-1,1])
        R=torch.tensor([[math.cos(alpha),-s*math.sin(alpha)],[math.sin(alpha),s*math.cos(alpha)]])
        return(R)

    def rand_transform(self,X,Y):
        '''
        Input: X,Y - torch.Tensor - shape (batch_size,n,2)
        Output: X,Y - torch.Tensor - shape (batch_size,n,2) - randomly roto-reflected X and roto-reflected Y 
        '''
        #Sample a random rotation matrix:
        R=self.rand_orthogonal_mat()
        
        #Return rotated versions:
        return(torch.matmul(X,R.t()),torch.matmul(Y,R.t()))
    
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
        if self.transform:
            X,Y=self.rand_transform(X,Y)
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

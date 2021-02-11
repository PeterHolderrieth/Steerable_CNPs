import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

import numpy as np
import math
import sys
import time

class CNPEncoder(nn.Module):
    def __init__(self, dim_X, dim_Y, hidden_layers, dim_R):
        super(CNPEncoder, self).__init__()
        '''
        dim_X: int
               Dimension of the state space
        dim_Y: int
               Dimension of the output space
        hidden_layers: list of integers where the length of the list is the number of hidden layers
                       gives the sizes of the hidden layers
        dim_R: int
               Dimension of the latent representation (designated with "r" in the paper)
        '''
        #Save the dimensions:  
        self.dim_X=dim_X
        self.dim_Y=dim_Y
        self.dim_R=dim_R
        self.hidden_layers=hidden_layers

        if (len(hidden_layers)==0):
            layers_list=[nn.Linear(dim_X + dim_Y, self.dim_R)]
        else: 
            layers_list=[nn.Linear(dim_X + dim_Y, hidden_layers[0])]
        for i in range(0,len(self.hidden_layers)-1):
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Linear(self.hidden_layers[i],self.hidden_layers[i+1]))
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Linear(self.hidden_layers[-1],self.dim_R))

        self.encoder=nn.Sequential(*layers_list)

    def forward(self,x,y):
        '''
        Input:
        x: torch.Tensor
           Shape (batch_size,n_context_points,self.dim_X)
        y: torch.Tensor
           Shape (batch_size,n_context_points,self.dim_Y)
        Output:
        encoder_mean: torch.Tensor
                      Shape (batch_size,self.dim_R)
                      Forwards concatenated pairs (x,y) through MLP and takes the mean 
                      through a context set
        '''
        if (len(x.shape)!=3) or (x.size(2)!=self.dim_X):
            sys.exit("Input x for Encoder has wrong shape.")
        if (len(y.shape)!=3) or (y.size(2)!=self.dim_Y):
            sys.exit("Input y for Encoder has wrong shape.")
        if (y.size(0)!=x.size(0)) or (y.size(1)!=x.size(1)):
            sys.exit("Inputs x and y for Encoder do not have the same number of context points or minibatch size.")       

        encoder_input_pairs=torch.cat((x,y),dim=2)
        encoder_output=self.encoder(encoder_input_pairs)
        #Take the mean of the vectors:
        encoder_mean=encoder_output.mean(dim=1)
        return(encoder_mean)
        
class CNPDecoder(nn.Module):
    def __init__(self, dim_X, dim_Y, dim_R, hidden_layers):
        super(CNPDecoder, self).__init__()
        '''
        Inputs:
          dim_X: int
                dimension of the state space 
          dim_Y: int
                dimension of the label/output space
          dim_R: int
                dimension of the data set representation (denoted by r in the CNP paper)
          hidden_layers: list of integers with length = number of hidden layers
                         lists the sizes of the hidden layers
        '''
        #Save the variables:  
        self.dim_X=dim_X
        self.dim_Y=dim_Y
        self.hidden_layers=hidden_layers
        self.dim_R=dim_R
        
        if (len(hidden_layers)==0):
            layers_list=[nn.Linear(dim_X +self.dim_R, 2*self.dim_Y)]
        else: 
            layers_list=[nn.Linear(dim_X +self.dim_R, hidden_layers[0])]
        for i in range(0,len(self.hidden_layers)-1):
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Linear(self.hidden_layers[i],self.hidden_layers[i+1]))
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Linear(self.hidden_layers[-1],2*self.dim_Y))

        self.decoder=nn.Sequential(*layers_list)
  
    def forward(self,x,r):
        '''
        Inputs:
          x: torch.tensor
             shape (batch_size,n_target_points,self.dim_X)
          r: torch.tensor
             shape (batch_size,self.dim_R)

        Returns:
        dist_tuple:torch.distributions.normal.Normal 
                   this is a tuple of one-dimensional normal distributions of shape (batch_size,n_target_points,self.dim_Y)
                   with mean_vec as means and scale_vec as standard deviations
        mean_vec:  torch.tensor
                   shape (batch_size,n_target_points,self.dim_Y)
                   vector of means of the normal distributions
        scale_vec: torch.tensor
                   shape (batch_size,n_target_points,self.dim_Y)
                   vector of standard deviations of the normal distributions
        '''
        batch_size=x.size(0)
        n_target_points=x.size(1)

        #Control inputs:
        if (len(x.shape)!=3) or (x.size(2)!=self.dim_X):
            sys.exit("Input x for Decoder has the wrong shape.")
        if (len(r.shape)!=2) or (r.size(1)!=self.dim_R) or (r.size(0)!=batch_size):
            sys.exit("Input r for Decoder has the wrong shape.")


        #Expand r by adding a dimension for the target points and replicate them:
        r=r.unsqueeze(1).expand(batch_size,n_target_points,self.dim_R)
        decoder_input=torch.cat((x,r),dim=2)

        #Send the input through the MLP:
        decoder_output=self.decoder(decoder_input)

        #First half of the components is the mean vector:
        mean_vec=decoder_output[:,:,:self.dim_Y]

        #Second half is the reparameterized variance:
        repar_scale_vec=decoder_output[:,:,self.dim_Y:]

        #Obtain original for variance by a transformation (I follow the official implementation here):
        scale_vec=0.1 + 0.9 * F.softplus(repar_scale_vec)

        return mean_vec, scale_vec


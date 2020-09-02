import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
from numpy import loadtxt
from numpy import savetxt

#Import own files:
import Enc_Dec_Models as models
sys.path.append('../')
import My_Tools
import Tasks.GP_Data.GP_div_free_circle.loader as DataLoader

class ConditionalNeuralProcess(nn.Module):
    def __init__(self, dim_X, dim_Y_in,dim_Y_out, dim_R, hidden_layers_encoder, 
               hidden_layers_decoder):
        super(ConditionalNeuralProcess, self).__init__()
        '''
        Inputs:
        dim_X: int
               dimension of state space
        dim_Y_in,dim_Y_out: int
                            dimension of label space of inputs and outputs
        dim_R: int
               dimension of latent representation
        hidden_layers_encoder: list of int
                                hidden layers for the encoder 
        hidden_layers_decoder:  list of int
                                hidden layers for the decoder 
        '''

        #Initialize the dimension of the spaces:
        self.dim_X=dim_X
        self.dim_Y_in=dim_Y_in
        self.dim_Y_out=dim_Y_out
        self.dim_R=dim_R

        #Initialize the latent path:
        self.encoder=models.CNPEncoder(dim_X=dim_X, dim_Y=dim_Y_in, hidden_layers=hidden_layers_encoder, 
                                      dim_R=dim_R)

        #Intialize the decoder:
        self.decoder=models.CNPDecoder(dim_X=dim_X, dim_Y=dim_Y_out, dim_R=dim_R, hidden_layers=hidden_layers_decoder)


    def forward(self,x_context,y_context,x_target):
        '''
        Input:
          x_context: torch.Tensor
                      Shape (batch_size,n_context_points,self.dim_X)
          y_context: torch.Tensor
                      Shape (batch_size,n_context_points,self.dim_Y)
          x_target:  torch.Tensor
                      Shape (batch_size,n_target_points,self.dim_X)
        Output:
            mean_vec - torch.Tensor - shape (batch_size,n_target_points,self.dim_Y)
            Covs - torch.Tensor - shape (batch_size,n_target_points,self.dim_Y,self.dim_Y)
        '''
        batch_size,n_target_points,_=x_target.size()

        r=self.encoder(x_context,y_context) #Shape of r: (batch_size,self.dim_R)
        
        mean_vec, scale_vec=self.decoder(x=x_target,r=r)
        
        Covs=scale_vec.diag_embed()
        return mean_vec,Covs

    def loss(self,Y_Target,Predict,Covs):
        '''
            Inputs: Y_Target: torch.tensor - shape (batch_size,n,2) - Target set locations and vectors
                    Predict: torch.tensor - shape (batch_size,n,2) - Predictions of Y_Target at X_Target
                    Covs: torch.tensor - shape (batch_size,n,2,2) - covariance matrices of Y_Target at X_Target
            Output: -log_ll,log_ll
        '''
        log_ll_vec=My_Tools.batch_multivar_log_ll(Means=Predict,Covs=Covs,Data=Y_Target)
        log_ll=log_ll_vec.mean()
        loss=-log_ll
        return(loss,log_ll)

    def give_dict(self):
        dictionary={
             'dim_X': self.dim_X,
             'dim_Y_in': self.dim_Y_in, 
             'dim_Y_out': self.dim_Y_out, 
             'dim_R': self.dim_R,
             'hidden_layers_encoder': self.encoder.hidden_layers,
             'hidden_layers_decoder': self.decoder.hidden_layers, 
        }
        return(dictionary)
    def create_model_from_dict(dictionary):
        return(ConditionalNeuralProcess(**dictionary))



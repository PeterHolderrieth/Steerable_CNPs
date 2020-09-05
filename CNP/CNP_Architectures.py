import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
from numpy import loadtxt
from numpy import savetxt

#Import own files:
import CNP.Enc_Dec_Models as models
import CNP.CNP_Model as CNP_Model
import My_Tools

def give_CNP_architecture(name,dim_X=2,dim_Y=2):
    if name=='paper':
        dim_R=128 
        hidden_layers_encoder=[128,128,128] 
        hidden_layers_decoder=[128,128]
    
    elif name=='double'
        dim_R=2*128 
        hidden_layers_encoder=[2*128,2*128,2*128] 
        hidden_layers_decoder=[2*128,2*128]
    
    elif name=='small'
        dim_R=128 
        hidden_layers_encoder=[128,128] 
        hidden_layers_decoder=[128,128]

    elif name=='big'
        dim_R=2*128 
        hidden_layers_encoder=[2*128,2*128,2*128,2*128] 
        hidden_layers_decoder=[2*128,2*128]
    
    elif name=='thin'
        dim_R=2*128 
        hidden_layers_encoder=[4*128,4*128]
        hidden_layers_decoder=[4*128,4*128]
    
    else:
        sys.exit("Unknown architecture name.")
    
    return(CNP_Model.ConditionalNeuralProcess(dim_X=dim_X,dim_Y_in=dim_Y,dim_Y_out=dim_Y,dim_R=dim_R,
                                                hidden_layers_encoder=hidden_layers_encoder,hidden_layers_decoder=hidden_layers_decoder))
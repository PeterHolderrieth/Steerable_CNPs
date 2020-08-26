#LIBRARIES:
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

#E(2)-steerable CNNs - librar"y:
from e2cnn import gspaces    
from e2cnn import nn as G_CNN   
import e2cnn

#Plotting in 2d/3d:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
import matplotlib.cm as cm

#Tools:
import datetime
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Own files:
import Kernel_and_GP_tools as GP
import My_Tools
import EquivDeepSets 


#HYPERPARAMETERS and set seed:
torch.set_default_dtype(torch.float)

'''
-------------------------------------------------------------------------
--------------------------DECODER CLASSES----------------------------------
-------------------------------------------------------------------------
'''
#A STACK OF CNN LAYERS:
class CNNDecoder(nn.Module):
    def __init__(self,list_hid_channels,kernel_sizes,dim_cov_est,non_linearity=["ReLU"],dim_features_inp=2):
        '''
        Input: list_hid_channels - list of ints -  element i gives the number of channels of hidden layer i 
               kernel_sizes - list of odd ints - sizes of kernels for convolutional layers 
                                                (need to be odd because height and width of input and output tensors have to be the same)
                dim_cov_est - int - dimension of covariance estimation: 1 - scalar rep, 2 - diagonal covariance matrix, 3 - eigenvalue, 4 - quadratic
               non_linearity - list of strings - gives names of non-linearity to be used
                                                 Either length 1 (then same non-linearity for all)
                                                 or length is the number of layers (giving a custom non-linearity for every
                                                 layer)                   
                dim_features_in,dim_features_out - int - dimension of feature space for inputs and outputs (usually dim_features_in=dim_features_out)
        -->Creates a stack of CNN layers with number of channels given by "list_n_channels" and 
        kernel sizes given by self.kernel_sizes - we perform padding such that the height and width do not change
        '''    
        #Initialize:
        super(CNNDecoder, self).__init__()
        #Save a list of the number of channels per layer (input and output are given):
        self.list_n_channels=[1+dim_features_inp,*list_hid_channels,dim_cov_est+2]       
        #Save the kernel sizes:
        self.kernel_sizes=kernel_sizes
        #Save the number of layers:
        self.n_layers=len(self.list_n_channels)
        #Save the dimension of covariance estimation:
        self.dim_cov_est=dim_cov_est
        #Save the number of hidden channels:
        self.list_hid_channels=list_hid_channels

        #Save the dimension of the input and the output features:
        self.dim_features_inp=dim_features_inp

        #-----CREATE LIST OF NON-LINEARITIES----
        if len(non_linearity)==1:
            self.non_linearity=(self.n_layers-2)*non_linearity
        elif len(non_linearity)!=(self.n_layers-2):
            sys.exit("List of non-linearities invalid: must have either length 1 or n_layers-2")
        else:
            self.non_linearity=non_linearity
        #-----END CREATE LIST OF NON-LINEARITIES----

        #----------CREATE DECODER----------------
        '''
        We create a stack of CNN layers with number of channels given by "list_n_channels" and 
        kernel sizes given by self.kernel_sizes - we perform padding such that the height and width do not change
        '''
        #Create layers list and append it:
        layers_list=[nn.Conv2d(self.list_n_channels[0],self.list_n_channels[1],
                            kernel_size=kernel_sizes[0],padding=(kernel_sizes[0]-1)//2)]

        for it in range(self.n_layers-2):
            if self.non_linearity[it]=="ReLU":
                layers_list.append(nn.ReLU())
            else:
                sys.exit("Unknown non-linearity.")
            layers_list.append(nn.Conv2d(self.list_n_channels[it+1],self.list_n_channels[it+2],
                                            kernel_size=kernel_sizes[it],padding=(kernel_sizes[it]-1)//2))
        #Create a steerable decoder out of the layers list:
        self.decoder=nn.Sequential(*layers_list)
        #----------END CREATE DECODER--------------

        #-----------CONTROL INPUTS------------------
        if any([j%2-1 for j in kernel_sizes]): sys.exit("All kernels need to have odd sizes")
        if len(kernel_sizes)!=(self.n_layers-1): sys.exit("Number of layers and number kernels do not match.")
        if len(self.non_linearity)!=(self.n_layers-2): sys.exit("Number of layers and number of non-linearities do not match.")
        if not isinstance(self.dim_cov_est,int) or self.dim_cov_est>4:
            sys.exit("The dimension of covariance estimation must be less or equal to 4.")
        #------------END CONTROL INPUTS--------------

    def forward(self,X):
        '''
        X - torch.tensor - shape (batch_size,self.list_n_channels[0],height,width)
        '''
        return(self.decoder(X))
    
    def give_dict(self):
        dictionary={
            'list_hid_channels': self.list_hid_channels,
            'kernel_sizes': self.kernel_sizes,
            'dim_cov_est': self.dim_cov_est,
            'non_linearity': self.non_linearity,
            'dim_features_inp': self.dim_features_inp,
            'decoder_class': self.__class__.__name__,
            'decoder_info': self.decoder.__str__(),
            'decoder_par': self.decoder.state_dict()
        }
        return(dictionary)

    def save_model_dict(self,filename):
        torch.save(self.give_dict(),f=filename)

    def create_model_from_dict(dictionary):
        '''
        Input: dictionary - dictionary - gives parameters for decoder (weights and biases are randomly initialised
                                        if decoder parameters are not given)
        Output: Decoder - instance of CNN_Decoder (see above) 
        '''
        Decoder=CNNDecoder(list_hid_channels=dictionary['list_hid_channels'],
                                    kernel_sizes=dictionary['kernel_sizes'],
                                    dim_cov_est=dictionary['dim_cov_est'],
                                    non_linearity=dictionary['non_linearity'],
                                    dim_features_inp=dictionary['dim_features_inp'],
                                )
        if 'decoder_par' in dictionary:
            Decoder.decoder.load_state_dict(dictionary['decoder_par'])
        return(Decoder)

    def load_model_from_dict(filename):
        dictionary=torch.load(f=filename)
        return(CNN_Decoder.create_model_from_dict(dictionary))


#A decoder class which is equivariant with respect the cyclic group C_N (i.e. rotations of 360/N degrees):
class EquivDecoder(nn.Module):
    def __init__(self,hidden_fib_reps,kernel_sizes,dim_cov_est,context_fib_rep=[1],non_linearity=["NormReLU"],N=4,flip=False):
        '''
        Input: context_fib_rep - list of ints: gives the input fiber representation
               hidden_fib_reps - list of lists of ints - "-1"...encodes regular repr
                                                  k=0,1,2,...,flatten(N/2) (for )...encodes irrep(k) 
                                                  In particular: "0"...encodes trivial repr
               kernel_sizes - list of ints - sizes of kernels for convolutional layers
               dim_cov_est - dimension of covariance estimation, either 1,2,3 or 4
               non_linearity - list of strings - gives names of non-linearity to be used
                                                 Either length 1 (then same non-linearity for all)
                                                 or length is the number of layers (giving a custom non-linearity for every
                                                 layer)                   
        '''
        super(EquivDecoder, self).__init__()
        #Save the rotation group, if flip is true, then include all corresponding reflections:
        if flip:
            self.G_act=gspaces.FlipRot2dOnR2(N=N)
        else:
            self.G_act=gspaces.Rot2dOnR2(N=N)
        #Save the group order (if N=-1 it is infinity):
        self.Group_order=N 

        #Save the parameters:
        self.kernel_sizes=kernel_sizes
        self.n_layers=len(hidden_fib_reps)+2
        self.context_fib_rep=context_fib_rep
        self.hidden_fib_reps=hidden_fib_reps
        self.dim_cov_est=dim_cov_est
        
        #-----CREATE LIST OF NON-LINEARITIES----
        if len(non_linearity)==1:
            self.non_linearity=(self.n_layers-2)*non_linearity
        elif len(non_linearity)!=(self.n_layers-2):
            sys.exit("List of non-linearities invalid: must have either length 1 or n_layers-2")
        else:
            self.non_linearity=non_linearity
        #-----ENDE LIST OF NON-LINEARITIES----

        #-----------CREATE DECODER-----------------
        '''
        Create a list of layers based on the kernel sizes. Compute the padding such
        that the height h and width w of a tensor with shape (batch_size,n_channels,h,w) does not change
        while being passed through the decoder
        '''
        #Create list of feature types:
        feat_types=self.give_feat_types()
        self.feature_in=feat_types[0]
        self.feature_out=feat_types[-1]
        #Create layers list and append it:
        layers_list=[G_CNN.R2Conv(feat_types[0],feat_types[1],kernel_size=kernel_sizes[0],padding=(kernel_sizes[0]-1)//2)]

        for it in range(self.n_layers-2):
            if self.non_linearity[it]=="ReLU":
                layers_list.append(G_CNN.ReLU(feat_types[it+1]))
            elif self.non_linearity[it]=="NormReLU":
                layers_list.append(G_CNN.NormNonLinearity(feat_types[it+1]))
            else:
                sys.exit("Unknown non-linearity.")
            layers_list.append(G_CNN.R2Conv(feat_types[it+1],feat_types[it+2],kernel_size=kernel_sizes[it],padding=(kernel_sizes[it]-1)//2))
        #Create a steerable decoder out of the layers list:
        self.decoder=G_CNN.SequentialModule(*layers_list)
        #-----------END CREATE DECODER---------------

        #-----------CONTROL INPUTS------------------
        #Control that all kernel sizes are odd (otherwise output shape is not correct):
        if any([j%2-1 for j in kernel_sizes]): sys.exit("All kernels need to have odd sizes")
        if len(kernel_sizes)!=(self.n_layers-1): sys.exit("Number of layers and number kernels do not match.")
        if len(self.non_linearity)!=(self.n_layers-2): sys.exit("Number of layers and number of non-linearities do not match.")
        #------------END CONTROL INPUTS--------------
    
    #A tool for initialising the class:
    def give_feat_types(self):
        '''
        Output: feat_types - list of features types (see class ) 
                           - self.fib_reps[i]=[k_1,...,k_l] gives a list of integers where
                             k_i stands for irrep(k_i) of the rotation group or if k_i=-1 for the regular representation
                             the sume of rep(k_1),...,rep(k_l) determines the ith element of "feat_types"
        '''
        feat_types=[G_CNN.FieldType(self.G_act,[self.G_act.trivial_repr,self.G_act.irrep(self.context_fib_rep)])]
        for reps in self.hidden_fib_reps:
            #New layer collects the sum of individual representations to one list:
            new_layer=[]
            for rep in reps:
                #The representation can be trivial (irrep(0)), regular or a non-trivial irreducible:
                if rep==0:
                    new_layer.append(self.G_act.trivial_repr)
                elif rep==-1:
                    new_layer.append(self.G_act.regular_repr)
                elif rep%1==0 and 1<=rep and rep<=math.floor(self.Group_order/2):
                    new_layer.append(self.G_act.irrep(rep))
                else:
                    sys.exit("Unknown fiber representation.")
            #Append a new feature type given by the new layer:
            feat_types.append(G_CNN.FieldType(self.G_act, new_layer))

        if self.dim_cov_est==1:
            pre_cov_rep=self.G_act.trivial_repr
        else:
            pre_cov_rep,_=My_Tools.get_pre_psd_rep(self.G_act)
        feat_types.append(G_CNN.FieldType(self.G_act,[self.G_act.irrep(1),pre_cov_rep]))
        return(feat_types)
    
    def forward(self,X):
        '''
        Input: X - torch.tensor - shape (batch_size,n_in_channels,m,n)
        Output: torch.tensor - shape (batch_size,n_out_channels,m,n)
        '''
        #Convert X into a geometric tensor:
        X=G_CNN.GeometricTensor(X, self.feature_in)
        #Send it through the decoder:
        Out=self.decoder(X)
        #Return the resulting tensor:
        return(Out.tensor)

    #Two functions to save the model in a dictionary:
    #1.Create dictionary with parameters:
    def give_dict(self):
        dictionary={
            'N': self.Group_order,
            'hidden_fib_reps': self.hidden_fib_reps,
            'dim_cov_est': self.dim_cov_est,
            'kernel_sizes': self.kernel_sizes,
            'non_linearity': self.non_linearity,
            'decoder_class': self.__class__.__name__,
            'decoder_info': self.decoder.__str__(),
            'decoder_par': self.decoder.state_dict()
        }
        return(dictionary)
    #2.Save dictionary:
    def save_model_dict(self,filename):
        torch.save(self.give_dict(),f=filename)

    #Two functions to load the model from file:
    #1.Create Model from dictionary:
    def create_model_from_dict(dictionary):
        '''
        Input: dictionary - dictionary - gives parameters for decoder which is randomly initialised
                                        if decoder parameters (weights and biases) are given 
                                        the weights are loaded into the decoder
        Output: Decoder - instance of Cyclic_Decoder (see above) 
        '''
        Decoder=Cyclic_Decoder(N=dictionary['N'],
                                    hidden_fib_reps=dictionary['hidden_fib_reps'],
                                    dim_cov_est=dictionary['dim_cov_est'],
                                    kernel_sizes=dictionary['kernel_sizes'],
                                    non_linearity=dictionary['non_linearity'],
                                )
        if 'decoder_par' in dictionary:
            if dictionary['decoder_par'] is not None:
                Decoder.decoder.load_state_dict(dictionary['decoder_par'])
        return(Decoder)

    #2. Load dictionary from file and create it:
    def load_model_from_dict(filename):
        '''
        Input: filename - string - name of file where dictionary is saved
        Output: instance of Cyclic_Decoder with parameters as specified at "filename"
        '''
        dictionary=torch.load(f=filename)
        return(Cyclic_Decoder.create_model_from_dict(dictionary))

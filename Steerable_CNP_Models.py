#LIBRARIES:
#Tensors:

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torchsummary import summary

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



#HYPERPARAMETERS:
torch.set_default_dtype(torch.float)
quiver_scale=15

'''
-------------------------------------------------------------------------
--------------------------ENCODER CLASS----------------------------------
-------------------------------------------------------------------------
'''
class Steerable_Encoder(nn.Module):
    def __init__(self, x_range=[-2,2],y_range=None,n_x_axis=10,n_y_axis=None,kernel_dict={'kernel_type':"rbf"},
                 l_scale=1.,normalize=True):
        super(Steerable_Encoder, self).__init__()
        '''
        Inputs:
            x_range,y_range: float lists of size 2 - give range of grid points at x-axis/y-axis
            n_x_axis: int - number of grid points along the x-axis
            n_y_axis: int - number of grid points along the y-axis
            kernel_dict: dictionary - parameters for function mat_kernel (see My_Tools.Gram_mat)
            l_scale: float - initialisation of length scale
            normalize: boolean - indicates whether feature channels is divided by density channel
        '''
        #-------------------------SET PARAMETERS-----------------
        #Bool whether to normalize:
        self.normalize=normalize
        
        #Kernel parameters:
        self.kernel_type=kernel_dict['kernel_type']
        self.log_l_scale=nn.Parameter(torch.log(torch.tensor(l_scale,dtype=torch.get_default_dtype())),requires_grad=False)
        self.kernel_dict=kernel_dict

        #Grid parameters:
        #x-axis:
        self.x_range=x_range
        self.n_x_axis=n_x_axis
        #y-axis (same as for x-axis if not given):
        self.y_range=y_range if y_range is not None else x_range
        self.n_y_axis=n_y_axis if n_y_axis is not None else n_x_axis

        #Grid:
        '''
        Create a flattened grid--> shape (n_y_axis*n_x_axis,2) 
        #x-axis is counted periodically, y-axis stays the same per period of counting the x-axis.
        i.e. self.grid[k*n_x_axis+j] corresponds to element k in the y-grid and j in the x-grid.
        Important: The counter will go BACKWARDS IN THE Y-AXIS - this is because
        if we look at a (m,n)-matrix as a matrix with pixels, then the higher 
        the row index, the lower its y-axis value, i.e. the y-axis is counted 
        mirrored.
        '''
        self.grid=nn.Parameter(My_Tools.Give_2d_Grid(min_x=self.x_range[0],max_x=self.x_range[1],
                               min_y=self.y_range[1],max_y=self.y_range[0],
                               n_x_axis=self.n_x_axis,n_y_axis=self.n_y_axis,flatten=True),requires_grad=False)
            
        #-------------------------SET PARAMETERS FINISHED-----------------
        
        #-------------------------CONTROL PARAMETERS-----------------

        #To prevent a clash, 'B' and 'l_scale' should not be in kernel_dict:        
        if 'B' in kernel_dict or 'l_scale' in kernel_dict:
            sys.exit("So far, we do not allow for a multi-dimensional kernel in the embedding and no l_scale is allowed")
        if not isinstance(l_scale,float) or l_scale<=0:
            sys.exit("Encoder error: l_scale not correct.")
        if self.x_range[0]>=self.x_range[1] or self.y_range[0]>=self.y_range[1]:
            sys.exit("x and y range are not valid.")
        #-------------------------CONTROL PARAMETERS FINISHED-----------------

    #Expansion of a label vector y in the embedding.
    #This is the function y->(1,y,y^2,y^3,...,y^K) in the ConvCNP paper.
    #For now it just adding a one, i.e. y->(1,y), since we assume multiplicity one:
    def Psi(self,Y):
        '''
        Input: Y - torch.tensor - shape (n,2)
        Output: torch.tensor -shape (n,3) - added a column of ones to Y (at the start) Y[i,j]<--[1,Y[i,j]]
        '''
        return(torch.cat((torch.ones((Y.size(0),1),device=Y.device),Y),dim=1))

    def forward(self,X,Y):
        '''
        Inputs:
            X: torch.tensor - shape (n,2)
            Y: torch.tensor - shape (n,self.dim_Y)
        Outputs:
            torch.tensor - shape (1,self.dim_Y+1,self.n_y_axis,self.n_x_axis) (shape which can be processed by CNN)
        '''
        #DEBUG: Control whether X and the grid are on the same device:
        if self.grid.device!=X.device:
            print("Grid and X are on different devices.")
            self.grid=self.grid.to(X.device)
        
        #Compute the length scale out of the log-scale (clamp for numerical stability):
        l_scale=torch.exp(torch.clamp(self.log_l_scale,max=5.,min=-5.))
        #Compute for every grid-point x' the value k(x',x_i) for all x_i in the data-->shape (self.n_y_axis*self.n_x_axis,n)
        Gram=GP.Gram_matrix(self.grid,X,l_scale=l_scale,**self.kernel_dict,B=torch.ones((1),device=X.device))
        
        #Compute feature expansion:
        Expand_Y=self.Psi(Y)
        
        #Compute feature map -->shape (self.n_y_axis*self.n_x_axis,3)
        Feature_Map=torch.mm(Gram,Expand_Y)

        #If wanted, normalize the weights for the channel which is not the density channel:
        if self.normalize:
            #Normalize the functional representation:
            Norm_Feature_Map=torch.empty(Feature_Map.size(),device=Feature_Map.device)
            Norm_Feature_Map[:,1:]=Feature_Map[:,1:]/Feature_Map[:,0].unsqueeze(1)
            Norm_Feature_Map[:,0]=Feature_Map[:,0]
            #Reshape the Feature Map to the form (1,n_channels=3,n_y_axis,n_x_axis) (because this is the form required for a CNN):
            return(Norm_Feature_Map.reshape(self.n_y_axis,self.n_x_axis,Expand_Y.size(1)).permute(dims=(2,0,1)).unsqueeze(0))        
        #Reshape the Feature Map to the form (1,n_channels=3,n_y_axis,n_x_axis) (because this is the form required for a CNN):
        else:   
            return(Feature_Map.reshape(self.n_y_axis,self.n_x_axis,Expand_Y.size(1)).permute(dims=(2,0,1)).unsqueeze(0))    
    
    def plot_embedding(self,Embedding,X_context=None,Y_context=None,title=""):
        '''
        Input: 
               Embedding - torch.tensor - shape (1,n_grid_points,3) - Embedding obtained from self.forward (usually  from X_context,Y_context)
                                                                      where Embedding[0,0] is the density channel
                                                                      and Embedding[0,1:] is the smoothing channel
               X_context,Y_context - torch.tensor - shape (n,2) - context locations and vectors
               title - string - title of plots
        Plots locations X_context with vectors Y_context attached to it
        and on top plots the kernel smoothed version (i.e. channel 2,3 of the embedding)
        Moreover, it plots a density plot (channel 1 of the embedding)
        '''
        #Hyperparameter for function for plotting in notebook:
        size_scale=2
        #----------CREATE FIGURE --------------
        #Create figures, set title and adjust space:
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(size_scale*10,size_scale*5))
        plt.gca().set_aspect('equal', adjustable='box')
        fig.suptitle(title)
        fig.subplots_adjust(wspace=0.2)
        #----------END CREATE FIGURE-------

        #---------SMOOTHING PLOT---------
        #Set titles for subplots:
        ax[0].set_title("Smoothing channel")
        
        if X_context is not None and Y_context is not None:
            #Plot context set in black:
            ax[0].scatter(X_context[:,0],X_context[:,1],color='black')
            ax[0].quiver(X_context[:,0],X_context[:,1],Y_context[:,0],Y_context[:,1],
              color='black',pivot='mid',label='Context set',scale=quiver_scale)

        #Get embedding of the form (3,self.n_y_axis,self.n_x_axis)
        Embedding=Embedding.squeeze()
        #Get density channel --> shape (self.n_y_axis,self.n_x_axis)
        Density=Embedding[0]
        #Get Smoothed channel -->shape (self.n_y_axis*self.n_x_axis,2)
        Smoothed=Embedding[1:].permute(dims=(1,2,0)).reshape(-1,2)
        #Plot the kernel smoothing:
        ax[0].quiver(self.grid[:,0],self.grid[:,1],Smoothed[:,0],Smoothed[:,1],color='red',pivot='mid',label='Embedding',scale=quiver_scale)
        #--------END SMOOTHING PLOT------

        #--------DENSITY PLOT -----------
        ax[1].set_title("Density channel")
        #Get X values of grid:
        X=self.grid[:self.n_x_axis,0]
        #Get Y values of grid:
        Y=self.grid.view(self.n_y_axis,self.n_x_axis,2).permute(dims=(1,0,2))[0,:self.n_y_axis,1]  
        #Set X and Y range to the same as for the first plot:
        ax[1].set_xlim(ax[0].get_xlim())
        ax[1].set_ylim(ax[0].get_ylim())
        #Plot a contour plot of the density channel:
        ax[1].set_title("Density channel")
        ax[1].contour(X,Y, Density, levels=14, linewidths=0.5, colors='k')
        #Add legend to first plot:
        leg = ax[0].legend(loc='upper right')
        #---------END DENSITY PLOT--------
    
    def give_dict(self):
        dictionary={
            'x_range':self.x_range,
            'y_range':self.y_range,
            'n_x_axis':self.n_x_axis,
            'n_y_axis':self.n_y_axis,
            'kernel_dict':self.kernel_dict,
            'l_scale': torch.exp(self.log_l_scale).item(),
            'normalize': self.normalize
        }
        return(dictionary)

'''
-------------------------------------------------------------------------
--------------------------DECODER CLASSES----------------------------------
-------------------------------------------------------------------------
'''
#A STACK OF CNN LAYERS:
class CNN_Decoder(nn.Module):
    def __init__(self,list_n_channels,kernel_sizes,non_linearity=["ReLU"]):
        '''
        Input: list_n_channels - list of ints -  element i gives the number of channels of layer i with layer 0 is the input layer
               kernel_sizes - list of odd ints - sizes of kernels for convolutional layers 
                                                (need to be odd because height and width of input and output tensors have to be the same)
               non_linearity - list of strings - gives names of non-linearity to be used
                                                 Either length 1 (then same non-linearity for all)
                                                 or length is the number of layers (giving a custom non-linearity for every
                                                 layer)                   
        -->Creates a stack of CNN layers with number of channels given by "list_n_channels" and 
        kernel sizes given by self.kernel_sizes - we perform padding such that the height and width do not change
        '''    
        #Initialize:
        super(CNN_Decoder, self).__init__()
        self.list_n_channels=list_n_channels        
        self.kernel_sizes=kernel_sizes
        self.n_layers=len(list_n_channels)

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
        layers_list=[nn.Conv2d(list_n_channels[0],list_n_channels[1],
                            kernel_size=kernel_sizes[0],padding=(kernel_sizes[0]-1)//2)]

        for it in range(self.n_layers-2):
            if self.non_linearity[it]=="ReLU":
                layers_list.append(nn.ReLU())
            else:
                sys.exit("Unknown non-linearity.")
            layers_list.append(nn.Conv2d(list_n_channels[it+1],list_n_channels[it+2],
                                            kernel_size=kernel_sizes[it],padding=(kernel_sizes[it]-1)//2))
        #Create a steerable decoder out of the layers list:
        self.decoder=nn.Sequential(*layers_list)
        #----------END CREATE DECODER--------------

        #-----------CONTROL INPUTS------------------
        if any([j%2-1 for j in kernel_sizes]): sys.exit("All kernels need to have odd sizes")
        if len(kernel_sizes)!=(self.n_layers-1): sys.exit("Number of layers and number kernels do not match.")
        if len(self.non_linearity)!=(self.n_layers-2): sys.exit("Number of layers and number of non-linearities do not match.")
        #------------END CONTROL INPUTS--------------

    def forward(self,X):
        return(self.decoder(X))
    
    def give_dict(self):
        dictionary={
            'list_n_channels': self.list_n_channels,
            'kernel_sizes': self.kernel_sizes,
            'non_linearity': self.non_linearity,
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
        Decoder=CNN_Decoder(list_n_channels=dictionary['list_n_channels'],
                                    kernel_sizes=dictionary['kernel_sizes'],
                                    non_linearity=dictionary['non_linearity']
                                )
        if 'decoder_par' in dictionary:
            Decoder.decoder.load_state_dict(dictionary['decoder_par'])
        return(Decoder)

    def load_model_from_dict(filename):
        dictionary=torch.load(f=filename)
        return(CNN_Decoder.create_model_from_dict(dictionary))


#A decoder class which is equivariant with respect the cyclic group C_N (i.e. rotations of 360/N degrees):
class Cyclic_Decoder(nn.Module):
    def __init__(self,fib_reps,kernel_sizes,non_linearity=["ReLU"],N=4):
        '''
        Input: fib_reps - list of lists of ints - "-1"...encodes regular repr
                                                  k=0,1,2,..,flatten(N/2)...encodes irrep(k)
                                                  In particular: "0"...encodes trivial repr
               kernel_sizes - list of ints - sizes of kernels for convolutional layers
               non_linearity - list of strings - gives names of non-linearity to be used
                                                 Either length 1 (then same non-linearity for all)
                                                 or length is the number of layers (giving a custom non-linearity for every
                                                 layer)                   
        '''
        super(Cyclic_Decoder, self).__init__()
        #Save the group and the parameters:
        self.Group_order=N
        self.G_act=gspaces.Rot2dOnR2(N=N)
        self.kernel_sizes=kernel_sizes
        self.n_layers=len(fib_reps)
        self.fib_reps=fib_reps
        
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
        feat_types=[]
        for reps in self.fib_reps:
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
            'fib_reps': self.fib_reps,
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
                                    fib_reps=dictionary['fib_reps'],
                                    kernel_sizes=dictionary['kernel_sizes'],
                                    non_linearity=dictionary['non_linearity']
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



'''
-------------------------------------------------------------------------
--------------------------STEERABLE CNP CLASS----------------------------
-------------------------------------------------------------------------
'''     
class Steerable_CNP(nn.Module):
    def __init__(self, encoder, decoder, dim_cov_est=3,
                         kernel_dict_out={'kernel_type':"rbf"},l_scale=1.,normalize_output=True):
        '''
        Inputs:
            encoder - instance of Steerable_Encoder class above
            decoder - nn.Module - takes input (batch_size,3,height,width) and gives (batch_size,5,height,width) 
                                  (dim_cov_est=3) or (batch_size,3,height,width) (if dim_cov_est=1) as output
            decoder_type - string - type of decoder: possible - "Conv" (standard CNN)
                                                              - "Cyclic" (Group-Equivariant network w.r.t. cyclic group)
            kernel_dict_out - gives parameters for kernel smoother of output
            l_scale - float - gives initialisation for learnable length parameter
            normalize_output  - Boolean - indicates whether kernel smoothing is performed with normalizing
        '''
        #-----------------------SAVING OF PARAMETERS ----------------------------------
        super(Steerable_CNP, self).__init__()
        #Initialse the encoder:
        self.encoder=encoder
        #Decoder and save the type (Convolutional, Steerable and if Steerable which group)
        self.decoder=decoder
        self.decoder_type=decoder.__class__.__name__
        #Get the parameters for kernel smoother for the target set:
        self.log_l_scale_out=nn.Parameter(torch.log(torch.tensor(l_scale,dtype=torch.get_default_dtype())),requires_grad=True)
        #Get the other kernel parameters for the kernel smoother for the target set (others are fixed):
        self.kernel_dict_out=kernel_dict_out
        #Save whether output is normalized (i.e. kernel smoothing is performed with normalizing):
        self.normalize_output=normalize_output
        #Save the dimension of the covariance estimator of the last layer:
        self.dim_cov_est=dim_cov_est
        #-----------------------SAVING of PARAMETERS FINISHED---------------------------------


        #--------------------CONTROL OF PARAMETERS -------------------------
        #So far, the dimension of the covariance estimator has to be either 1 or 3 
        #(i.e. number of output channels either 3 or 5):
        if (self.dim_cov_est!=1) and (self.dim_cov_est!=3): sys.exit("N out channels must be either 3 or 5")
        if 'l_scale' in kernel_dict_out: sys.exit("l scale is variable and not fixed")
        if not isinstance(self.normalize_output,bool): sys.exit("Normalize output has to be boolean.")
        if not isinstance(l_scale,float): sys.exit("l_scale initialization has to be a float.")
        if not isinstance(encoder,Steerable_Encoder): sys.exit("Enoder is not correct.")
        if not isinstance(decoder, nn.Module): sys.exit("Decoder has to be nn.Module")
        if self.decoder_type!="CNN_Decoder" and self.decoder_type!="Cyclic_Decoder": sys.exit("Unknown decoder type.")
        #--------------------END CONTROL OF PARAMETERS----------------------

        #-------------------CONTROL WHETHER DECODER ACCEPTS AND RETURNS CORRECT SHAPES----
        test_input=torch.randn([5,3,encoder.n_y_axis,encoder.n_x_axis])  
        test_output=decoder(test_input)
        if len(test_output.shape)!=4 or test_output.size(0)!=test_input.size(0) or test_output.size(2)!=encoder.n_y_axis or test_output.size(3)!=encoder.n_x_axis:
            sys.exit("Decoder error: shape of output is not correct.")
        if (self.dim_cov_est+2)!=test_output.size(1):sys.exit("Number of output channels!=2+dim of cov estimation.")
        #-------------------END CONTROL WHETHER DECODER ACCEPTS AND RETURNS CORRECT SHAPES----

    #Define the function which maps the output of the decoder to
    #predictions on the target set based on kernel smoothing, i.e. the predictions on 
    #the target set are obtained by kernel smoothing of these points on the grid of encoder
    def target_smoother(self,X_target,Final_Feature_Map):
        '''
        Input: X_target - torch.tensor- shape (n_target,2)
               Final_Feature_Map- torch.tensor - shape (4,self.encoder.n_y_axis,self.encoder.n_x_axis)
        Output: Predictions on X_target - Means_target - torch.tensor - shape (n_target,2)
                Covariances on X_target - Covs_target - torch.tensor - shape (n_target,2,2)
        '''
        #-----------SPLIT FINAL FEATURE MAP INTO MEANS AND COVARIANCE PARAMETERS----------
        #Reshape the Final Feature Map:
        Resh_Final_Feature_Map=Final_Feature_Map.permute(dims=(1,2,0)).reshape(self.encoder.n_y_axis*self.encoder.n_x_axis,-1)
        #Split into mean and parameters for covariance:
        Means_grid=Resh_Final_Feature_Map[:2]
        Pre_Activ_Covs_grid=Resh_Final_Feature_Map[2:]
        #----------END SPLIT FINAL FEATURE MAP INTO MEANS AND COVARIANCE PARAMETERS----------

        #-----------APPLY ACITVATION FUNCTION ON COVARIANCES---------------------
        #Get shape (n_x_axis*n_y_axis,2,2):
        if self.dim_cov_est==1:
            #Apply softplus (add noise such that variance does not become (close to) zero):
            Covs_grid=1e-4+F.softplus(Pre_Activ_Covs_grid).repeat(1,2)
            Covs_grid=Covs_grid.diag_embed()
        else:
            Covs_grid=My_Tools.stable_cov_activation_function(Pre_Activ_Covs_grid)
        #-----------END APPLY ACITVATION FUNCTION ON COVARIANCES---------------------

        #-----------APPLY KERNEL SMOOTHING --------------------------------------
        #Set the lenght scale (clamp for numerical stability):
        l_scale=torch.exp(torch.clamp(self.log_l_scale_out,max=5.,min=-5.))
        #Means on Target Set (via Kernel smoothing) --> shape (n_target,2):
        Means_target=GP.Kernel_Smoother_2d(X_Context=self.encoder.grid,Y_Context=Means_grid,
                                           X_Target=X_target,normalize=self.normalize_output,
                                           l_scale=l_scale,**self.kernel_dict_out)
        
        #Create flattened version (needed for target smoother):
        Covs_grid_flat=Covs_grid.view(self.encoder.n_y_axis*self.encoder.n_x_axis,-1)
        #3.Get covariances on target set--> shape (n_target,4):
        Covs_target_flat=GP.Kernel_Smoother_2d(X_Context=self.encoder.grid,Y_Context=Covs_grid_flat,
                                          X_Target=X_target,normalize=self.normalize_output,
                                          l_scale=l_scale,**self.kernel_dict_out)                                 
        #Reshape covariance matrices to proper matrices --> shape (n_target,2,2):
        Covs_target=Covs_target_flat.view(X_target.size(0),2,2)
        #-----------END APPLY KERNEL SMOOTHING --------------------------------------
        return(Means_target, Covs_target)
 
    #Define the forward pass of ConvCNP: 
    def forward(self,X_context,Y_context,X_target):
        '''
        Inputs:
            X_context: torch.tensor - shape (n_context,2)
            Y_context: torch.tensor - shape (n_context,2)
            X_target: torch.tensor - shape (n_target,2)
        Outputs:
            Means_target: torch.tensor - shape (n_target,2) - mean of predictions
            Sigmas_target: torch.tensor -shape (n_target,2) - scale of predictions
        '''
        #1.Context Set -> Embedding (via Encoder) --> shape (3,self.encoder.n_y_axis,self.encoder.n_x_axis):
        Embedding=self.encoder(X_context,Y_context)
        #2.Embedding ->Feature Map (via CNN) --> shape (2+self.dim_cov_est,self.encoder.n_y_axis,self.encoder.n_x_axis):
        Final_Feature_Map=self.decoder(Embedding).squeeze()
        #Smooth the output:
        return(self.target_smoother(X_target,Final_Feature_Map))

    def plot_Context_Target(self,X_Context,Y_Context,X_Target,Y_Target=None,title=""):
        '''
            Inputs: X_Context, Y_Context, X_Target: torch.tensor - see self.forward
                    Y_Target: torch.tensor - shape (n_context_points,2) - ground truth
            Output: None - plots predictions
        
        '''
        #Get predictions:
        Means,Covs=self.forward(X_Context,Y_Context,X_Target)
        #Plot predictions against ground truth:
        My_Tools.Plot_Inference_2d(X_Context,Y_Context,X_Target,Y_Target,Predict=Means.detach(),Cov_Mat=Covs.detach(),title=title)
    
    def loss(self,Y_Target,Predict,Covs,shape_reg=None):
        '''
            Inputs: X_Target,Y_Target: torch.tensor - shape (n,2) - Target set locations and vectors
                    Predict: torch.tensor - shape (n,2) - Predictions of Y_Target at X_Target
                    Covs: torch.tensor - shape (n,2,2) - covariance matrices of Y_Target at X_Target
                    shape_reg: float/None - if float gives the weight of the shape_regularizer term (see My_Tools.shape_regularizer)
            Output: -log_ll+shape_reg*shape_diff: log_ll is the log-likelihood at Y_Target given the parameters Predict and Covs
                                                  shape_diff is the "shape difference" (interpreted here as the variance
                                                  of the difference Prdict-Y_Target computed by My_Tools.shape_regularizer)
        '''
        log_ll_vec=My_Tools.batch_multivar_log_ll(Means=Predict,Covs=Covs,Data=Y_Target)
        log_ll=log_ll_vec.mean()
        if shape_reg is not None: 
            return(-log_ll+shape_reg*My_Tools.shape_regularizer(Y_1=Y_Target,Y_2=Predict))
        else: 
            return(-log_ll)
    #Two functions to save the model in a dictionary:
    #1.Create a dictionary:
    def give_dict(self):
        dictionary={
            'encoder_dict': self.encoder.give_dict(),
            'decoder_dict': self.decoder.give_dict(),
            'decoder_class': self.decoder_type,
            'log_l_scale_out': self.log_l_scale_out.detach().item(),
            'normalize_output': self.normalize_output,
            'dim_cov_est': self.dim_cov_est,
            'kernel_dict_out': self.kernel_dict_out
        }
        return(dictionary)
    #2.Save the dictionary in a file:
    def save_model_dict(self,filename):
        torch.save(self.give_dict(),f=filename)

    #Two functions to load the model from a dictionary:
    #1.Create model from dictionary:
    def create_model_from_dict(dictionary):
        '''
        Input: dictionary - dict - parameters to load into Steerable_CNP class (including weights and biases for decoder and encoder)
        Output: instance of Steerable_CNP with parameters as specified in dictionary
        '''
        #Load Encoder:
        Encoder=Steerable_Encoder(**dictionary['encoder_dict'])
        #Load Decoder (depending on type of decoder use different functions):
        if dictionary['decoder_class']=="Cyclic_Decoder":
            Decoder=Cyclic_Decoder.create_model_from_dict(dictionary['decoder_dict'])
        elif dictionary['decoder_class']=="CNN_Decoder":
            Decoder=CNN_Decoder.create_model_from_dict(dictionary['decoder_dict'])
        else:
            sys.exit("Unknown decoder type.")
    
        #Create model:
        Model=Steerable_CNP(encoder=Encoder,decoder=Decoder,
                        dim_cov_est=dictionary['dim_cov_est'], kernel_dict_out=dictionary['kernel_dict_out'],
                        l_scale=math.exp(dictionary['log_l_scale_out']), normalize_output=dictionary['normalize_output'])
        return(Model)

    #2. Load dictionary and from dictionary load model:
    def load_model_from_dict(filename):
        '''
        Input: filename - string -location of dictionary
        Output: instance of Steerable_CNP with parameters as specified in dictionary at path "filename"
        '''
        dictionary=torch.load(f=filename)
        return(Steerable_CNP.create_model_from_dict(dictionary))




#CONTROL CODE:
'''
#1. Saving and loading encoder - control whether it is equal:
Encoder=Steerable_Encoder(x_range=[-1,2],y_range=[3,4],n_x_axis=11,n_y_axis=7,kernel_dict={'kernel_type': 'rbf','sigma_var': 2},
                          l_scale=0.6,normalize=False)
X_Test=torch.randn((100,2))
Y_Test=torch.randn((100,2))
#PROBLEM WITH PLOT EMBEDDING:
#Test_embedding=Encoder.plot_embedding(X_Test,Y_Test)
Encoder_dict=Encoder.give_dict()
Encoder_rel=Steerable_Encoder(**Encoder_dict)
print(Encoder_rel.give_dict()==Encoder_dict)
'''
'''
#2. Saving and loading CNN_Decoder - control whether it is equal:
kernel_sizes=[5,7,9,3]
list_n_channels=[3,5,7,8,3]
non_linearity=["ReLU","ReLU","ReLU"]
Conv_Decoder=CNN_Decoder(list_n_channels,kernel_sizes,non_linearity)
Conv_Decoder.save_model_dict(filename="Test_CNN_Decoder")
Rel_Conv_Decoder=CNN_Decoder.load_model_from_dict(filename="Test_CNN_Decoder")
#for key in Rel_Conv_Decoder.give_dict().keys():
#    print(key)
#    print(Rel_Conv_Decoder.give_dict()[key]==Conv_Decoder.give_dict()[key])
#
#Compare parameters:
for parameter in Rel_Conv_Decoder.decoder.parameters():
    print(parameter.flatten()[0])
for parameter in Conv_Decoder.decoder.parameters():
    print(parameter.flatten()[0])
'''
'''
#3. Saving and loading Cyclic_Decoder - control whether it is equal:
kernel_sizes=[5,7,9,3]
fib_reps=[[1,0],[1,-1,-1],[1,-1,0],[-1,-1,-1,-1],[1,0]]
non_linearity=["NormReLU","NormReLU","NormReLU"]
Geom_Decoder=Cyclic_Decoder(fib_reps,kernel_sizes,non_linearity,N=8)
Geom_Decoder.save_model_dict(filename="Test_Cyclic_Decoder")
Rel_Geom_Decoder=Cyclic_Decoder.load_model_from_dict(filename="Test_Cyclic_Decoder")
#for key in Rel_Geom_Decoder.give_dict().keys():
#    print(key)
#    print(Rel_Geom_Decoder.give_dict()[key]==Geom_Decoder.give_dict()[key])

#Compare parameters:
for parameter in Rel_Geom_Decoder.decoder.parameters():
    print(parameter.flatten()[0])
for parameter in Geom_Decoder.decoder.parameters():
    print(parameter.flatten()[0])
'''

#4. Saving and loading Steerable_CNP - control whether is equal:
Encoder=Steerable_Encoder(x_range=[-1,2],y_range=[3,4],n_x_axis=11,n_y_axis=7,kernel_dict={'kernel_type': 'rbf','sigma_var': 2},
                          l_scale=0.6,normalize=False)
kernel_sizes=[5,7,9,3]
list_n_channels=[3,5,7,8,3]
non_linearity=["ReLU","ReLU","ReLU"]
Conv_Decoder=CNN_Decoder(list_n_channels,kernel_sizes,non_linearity)
CNP_Model=Steerable_CNP(encoder=Encoder,decoder=Conv_Decoder,dim_cov_est=1,l_scale=0.33,normalize_output=False)
CNP_Model.save_model_dict(filename="Test_CNP_1")
Rel_CNP_Model=Steerable_CNP.load_model_from_dict("Test_CNP_1")
for key in CNP_Model.give_dict().keys():
    print(key)
    print(CNP_Model.give_dict()[key])
    print(Rel_CNP_Model.give_dict()[key])
#Compare parameters:
#for parameter in Rel_CNP_Model.parameters():
#    print(parameter.flatten()[0])
#for parameter in CNP_Model.parameters():
#    print(parameter.flatten()[0])
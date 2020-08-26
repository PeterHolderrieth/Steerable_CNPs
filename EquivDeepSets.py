#LIBRARIES:
#Tensors:
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



#HYPERPARAMETERS and set seed:
torch.set_default_dtype(torch.float)

class EquivDeepSet(nn.Module):
    def __init__(self, grid_size=1.,l_scale=1.,normalize=True):
        super(EquivDeepSet, self).__init__()
        '''
        Inputs:
            grid_size: float - grid size for the grid where we smooth the context set on
            l_scale: float - initialisation of length scale for the kernel
            normalize: boolean - indicates whether feature channels is divided by density channel
        '''
        #-------------------------SET PARAMETERS-----------------
        #Bool whether to normalize:
        self.normalize=normalize
        self.grid_size=grid_size

        #Kernel parameters:
        self.kernel_type="rbf"
        self.log_l_scale=nn.Parameter(torch.log(torch.tensor(l_scale,dtype=torch.get_default_dtype())),requires_grad=True)
        #-------------------------SET PARAMETERS FINISHED-----------------
        
        #-------------------------CONTROL PARAMETERS-----------------

        #To prevent a clash, 'B' and 'l_scale' should not be in kernel_dict:        
        if not isinstance(l_scale,float) or l_scale<=0:
            sys.exit("Encoder error: l_scale not correct.")
        if not isinstance(normalize,bool):
            sys.exit("Encoder error: normalize is not a bool.")       
        #-------------------------CONTROL PARAMETERS FINISHED-----------------

    #Expansion of a label vector y in the embedding: y -> (1,y):
    def expand_with_ones(self,Y):
        '''
        Input: Y - torch.Tensor - shape (batch_size,n,C)
        Output: torch.Tensor -shape (batch_size,n,C+1) - added a column of ones to Y (at the start) Y[i,j]<--[1,Y[i,j]]
        '''
        return(torch.cat([torch.ones([Y.size(0),Y.size(1),1],device=Y.device),Y],dim=2))
    
    def get_surrounding_grid(X_c,X_t=None):
        '''
        Inputs:
            X_c,X_t -torch.Tensor - shape (*,2) - context and target coordinates
        Outputs:
            torch.Tensor - shape (n,2) - a 2d grid including all context and target points
        '''
        if X_t is None:
            X_t=X_c

        #Get the range of points:
        x1_min=min(torch.min(X1[:,0]),torch.min(X2[:,0]))
        x2_min=min(torch.min(X1[:,1]),torch.min(X2[:,1]))
        x1_max=max(torch.max(X1[:,0]),torch.max(X2[:,0]))
        x2_max=max(torch.max(X1[:,1]),torch.max(X2[:,1]))

        #Get the grid vectors in the x1- and x2-axis:
        x1_grid_vec=torch.arange(x1_min,x1_max+self.grid_size,self.grid_size)
        x2_grid_vec=torch.arange(x2_min,x2_max+self.grid_size,self.grid_size)
        X2,X1=torch.meshgrid(x2_grid_vec,x1_grid_vec)
        Z=torch.stack((X2,X1),2)
        return(Z)

    def forward(self,X_c,Y_c,X_t=None):
        '''
        Inputs:
            X_c: torch.Tensor - shape (batch_size,n,2)
            Y: torch.Tensor - shape (batch_size,n,self.dim_Y)
        Outputs:
            torch.Tensor - shape (batch_size,self.dim_Y+1,self.n_y_axis,self.n_x_axis)
        '''
        #DEBUG: Control whether X and the grid are on the same device:
        if self.grid.device!=X_c.device:
            print("Grid and X are on different devices.")
            self.grid=self.grid.to(X_c.device)
        
        #Get batch size:
        batch_size=X_c.size(0)
        #Compute the length scale out of the log-scale (clamp for numerical stability):
        l_scale=torch.exp(torch.clamp(self.log_l_scale,max=5.,min=-5.))
        
        #Get a grid including the context and the target:
        grid=self.get_surrounding_grid(X_c,X_t)
        flat_grid=grid.view(-1,2)

        #Compute for every grid-point x' the value k(x',x_i) for all x_i in the data-->shape (batch_size,self.n_y_axis*self.n_x_axis,n)
        Gram=GP.Batch_Gram_matrix(flat_grid.unsqueeze(0).repeat((batch_size,1,1,1)),
                                  X_c,l_scale=l_scale,kernel_type=self.kernel_type,B=torch.ones((1),device=X.device))
        
        #Compute feature expansion --> shape (batch_size,n,self.dim_Y+1)
        Expand_Y=self.expand_with_ones(Y)

        #Compute feature map -->shape (self.n_y_axis*self.n_x_axis,self.dim_Y+1)
        Feature_Map=torch.matmul(Gram,Expand_Y)

        #If wanted, normalize the weights for the channel which is not the density channel:
        if self.normalize:
            Feature_Map[:,:,1:]=Feature_Map[:,:,1:]/Feature_Map[:,:,0].unsqueeze(2)
            #Reshape the Feature Map to the form (1,n_channels=3,n_y_axis,n_x_axis) (because this is the form required for a CNN):
        
        return(Feature_Map.reshape(batch_size,self.n_y_axis,self.n_x_axis,Expand_Y.size(2)).permute(dims=(0,3,1,2)))        
  
    
    def plot_embedding(self,Embedding,X_context=None,Y_context=None,title=""):
        '''
        Input: 
               Embedding - torch.tensor - shape (3,self.n_y_axis,self.n_x_axis) - Embedding obtained from self.forward (usually  from X_context,Y_context)
                                                                      where Embedding[:,0] is the density channel
                                                                      and Embedding[:,1:] is the smoothing channel
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
            'grid_size': self.grid_size,
            'l_scale': torch.exp(self.log_l_scale).item(),
            'normalize': self.normalize
        }
        return(dictionary)
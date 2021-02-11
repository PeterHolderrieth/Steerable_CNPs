#LIBRARIES:
#Tensors:

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

#E(2)-steerable CNNs - library:
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
import kernel_and_gp_tools as GP
import my_utils



#HYPERPARAMETERS and set seed:
torch.set_default_dtype(torch.float)


class EquivEncoder(nn.Module):
    def __init__(self, x_range,n_x_axis,y_range=None,n_y_axis=None,
                 l_scale=1.,normalize=True,train_l_scale=False):
        super(EquivEncoder, self).__init__()
        '''
        Inputs:
            x_range,y_range: float lists of size 2 - give range of grid points at x-axis/y-axis
            n_x_axis: int - number of grid points along the x-axis
            n_y_axis: int - number of grid points along the y-axis
            l_scale: float - initialisation of length scale
            normalize: boolean - indicates whether feature channels is divided by density channel
        '''
        #-------------------------SET PARAMETERS-----------------
        #Save whether to normalize and train l scale:
        self.normalize=normalize
        self.train_l_scale=train_l_scale
        
        #Kernel parameters:
        self.kernel_type="rbf"
        self.log_l_scale=nn.Parameter(torch.log(torch.tensor(l_scale,dtype=torch.get_default_dtype())),requires_grad=train_l_scale)

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
        self.grid=nn.Parameter(my_utils.give_2d_grid(min_x=self.x_range[0],max_x=self.x_range[1],
                               min_y=self.y_range[1],max_y=self.y_range[0],
                               n_x_axis=self.n_x_axis,n_y_axis=self.n_y_axis,flatten=True),requires_grad=False)
            
        #-------------------------SET PARAMETERS FINISHED-----------------
        
        #-------------------------CONTROL PARAMETERS-----------------
        if not isinstance(l_scale,float) or l_scale<=0:
            sys.exit("Encoder error: l_scale not correct.")
        if self.x_range[0]>=self.x_range[1] or self.y_range[0]>=self.y_range[1]:
            sys.exit("x and y range are not valid.")
        #-------------------------CONTROL PARAMETERS FINISHED-----------------

    #Function to add a one to every vector: y->(1,y):
    def expand_with_ones(self,Y):
        '''
        Input: Y - torch.Tensor - shape (batch_size,n,C)
        Output: torch.Tensor -shape (batch_size,n,C+1) - added a column of ones to Y (at the start) Y[i,j]<--[1,Y[i,j]]
        '''
        return(torch.cat([torch.ones([Y.size(0),Y.size(1),1],device=Y.device),Y],dim=2))

    def forward(self,X,Y):
        '''
        Inputs:
            X: torch.Tensor - shape (batch_size,n,2)
            Y: torch.Tensor - shape (batch_size,n,dim_Y)
        Outputs:
            torch.Tensor - shape (batch_size,dim_Y+1,self.n_y_axis,self.n_x_axis)
        '''
        #DEBUG: Control whether X and the grid are on the same device:
        if self.grid.device!=X.device:
            self.grid=self.grid.to(X.device)
        
        #Get the batch size:
        batch_size=X.size(0)
        #Compute the length scale out of the log-scale (clamp for numerical stability):
        l_scale=torch.exp(self.log_l_scale)#torch.clamp(self.log_l_scale,max=5.,min=-5.))
        
        #Compute for every grid-point x' the value k(x',x_i) for all x_i in the data-->shape (batch_size,self.n_y_axis*self.n_x_axis,n)
        Gram=GP.batch_gram_matrix(self.grid.unsqueeze(0).expand(batch_size,self.n_y_axis*self.n_x_axis,2),
                                  X,l_scale=l_scale,kernel_type=self.kernel_type,B=torch.ones((1),device=X.device))
        
        #Compute feature expansion --> shape (batch_size,n,self.dim_Y+1)
        Expand_Y=self.expand_with_ones(Y)
        #Compute feature map -->shape (self.n_y_axis*self.n_x_axis,self.dim_Y+1)
        Feature_Map=torch.matmul(Gram,Expand_Y)

        #If wanted, normalize the weights for the channel which is not the density channel:
        if self.normalize:
            Feature_Map[:,:,1:]=Feature_Map[:,:,1:]/Feature_Map[:,:,0].unsqueeze(2)
        
        #Reshape the Feature Map to the form (batch_size,dim_Y+1,self.n_y_axis,self.n_x_axis) (because this is the form required for a an EquivCNN):
        return(Feature_Map.reshape(batch_size,self.n_y_axis,self.n_x_axis,Expand_Y.size(2)).permute(dims=(0,3,1,2)))     
    
    def plot_embedding(self,Embedding,X_context=None,Y_context=None,title="",quiver_scale=1.,size_scale=2):
        '''
        Input: 
               Embedding - torch.Tensor - shape (3,self.n_y_axis,self.n_x_axis) - Single Embedding obtained from self.forward (usually  from X_context,Y_context)
                                                                                    where Embedding[:,0] is the density channel
                                                                                    and Embedding[:,1:] is the smoothing channel
               X_context,Y_context - torch.Tensor - shape (n,2) - context locations and vectors
               title - string - title of plots
               quiver_scale, size_scale - hyperparameters for size of arrows and figures for plots
        Plots locations X_context with vectors Y_context attached to it
        and on top plots the kernel smoothed version (i.e. channel 2,3 of the embedding)
        Moreover, it plots a density plot (channel 1 of the embedding).
        '''
        #----------CREATE FIGURE --------------
        #Create figures, set title and adjust space:
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(size_scale*10,size_scale*5))
        #plt.gca().set_aspect('equal', adjustable='box')
        fig.suptitle(title)
        fig.subplots_adjust(wspace=0.2)
        #----------END CREATE FIGURE-------

        #---------SMOOTHING PLOT---------
        #Set titles for subplots:
        ax[0].set_title("Smoothing channel")
        
        #Get density channel --> shape (self.n_y_axis,self.n_x_axis)
        Density=Embedding[0]
        #Get Smoothed channel -->shape (self.n_y_axis*self.n_x_axis,2)
        Smoothed=Embedding[1:].permute(dims=(1,2,0)).reshape(-1,2)
        #Plot the kernel smoothing:
        ax[0].quiver(self.grid[:,0],self.grid[:,1],Smoothed[:,0],Smoothed[:,1],color='black',pivot='mid',label='Smoothing Channel',scale=quiver_scale,scale_units='inches',alpha=0.7)
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
        ax[1].contourf(X,Y, Density, levels=14, linewidths=0.5, cmap=cm.get_cmap('cividis'),aspect='auto',label='Density Channel')

        #Add legend to first plot:
        #leg = ax[0].legend(loc='upper right')
        #---------END DENSITY PLOT--------

        #Add context points to the smoothing and density channel:
        if X_context is not None and Y_context is not None:
            ax[0].quiver(X_context[:,0],X_context[:,1],Y_context[:,0],Y_context[:,1],
              color='firebrick',pivot='mid',label='Context set',scale=quiver_scale,scale_units='inches')
            ax[1].scatter(X_context[:,0],X_context[:,1],color='firebrick',label='Context set')
            
    #A function to save a dictionary dict such that EquivEncoder(**dict) will return the same object.
    #This is used to save an instance of the class.     
    def give_dict(self):
        dictionary={
            'x_range':self.x_range,
            'n_x_axis':self.n_x_axis,
            'y_range':self.y_range,
            'n_y_axis':self.n_y_axis,
            'l_scale': torch.exp(self.log_l_scale).item(),
            'normalize': self.normalize,
            'train_l_scale': self.train_l_scale
        }
        return(dictionary)


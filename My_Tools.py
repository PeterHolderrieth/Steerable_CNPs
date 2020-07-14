# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:49:27 2020

@author: 49157
"""
#Libraries:
#Tensors:
import torch
import torch.utils.data as utils
import torch.nn.functional as F
import numpy as np

#Plotting in 2d/3d:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
import matplotlib.cm as cm

#Tools:
from itertools import product, combinations
import sys
import math
from numpy import savetxt
import csv

#E(2)-steerable CNNs - librar"y:
from e2cnn import gspaces    
import e2cnn
from e2cnn import nn as G_CNN   

#HYPERPARAMETERS:
#Set default as double:
torch.set_default_dtype(torch.float)
#Scale for plotting with plt quiver
quiver_scale=15

#%%
'''
____________________________________________________________________________________________________________________

----------------------------General Tools---------------------------------------------------------------------
____________________________________________________________________________________________________________________
'''
#Tool to split a function in a context and target set (choice is random but size of context set is given):
def Rand_Target_Context_Splitter(X,Y,n_context_points):
    '''
    Inputs: X: torch.tensor - shape (n,d) - n...number of observations, d...dimension of state space
            Y: torch.tensor - shape (n,D) - N...number of observations, D...dimension of label space
            n_context_points: int - size of context set
    Outputs:
        X_Context: torch.tensor - shape (n_context_points,d)
        Y_Context: torch.tensor - shape (n_context_points,D)
        X_Target:  torch.tensor - shape (n-n_context_points,d)
        Y_Target:  torch.tensor - shape (n-n_context_points,D)
    '''
    n=X.size(0)
    ind_shuffle=torch.randperm(n)
    X_Context=X[ind_shuffle[:n_context_points],]
    Y_Context=Y[ind_shuffle[:n_context_points],]
    X_Target=X[ind_shuffle[n_context_points:,]]
    Y_Target=Y[ind_shuffle[n_context_points:,]]
    return(X_Context,Y_Context,X_Target,Y_Target)

def get_outer_circle_indices(n):
    '''
    n - int - size of square
    '''
    x_axis=torch.linspace(start=0,end=n-1,steps=n)
    y_axis=torch.linspace(start=0,end=n-1,steps=n)
    X1,X2=torch.meshgrid(x_axis,y_axis)
    Ind=torch.stack((X1,X2),2)
    Ind=Ind[torch.norm(Ind-(n-1)/2,dim=2)>(n-1)/2].long()
    return(Ind)
    

#%%
          
'''
____________________________________________________________________________________________________________________

----------------------------2d Tools ---------------------------------------------------------------------
____________________________________________________________________________________________________________________
'''   
#A tool to make a 2d_grid:
def Give_2d_Grid(min_x,max_x,n_x_axis,min_y=None,max_y=None,n_y_axis=None,flatten=True):
    '''
    Input:
        min_x,max_x,min_y,max_y: float - range of x-axis/y-axis
        n_x_axis,n_y_axis: int - number of points per axis
        flatten: Boolean - determines shape of output
    Output:
        torch.tensor - if flatten is True: shape (n_x_axis*n_y_axis,2)
                       if flatten is not True: shape (n_x_axis,n_y_axis,2)
    '''
    if min_y is None:
        min_y=min_x
    if max_y is None:
        max_y=max_x
    if n_y_axis is None:
        n_y_axis=n_x_axis
        
    x_grid_vec=torch.linspace(min_x,max_x,n_x_axis)
    y_grid_vec=torch.linspace(min_y,max_y,n_y_axis)
    X1,X2=torch.meshgrid(x_grid_vec,y_grid_vec)
    X=torch.stack((X1,X2),2)
    if flatten:
        X=X.view(n_x_axis*n_y_axis,2)
    return(X)
           
#Tool to plot context set, ground truth for target and predictions for target in one plot:
def Plot_Inference_2d(X_Context,Y_Context,X_Target=None,Y_Target=None,Predict=None,Cov_Mat=None,title=""):
    '''
    Inputs: X_Context,Y_Context: torch.tensor - shape (n_context_points,2) - given context set
            X_Target,Y_Target: torch.tensor - shape (n_target_points,2) - target locations and outputs vectors 
            Predict: torch.tensor - shape (n_target_points,2) - target predictions (i.e. predicting Y_Target)
            Cov_Mat: torch.tensor - shape (n_target_points,2,2) - set of covariance matrices
                                  or - shape (n_target_points,2) - set of variances 
            title: string -  suptitle
    Outputs: None - plots the above tensors in two plots: one plots the means against the context sets, the other one the covariances/variances
    '''
    #Function hyperparameters for plotting in notebook:
    size_scale=2
    ellip_scale=0.8
    
    #Create plots:
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(size_scale*10,size_scale*5))
    plt.gca().set_aspect('equal', adjustable='box')
    fig.subplots_adjust(wspace=0.4)
    #Set titles:
    fig.suptitle(title)
    ax[0].set_title("Context set and predictions")
    ax[1].set_title("Variances")
    
    #Plot context set in blue:
    ax[0].scatter(X_Context[:,0],X_Context[:,1],color='black')
    ax[0].quiver(X_Context[:,0],X_Context[:,1],Y_Context[:,0],Y_Context[:,1],
      color='black',pivot='mid',label='Context set',scale=quiver_scale)
    
    #Plot ground truth in red if given:
    if Y_Target is not None and X_Target is not None:       
        ax[0].quiver(X_Target[:,0],X_Target[:,1],Y_Target[:,0],Y_Target[:,1],
          color='blue',pivot='mid',label='Ground truth',scale=quiver_scale)
    
    #Plot predicted means in green:
    if  Predict is not None and X_Target is not None:
        ax[0].quiver(X_Target[:,0],X_Target[:,1],Predict[:,0],Predict[:,1],color='red',pivot='mid',label='Predictions',scale=quiver_scale)

    leg = ax[0].legend(bbox_to_anchor=(1., 1.))
    
    if X_Target is not None and Cov_Mat is not None:
        #Get window limites for plot and set window for second plot:
        ax[1].set_xlim(ax[0].get_xlim())
        ax[1].set_ylim(ax[0].get_ylim())
        ax[1].scatter(X_Context[:,0],X_Context[:,1],color='black',label='Context Points',marker='+')
        #Go over all target points and plot ellipse of continour lines of density of distributions:
        for j in range(X_Target.size(0)):
            #Get covarinace matrix:
            A=Cov_Mat[j]
            if len(A.size())==1:
                A=torch.diag(A)
            
            #Decompose A:
            eigen_decomp=torch.eig(A,eigenvectors=True)
            #Get the eigenvector corresponding corresponding to the largest eigenvalue:
            u=eigen_decomp[1][:,0]

            #Get the angle of the ellipse in degrees:
            alpha=360*torch.atan(u[1]/u[0])/(2*math.pi)
        
            #Get the width and height of the ellipses (eigenvalues of A):
            D=torch.sqrt(eigen_decomp[0][:,0])
        
            #Plot the Ellipse:
            E=Ellipse(xy=X_Target[j,].numpy(),width=ellip_scale*D[0],height=ellip_scale*D[1],angle=alpha)
            ax[1].add_patch(E)
            
        #Create a legend:
        blue_ellipse = Ellipse(color='c', label='Confid. ellip.',xy=0,width=1,height=1)
        ax[1].legend(handles=[blue_ellipse])
        leg = ax[1].legend(bbox_to_anchor=(1., 1.))

#%%
'''
____________________________________________________________________________________________________________________

---------------------------- Spherical Tools ---------------------------------------------------------------------
____________________________________________________________________________________________________________________
'''
#This function project a bunch of vectors Y onto the tangent space of X.
#We assume that all element in X have norm 1 (i.e. are on the sphere).
#(Otherwise we would need to normalize)
def Tangent_Sphere_Projector(X,Y):
    '''
    Input:
        X torch.tensor
          Shape (n,d)
          X[i,] has norm 1 for all i
        Y torch.tensor
          Shape (n,d)
    Output:
        Z torch.tensor
        Shape (n)
        Z[i]=Projection of Y[i,] on orthogonal space of X[i,]
    '''
    #Compute dot products:
    Dot_Products=torch.diag(torch.mm(X,Y.t()))
    n=X.size(0)
    #Compute projections:
    Z=Y-Dot_Products.view(n,1)*X
    return(Z)
    
#The following function gives a grid over sphere - where the grid is based on
#a grid in spherical coordinates:
def Give_Spherical_Grid(n_grid_points=10,flatten=True):
    '''
    Input: n_grid_points - int - (half) number of grid points per angle 
           flatten - Boolean 
    Output: torch.tensor - shape (2*n_grid_points**2,3) if flatten is True
                         - shape (2*n_grid_points,n_grid_points,3) if flatten is False
    '''
    #Take a grid over angles:
    u, v = np.mgrid[0:2*np.pi:(n_grid_points*2j), 0:np.pi:(n_grid_points*1j)]
    #Compute corresponding value in 3d and stack them to a single data matrix:
    X1 = torch.tensor(np.cos(u)*np.sin(v))
    X2 = torch.tensor(np.sin(u)*np.sin(v))
    X3 = torch.tensor(np.cos(v))
    X=torch.stack((X1,X2,X3),2)
    if flatten:
        X=X.view(-1,3)
    return(X)
    

'''
____________________________________________________________________________________________________________________

----------------------------Matrix Tools ---------------------------------------------------------------------
____________________________________________________________________________________________________________________
'''
#A function to get the block diagonal matrix from a matrix:
def Get_Block_Diagonal(X,size=1):
    '''
    Input: X - torch.tensor - shape (n,n)
           size - int - size divides n
    Output: torch.tensor - shape (n/size,size*size) - block diagonals of X
    '''
    m=X.size(0)//size
    Sigma=torch.empty((m,size,size))
    for i in range(m):
        Sigma[i]=X[(i*size):(i+1)*size,(i*size):(i+1)*size]
    return Sigma

#A function to create a matrix from block matrices (merges the block matrices):
def Create_matrix_from_Blocks(X):
    '''
    Input:
        X - torch.tensor - shape (n,m,D_1,D_2)
    Output:
        torch-tensor - shape (n*D_1,m*D_2) - block (i,j) of size D_1*D_2 is matrix X[i,j] for i=1,...,n,j=1,...,m
    '''
    n=X.size(0)
    m=X.size(1)
    D_1=X.size(2)
    D_2=X.size(3)
    M=torch.cat([X[:,:,i,:].reshape(n,m*D_2) for i in range(D_1)])
    ind=torch.cat([torch.arange(i,D_1*n,n,dtype=torch.long) for i in range(n)])
    return(M[ind])

#Activation function to get a covariance matrix - apply softplus or other activation functions on eigenvalues:    
def plain_cov_activation_function(X,activ_type="softplus"):
    '''
    Input:
    X- torch.tensor - shape (n,3)
    activ_type - string -type of activation applied on the diagonal matrix
    Output: torch.tensor - shape (n,2,2) - consider X[i] as a symmetric 2x2 matrix
            we compute the eigendecomposition X[i]=UDU^T of that matrix and 
            if s is a an function, we apply it componentwise on the diagonal s(D)
            and return Us(D)U^T (in one batch)
    '''
    n=X.size(0)
    M=torch.stack([X[:,0],X[:,1],X[:,1],X[:,2]],dim=1).view(n,2,2)
    
    eigen_vals,eigen_vecs=torch.symeig(M,eigenvectors=True)
    if activ_type=="softplus":
        eigen_vals=F.softplus(eigen_vals)
    else: 
        sys.exit("Unknown activation type")
    return(torch.matmul(eigen_vecs, torch.matmul(eigen_vals.diag_embed(), eigen_vecs.transpose(-2, -1))))
        
def stable_cov_activation_function(X,activ_type="softplus",tol=1e-7):
    '''
    Input:
            X- torch.tensor - shape (n,3)
            activ_type - string -type of activation applied on the diagonal matrix
            tol - float - giving tolerance level, i.e. the distance to a pure scalar matrix s*Id,
                          if the distance is below that, no eigendecomposition is computed but instead
                          the covariance matrix is computed directly.
    Output: 
            torch.tensor - shape (n,2,2) - consider X[i] as a symmetric 2x2 matrix
            we compute the eigendecomposition X[i]=UDU^T of that matrix and 
            if s is a an function, we apply it componentwise on the diagonal s(D)
            and return Us(D)U^T (in one batch)
            The difference to "plain_cov_activation_function" is that we only compute the eigendecomposition
            for matrices which are not of the form s*Id for s in R but immediately use that (up to some error)
            This makes the function fully differentiable (however, the gradient is slightly changed)
    '''
    n=X.size(0)
    Out=torch.zeros([n,2,2],device=X.device)   
    below_tol=(torch.abs(X[:,1])<tol)&(torch.abs(X[:,0]-X[:,2])<tol)
    above_tol=~below_tol
    if any(above_tol):
        Out[above_tol]=plain_cov_activation_function(X[above_tol],activ_type=activ_type)
    
    if activ_type=="softplus":
        Out[below_tol,0,0]=F.softplus(X[below_tol][:,0])
        Out[below_tol,1,1]=F.softplus(X[below_tol][:,2])
    else: 
        sys.exit("Unknown activation type")
    return(Out)

def batch_multivar_log_ll(Means,Covs,Data,stupid=True):
    '''
    Input:
        Means - torch.tensor - shape (n,D) - Means 
        Covs - torch.tensor - shape (n,D,D) - Covariances
        Data - torch.tensor - shape (n,D) - observed data 
    Output:
        torch.tensor - shape (n) - log-likelihoods of observations
    '''
    n,D=Means.size()
    Diff=Data-Means
    Quad_Term=torch.matmul(Diff.unsqueeze(1),torch.matmul(Covs.inverse(),Diff.unsqueeze(2))).squeeze()
    log_normalizer=-0.5*torch.log(((2*math.pi)**D)*Covs.det())
    log_ll=log_normalizer-0.5*Quad_Term
    return(log_ll)
    
def get_pre_psd_rep(G_act):
    '''
        Input:
            G_act - instance of e2cnn.gspaces.r2.rot2d_on_r2.Rot2dOnR2 - underlying group
            
        Output:
            psd_rep - instance of e2cnn.group.Representation - group representation of the group representation before the covariance 
            feat_type_pre_rep - instance of G_CNN.FieldType - corresponding field type
    '''
    
    change_of_basis=np.array([[1,1.,0.],
                          [0.,0.,1.],
                          [1,-1.,0.]])
    if G_act.name=='4-Rotations':
        psd_rep=e2cnn.group.Representation(group=G_act.fibergroup,name="psd_rep",irreps=['irrep_0','irrep_2','irrep_2'],
                                   change_of_basis=change_of_basis,
                                   supported_nonlinearities=['n_relu'])
    elif G_act.name=='8-Rotations' or G_act.name=='16-Rotations':
        psd_rep=e2cnn.group.Representation(group=G_act.fibergroup,name="psd_rep",irreps=['irrep_0','irrep_2'],
                                   change_of_basis=change_of_basis,
                                   supported_nonlinearities=['n_relu'])
    else:
        sys.exit("Group not enabled for a pre psd representation yet.")
    feat_type_pre_rep=G_CNN.FieldType(G_act,[psd_rep])
    return(psd_rep,feat_type_pre_rep)

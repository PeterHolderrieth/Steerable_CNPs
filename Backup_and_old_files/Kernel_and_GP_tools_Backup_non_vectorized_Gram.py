#!/usr/bin/env python
# coding: utf-8

# To do:
# 1. Define GP inference function for Sphere/3d
# 2. Export various data sets in 3d
# 3. Insert in a GitHub Repo

# In[22]:


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
#get_ipython().run_line_magic('matplotlib', 'inline')

#Tools:
from itertools import product, combinations
import sys
import math
from numpy import savetxt
import csv
import datetime

#Own files:
import My_Tools


# In[23]:


#HYPERPARAMETERS:
#Set default as double:
torch.set_default_dtype(torch.float)


# In[24]:


#This function gives different known ONE-dimensional kernels:
def kernel(x1,x2,l_scale=1,sigma_var=1,kernel_type="rbf"):
    '''
    Input:
    x1,x1: torch.tensor 
           Shape (d) - d dimension of state space
    l_scale: scalar 
             length scale for RBF kernel
    sigma_var: scalar 
               var for RBF kernel
    kernel_type: string
                 Type of kernel we are using
    Ouput:
        torch.tensor
        Shape: Scalar
        kernel value K(x1,x2)
    '''
    #RBF Kernel:
    if (kernel_type=="rbf"):
        diff=torch.sum(torch.pow(x1-x2,2))
        return(sigma_var*torch.exp(-0.5*diff/l_scale))
    
    #White noise kernel:
    elif (kernel_type=="white_noise"):
        if (x1==x2):
            return(sigma_var)
        else:
            return(torch.tensor(0))
    
    #Dot product:    
    elif (kernel_type=="dot product"):
        return(torch.dot(x1,x2))
    
    #Else error:
    else:
        sys.exit("No valid kernel")


# In[25]:


#This function computes a MATRIX-valued kernel: 
#kernel_type either gives:
#1. A special matrix-valued kernel
#2. A one-dimensional kernel type. In this case, we compute the separable kernel K(x,y)=k(x,y)*B where B is a fixed matrix.
#If B is None, we set B=I_d where d is the dimension of x1,x2

#we also allow the Kernel to be projected on the orthogonal spaces of x,y
#Ker_project indicates that.

#The divergence free kernel used here can be found in "Kernels for Vector-valued functions: a Review" by  Alvarez et al

def mat_kernel(x1,x2,l_scale=1,sigma_var=1,kernel_type="rbf",B=None,Ker_project=False):
    '''
    Input:
    x1,x2: torch.tensor 
           Shape (d) - d dimension of state space
    l_scale: scalar 
             length scale for RBF kernel
    sigma_var: scalar 
               var for RBF kernel
    kernel_type: string
                 Type of kernel we are using
    B: torch.tensor or None
       Shape: (D,D) 
    project: Boolean
             Indicates whether we return the projected kernel (I-xx^T)K(x,y)(I-yy^T)
    Ouput:
        torch.tensor
        Shape: [D,D] where D is number of rows of B 
        Value of separable kernel: k(x1,x2)*B
        
    '''
    #Outer product:
    if kernel_type=="outer":
        K=torch.ger(x1,x2)/l_scale**2
    
    #Divergence free kernel:
    elif kernel_type=="div_free":
        d=x1.size(0)
        diff2=torch.sum(torch.pow(x1-x2,2))
        scalar=torch.exp(-0.5*diff2/l_scale)/l_scale
        A_x1_x2=torch.ger(x1-x2,x1-x2)/l_scale+((d-1)-diff2/l_scale)*torch.eye(d)
        K=scalar*A_x1_x2
        
    elif kernel_type=="exp_div_free":
        return(torch.exp(-mat_kernel(x1,x2,l_scale=l_scale,kernel_type="div_free",B=B,Ker_project=Ker_project)/l_scale))
    #Separable kernel:
    else:
        #Dimension of input space:
        d=x1.size(0)
        if B is None:
            B=torch.eye(d)
        K=kernel(x1,x2,l_scale,sigma_var,kernel_type)*B
        d=x1.size(0)
    
    #Either project...
    if Ker_project:
        #B needs to be of the same dimension:
        if B is not None:
            if B.size(0)!=d:
                sys.exit("Projection needs B to of same of dimension than x1,x2")
        #L (resp. R) Projection on orthogonal space of x1 (resp x2)
        L=torch.eye(d)-torch.ger(x1,x1)/torch.norm(x1)**2
        R=torch.eye(d)-torch.ger(x2,x2)/torch.norm(x2)**2
        return(torch.mm(torch.mm(L,K),R))
    
    #....or don't project:
    else:
        return(K)
        
# In[26]:


#This function computes the Gram-Matrix of a data X based on the function mat_kernel
#If two data sets X,Y are given: it computes K(X,Y) (covariance matrix)

#Note that: Here, we implicitly assume that the kernel is either separable or if not
#the dimension of the matrix-valued kernel is the same as the dimension of the data space.
def Gram_matrix(X,Y=None,l_scale=1,sigma_var=1, kernel_type="rbf",B=None,Ker_project=False):
    '''
    Input:
    X: torch.tensor
          Shape: (n,d)...n number of obs, d...dimension of state space
    Y: torch.tensor or None
          Shape: (m,d)...m number of obs, d...dimension of state space 
    l_scale,sigma_var,kernel_type,B,Ker_project: see function "mat_kernel"

    Output:
    Gram_matrix: torch.tensor
                 Shape (n*D,n*D) (if Y is given (n*D,m*D))
                 Block i,j of size dxd gives Kernel value of i-th X-data point and
                 j-th Y data point
    '''
    d=X.size(1)
    n=X.size(0)
    if (B is None):
        B=torch.eye(d)
    if (Y is None):
        Y=X
    m=Y.size(0)
    D=B.size(0)
    #Create empty matrix:
    Gram_matrix=torch.empty([n*D,m*D])
    #Fill matrix block-wise:
    for i in range(n):
        for j in range(m):
            Gram_matrix[D*i:(D*i+D),D*j:(D*j+D)]=mat_kernel(x1=X[i,],x2=Y[j,],l_scale=l_scale,sigma_var=sigma_var,kernel_type=kernel_type,B=B,Ker_project=Ker_project)
    return(Gram_matrix)
#%%
def Create_matrix_from_Blocks(X):
    '''
    X - torch.tensor - shape (n,m,D_1,D_2)
    '''
    n=X.size(0)
    m=X.size(1)
    D_1=X.size(2)
    D_2=X.size(3)
    M=torch.cat([X[:,:,i,:].reshape(n,m*D_2) for i in range(D_1)])
    ind=torch.cat([torch.arange(i,D_1*n,n,dtype=torch.long) for i in range(n)])
    return(M[ind])
#%%
def Gram_matrix_vectorized(X,Y=None,l_scale=1,sigma_var=1, kernel_type="rbf",B=None,Ker_project=False,flatten=True):
    '''
    Input:
    X: torch.tensor
          Shape: (n,d)...n number of obs, d...dimension of state space
    Y: torch.tensor or None
          Shape: (m,d)...m number of obs, d...dimension of state space 
    l_scale,sigma_var,kernel_type,B,Ker_project: see function "mat_kernel"

    Output:
    Gram_matrix: torch.tensor
                 Shape (n,m,D) (if Y is not given (n,n,D))
                 Block i,j of size DxD gives Kernel value of i-th X-data point and
                 j-th Y data point
    '''
    #Get dimension of data space and number of observations from X:
    d=X.size(1)
    n=X.size(0)
    #If B is not given, set to identity:
    if (B is None):
        B=torch.eye(d)
    #If Y is not given, set to X:
    if (Y is None):
        Y=X
    #Get number of observations from Y and dimension of B:
    m=Y.size(0)
    D=B.size(0)
    #RBF kernel:
    if kernel_type=="rbf":
        #Expand X,Y along different dimension to get a grid shape:
        X=X.unsqueeze(1).expand(n,m,d)
        Y=Y.unsqueeze(0).expand(n,m,d)
        #Compute the squared distance matrix:
        Dist_mat=torch.sum((X-Y)**2,dim=2)
        #Compute the RBF kernel from that:
        Gram_RBF=sigma_var*torch.exp(-0.5*Dist_mat/l_scale)
        #Expand B such that it has the right shape:
        B=B.view(1,1,D,D)
        B=B.expand(n,m,D,D)
        #Reshape RBF Gram matrix:
        Gram_RBF=Gram_RBF.view(n,m,1,1)
        #Multiply scalar Gram matrix with the matrix B:
        K=Gram_RBF*B

    elif kernel_type=="dot_product":
        #Get dot product Gram matrix:
        Gram_one_d=torch.matmul(X,Y.t())
        #Expand B:
        B=B.view(1,1,D,D)
        B=B.expand(n,m,D,D)
        #Expand one-dimensional Gram:
        Gram_one_d=Gram_one_d.view(n,m,1,1)
        #Multiply with B:
        K=Gram_one_d*B

    elif kernel_type=="div_free":
        X=X.unsqueeze(1).expand(n,m,d)
        Y=Y.unsqueeze(0).expand(n,m,d)
        Dist_mat=torch.sum((X-Y)**2,dim=2)
        Gram_RBF=torch.exp(-0.5*Dist_mat/l_scale)/l_scale
        Gram_RBF=Gram_RBF.view(n,m,1,1)
        Diff=X-Y
        Outer_Prod_Mat=torch.matmul(Diff.unsqueeze(3),Diff.unsqueeze(2))
        Id=torch.eye(d)
        Id=Id.view(1,1,d,d)
        Id=Id.expand(n,m,d,d)
        Mat_1=Outer_Prod_Mat/l_scale
        Mat_2=(d-1-Dist_mat.view(n,m,1,1)/l_scale)*Id
        A=Mat_1+Mat_2
        K=Gram_RBF*A
       
    else:
        sys.exit("Unknown kernel type")
    
    if flatten:
        return(Create_matrix_from_Blocks(K))
    else:
        return(K)
        



#%%
#This function samples a multi-dimensional GP with kernel given by the function mat_kernel:
#Observations are assumed to be noisy versions of the real underlying function:
def Multidim_GP_sampler(X,l_scale=1,sigma_var=1, kernel_type="rbf",B=None,Ker_project=False,chol_noise=1e-4,obs_noise=1e-4):
    '''
    Input:
    X: torch.tensor
       Shape (n,d) n...number of observations, d...dimension of state space
    l_scale,sigma_var,kernel_type,B,Ker_project: Kernel parameters (see mat_kernel)
    chol_noise: scalar - noise added to make cholesky decomposition numerically stable
    obs_noise: variance of observation noise
    
    Output:
    Y: torch.tensor
       Shape (n,D) D...dimension of label space 
       Sample of GP
    '''
    if (B is None):
        d=X.size(1)
        B=torch.eye(d)
    
    #Save dimensions:
    n=X.size(0)
    D=B.size(0)
    
    #Get Gram-Matrix:
    Gram_Mat=Gram_matrix(X,Y=None,l_scale=l_scale,sigma_var=sigma_var, kernel_type=kernel_type,B=B,Ker_project=Ker_project)
    
    #Get cholesky decomposition of Gram-Mat (adding some noise to make it numerically stable):
    L=(Gram_Mat+chol_noise*torch.eye(D*n)).cholesky()
    
    #Get multi-dimensional std normal sample:
    Z=torch.randn(n*D)
    
    #Function values + noise = Observation:
    Y=torch.mv(L,Z)+math.sqrt(obs_noise)*torch.randn(n*D)
    
    #Return reshaped version:
    return(Y.view(n,D))

#%%

#This function gives a GP sample in 2d on an evenly spaced grid:
def vec_GP_sampler_2dim(min_x=-2,max_x=2,n_grid_points=10,l_scale=1,sigma_var=1, 
                        kernel_type="rbf",B=None,Ker_project=False,obs_noise=1e-4):
    '''
    Input: 
    min_x,max_x: scalar - left-right/lower-upper limit of grid
    n_grid_points: int - number of grid points per axis
    l_scale,sigma_var,kernel_type,B, Ker_project: see kernel_mat
    obs_noise: variance of observation noise
    Output:
    X: torch.tensor 
       Shape (n_grid_points**2,2)
    Y: torch.tensor
       Shape (n_grid_points**2,D), D...dimension of label space
    '''
    #Create a grid:
    grid_vec=torch.linspace(min_x,max_x,n_grid_points)
    X1,X2=torch.meshgrid(grid_vec,grid_vec)
    X=torch.stack((X1,X2),2).view(n_grid_points**2,2)
    
    #Sample GP:
    Y=Multidim_GP_sampler(X,l_scale=l_scale,sigma_var=sigma_var, kernel_type=kernel_type,B=B,Ker_project=Ker_project,obs_noise=obs_noise)
    
    #Return grid and samples:
    return(X,Y)


# In[29]:


#This function plots a Gaussian process with Gaussian process on 2d with 2d outputs (i.e. a vector field) 
#in an evenly spaced grid:
def rand_plot_2d_vec_GP(n_plots=4,min_x=-2,max_x=2,n_grid_points=10,l_scale=1,sigma_var=1, kernel_type="rbf",B=None,Ker_project=False,obs_noise=1e-4):
    '''
    Inputs:
        n_plots: int number of samples and plot
        min_x,max_x,n_grid_points: see vec_GP_sampler_2dim
        l_scale,sigma_var,kernel_type,B,Ker_project: See mat_kernel
        obs_noise: float - variance of noise of observations
    Outputs:
        fig,ax of plots (matplotlib objects)
    '''
    #Function hyperparameter for scaling size of function and colors (only for notebook):
    size_scale=2
    colormap = cm.hot
    norm = Normalize()

    #Define figure and subplots, adjust space and title:
    fig, ax = plt.subplots(nrows=n_plots//2,ncols=2,figsize=(size_scale*10,size_scale*5))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Vector field GP samples',fontsize=size_scale*10)
    
    for i in range(n_plots):
        #Create sample:
        X,Y=vec_GP_sampler_2dim(min_x,max_x,n_grid_points,l_scale,sigma_var, kernel_type,B)
        #Scale the color of the vectors based on the length 
        C=-torch.norm(Y,dim=1)
        #Scale colors:
        norm.autoscale(C)
        #Scatter plot of locations:
        ax[i//2,i%2].scatter(X[:,0],X[:,1], color='black', s=2*size_scale)
        #Plots arrows:
        Q = ax[i//2,i%2].quiver(X[:,0],X[:,1],Y[:,0], Y[:,1],color=colormap(norm(C)),units='x', pivot='mid')
        #Subtitle:
        ax[i//2,i%2].set(title="Sample "+str(i))
    return(fig,ax)


# In[30]:


#This function project a bunch of vectors Y onto the tangent space of X.
#We assume that all element in X have norm 1 (i.e. are on the sphere).
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


# In[31]:


#This function samples a GP on the Sphere
#- We take a grid over the sphercal coordinates (angles).
#- The function also returns a version of the GP where Y[i,] is projected on the orthogonal
#  space of X[i,] if Y[i,] and X[i,] are in the same dimension.
def vec_GP_sampler_Sphere(l_scale=1,sigma_var=1, kernel_type="rbf",B=None,Ker_project=False,obs_noise=1e-4):
    '''
    Inputs:
        n_grid_points: Number of grid points of Y-rotation and Half number of grid points for Z-rotation
        l_scale,sigma_var,kernel_type,B,Ker_project: kernel parameters (see function "mat_kernel")
        obs_noise: sd of noise of observations
    Outputs:
        X: torch.tensor - Shape (n,d) - locations 
        Y: torch.tensor - Shape (n,D) - function values
        Proj_V: torch.tensor - Shape (n,d) if D=d (None otherwise) - Projection on orthogonal space
    '''
    #Take a grid over angles:
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    #Compute corresponding value in 3d and stack them to a single data matrix:
    X1 = torch.tensor(np.cos(u)*np.sin(v))
    X2 = torch.tensor(np.sin(u)*np.sin(v))
    X3 = torch.tensor(np.cos(v))
    X=torch.stack((X1,X2,X3),2).view(-1,3)
    
    #Sample a GP at X:
    Y=Multidim_GP_sampler(X,l_scale=l_scale,sigma_var=sigma_var,
                          kernel_type=kernel_type,B=B,
                          Ker_project=Ker_project,obs_noise=obs_noise)
    
    #If possible, compute projection:
    if Y.size(1)==X.size(1):
        Proj_Y=Tangent_Sphere_Projector(X,Y)
    else:
        Proj_Y=None
    
    return(X,Y,Proj_Y)


# In[32]:


#This function samples a GP on the sphere and plots the corresponding vector field.
#It is important to note that we assume here that the dimension of the input AND the output is both 3.
def Plot_Spherical_GP(l_scale=1,sigma_var=1, kernel_type="rbf",B=None,Ker_project=False,obs_noise=1e-4,GP_project=False,):
    '''
    Inputs:
        l_scale,sigma_var,kernel_type,B,Ker_project: kernel parameters (see "mat_kernel")
        obs_noise: variance of noise of observations
        GP_project: Boolean - if True a second plot is created, plotting the GP projected on the tangent space
    Outputs: None - Only plots
    '''
    
    #Function hyperparameter for plotting in notebook (scales figure size)
    size_scale=4

    #Create GP samples on the sphere and projection:
    X,Y,Proj_Y=vec_GP_sampler_Sphere(l_scale,sigma_var, kernel_type,B,Ker_project,obs_noise=obs_noise)
    
    #Create figure:
    fig = plt.figure(figsize=(size_scale*10,size_scale*5))
    ax = fig.gca(projection='3d')
    ax.set_title("GP on the sphere - kernel type: "+kernel_type,fontsize=size_scale*10)
    
    #Plot surface of the sphere:
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color="grey")
    #ax.plot_wireframe(x, y, z, color="blue")
    
    #Plot the GP - locations in red with points- function values as vectors in orange:
    ax.scatter(X[:,0],X[:,1],X[:,2], color='red', s=8*size_scale)
    ax.quiver(X[:,0], X[:,1],X[:,2], Y[:,0], Y[:,1],Y[:,2], length=0.2,color='orange',pivot='middle')
    
    #If wanted, repeat the plot for the projected version:
    if GP_project:
        #Repeat Sphere plot:
        fig2 = plt.figure(figsize=(size_scale*10,size_scale*5))
        ax2 = fig2.gca(projection='3d')
        ax2.set_title("GP projected on tangent space - kernel type: "+kernel_type,fontsize=size_scale*10)
        ax2.plot_surface(x, y, z, color="grey")
        
        #Repeat function plot:
        ax2.scatter(X[:,0],X[:,1],X[:,2], color='red', s=8*size_scale)
        ax2.quiver(X[:,0], X[:,1],X[:,2], Proj_Y[:,0], Proj_Y[:,1],Proj_Y[:,2], length=0.15,color='orange',pivot='middle')


# In[33]:

#This function splits a data set randomly into a target and context set:
def Rand_Target_Context_Splitter(X,Y,n_context_points):
    n=X.size(0)
    ind_shuffle=torch.randperm(n)
    X_Context=X[ind_shuffle[:n_context_points],]
    Y_Context=Y[ind_shuffle[:n_context_points],]
    X_Target=X[ind_shuffle[n_context_points:,]]
    Y_Target=Y[ind_shuffle[n_context_points:,]]
    return(X_Context,Y_Context,X_Target,Y_Target)


# In[34]:


#This functions perform GP-inference on the function values at X_Target (so no noise for the target value)
#based on context points X_Context and labels Y_Context:
def GP_inference(X_Context,Y_Context,X_Target,l_scale=1,sigma_var=1, kernel_type="rbf",obs_noise=1e-4,B=None,Ker_project=False):
    '''
    Input:
        X_Context - torch.tensor - Shape (n_context_points,d)
        Y_Context - torch.tensor- Shape (n_context_points,D)
        X_Target - torch.tensor - Shape (n_target_points,d)
    Output:
        Means - torch.tensor - Shape (n_target_points, D) - Means of conditional dist.
        Cov_Mat- torch.tensor - Shape (n_target_points*D,n_target_points*D) - Covariance Matrix of conditional dist.
        Vars - torch.tensor - Shape (n_target_points,D) - Variance of individual components 
    '''
    #Dimensions of data matrices:
    n_context_points=X_Context.size(0)
    n_target_points=X_Target.size(0)
    D=Y_Context.size(1)
    #Get matrix K(X_Context,X_Context) and add on the diagonal the observation noise:
    Gram_context=(Gram_matrix(X_Context,l_scale=l_scale,sigma_var=sigma_var, kernel_type=kernel_type,B=B,Ker_project=Ker_project)+
                        obs_noise*torch.eye(n_context_points*D))
    
    #Get Gram-matrix K(X_Target,X_Target):
    Gram_target=Gram_matrix(X_Target,l_scale=l_scale,sigma_var=sigma_var, kernel_type=kernel_type,B=B,Ker_project=Ker_project)
    
    #Get matrix K(X_Target,X_Context):
    Gram_target_context=Gram_matrix(X=X_Target,Y=X_Context,l_scale=l_scale,sigma_var=sigma_var, kernel_type=kernel_type,B=B,Ker_project=Ker_project)
    
    #Invert Gram-Context matrix:
    inv_Gram_context=Gram_context.inverse()
    
    #Get prediction means and reshape it:
    Means=torch.mv(Gram_target_context,torch.mv(inv_Gram_context,Y_Context.flatten()))
    Means=Means.view(n_target_points,D)

    #Get prediction covariance matrix:
    Cov_Mat=Gram_target-torch.mm(Gram_target_context,torch.mm(inv_Gram_context,Gram_target_context.t()))
    
    #Get the variances of the components and reshape it:
    Vars=torch.diag(Cov_Mat).view(n_target_points,D)
    
    return(Means,Cov_Mat,Vars)


# In[35]:


def Plot_inference_2d(X_Context,Y_Context,X_Target,Y_Target,Predict):
    #Function hyperparameters for plotting:
    size_scale=2
    
    #Create subplots:
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(size_scale*10,size_scale*5))
    plt.axis('equal')
    #Plot context set in blue:
    ax[0].scatter(X_Context[:,0],X_Context[:,1],color='blue')
    ax[0].quiver(X_Context[:,0],X_Context[:,1],Y_Context[:,0],Y_Context[:,1],color='blue',pivot='mid')
    #Plot ground truth in red:
    ax[0].quiver(X_Target[:,0],X_Target[:,1],Y_Target[:,0],Y_Target[:,1],color='red',pivot='mid')
    #Plot predicted means:
    ax[0].quiver(X_Target[:,0],X_Target[:,1],Predict[:,0],Predict[:,1],color='green',pivot='mid')
    
    
#This functions plots GP inference for 2d inputs with 2d outputs by randomly splitting data in context
#and target set and plot ground truth vs. predictions. In addition, a plot showing covariances is made:
def Plot_GP_inference_2d(X,Y,n_context_points=10,l_scale=1,sigma_var=1, kernel_type="rbf",obs_noise=1e-4,B=None,Ker_project=False):
    '''
    Input: X,Y: torch.tensor - shape (n,2)
           n_context_points: int - number of context points
           l_scale,sigma_var,kernel_type,obs_noise,B,Ker_project: see function GP_inference
    Output: None - only plots
    '''
    #Function hyperparameters for plotting:
    size_scale=2
    ellip_scale=0.7
    
    #Split data randomly in context and target:
    X_Context,Y_Context,X_Target,Y_Target=Rand_Target_Context_Splitter(X,Y,n_context_points)
    
    #Create subplots:
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(size_scale*10,size_scale*5))
    #Plot context set in blue:
    ax[0].scatter(X_Context[:,0],X_Context[:,1],color='blue')
    ax[0].quiver(X_Context[:,0],X_Context[:,1],Y_Context[:,0],Y_Context[:,1],color='blue',pivot='mid')
    #Plot ground truth in red:
    ax[0].quiver(X_Target[:,0],X_Target[:,1],Y_Target[:,0],Y_Target[:,1],color='red',pivot='mid')
    
    #Perform inference:
    Predict,Cov_Mat,Var=GP_inference(X_Context,Y_Context,X_Target,l_scale=l_scale,sigma_var=sigma_var, 
                             kernel_type=kernel_type,obs_noise=obs_noise,B=B,Ker_project=Ker_project)
    #Plot predicted means:
    ax[0].quiver(X_Target[:,0],X_Target[:,1],Predict[:,0],Predict[:,1],color='green',pivot='mid')
    
    #Get window limites for plot and set window for second plot:
    max_x=torch.max(X[0:,])
    min_x=torch.min(X[0:,])
    max_y=torch.max(X[1:,])
    min_y=torch.min(X[1:,])
    ax[1].set_xlim(min_x,max_x)
    ax[1].set_ylim(min_y,max_y)
    
    #Go over all target points and plot ellipse of continour lines of density of distributions:
    for j in range(X_Target.size(0)):
        #Get covarinace matrix:
        A=Cov_Mat[2*j:(2*j+2),2*j:(2*j+2)]
        #Get the eigenvector corresponding corresponding to the largest eigenvalue:
        u=torch.eig(A,eigenvectors=True)[1][:,0]

        #Get the angle of the ellipse in degrees:
        alpha=360*torch.atan(u[1]/u[0])/(2*math.pi)
        
        #Get the width and height of the ellipses (eigenvalues of A):
        D=torch.sqrt(torch.eig(A,eigenvectors=True)[0][:,0])
        
        #Plot the Ellipse:
        E=Ellipse(xy=X_Target[j,].numpy(),width=ellip_scale*D[0],height=ellip_scale*D[1],angle=alpha)
        ax[1].add_patch(E)
#%%
#A function to load the GP data:
def load_2d_GP_data(Id,folder='Test_data/2d_GPs/',batch_size=1):
    X=np.load(folder+"GP_data_X"+Id+".npy")
    Y=np.load(folder+"GP_data_Y"+Id+".npy")
    X=torch.tensor(X,dtype=torch.get_default_dtype())
    Y=torch.tensor(Y,dtype=torch.get_default_dtype())
    GP_data=utils.TensorDataset(X,Y)
    GP_data_loader=utils.DataLoader(GP_data,batch_size=batch_size,shuffle=True,drop_last=True)
    return(GP_data_loader)

 #%%   
#A function which performs kernel smoothing for 2d matrix-valued kernels:
#The normalizer for the kernel smoother is a matrix in this case.
#Here: We compute the inverse for the 2d matrix analytically since this is quicker than numerical methods:
def Kernel_Smoother_2d(X_Context,Y_Context,X_Target,normalize=True,l_scale=1,sigma_var=1,kernel_type="rbf",B=None,Ker_project=False):
    '''
    Inputs: X_Context,Y_Context - torch.tensor -shape (n_context_points,2)
            X_Target - torch.tensor - shape (n_target_points,2)
            l_scale,sigma_var,kernel_type,B,Ker_project: Kernel parameters - see kernel_mat
    Output:
            Kernel smooth estimates at X_Target 
            torch.tensor - shape - (n_target_points,2)
    '''
    print("Begin loop in Kernel Smoother 2d ", datetime.datetime.today())

    #Get the number of context and target points:
    n_context_points=X_Context.size(0)
    n_target_points=X_Target.size(0)
    #Get the Gram-matrix between the target and the context set --> shape (n_target_points*2,n_context_points*2):
    Gram=Gram_matrix(X=X_Target,Y=X_Context,l_scale=l_scale,sigma_var=sigma_var,kernel_type=kernel_type,B=B,Ker_project=Ker_project)
    print("After Gram ", datetime.datetime.today())

    #Get a kernel interpolation for the Target set and reshape it --> shape (n_target_points,2):
    Interpolate=torch.mv(Gram,Y_Context.flatten())
    print("After Interpolate ", datetime.datetime.today())

    #If wanted, normalize the output:
    if normalize: 
        #Define two vectors as tools to get the sum of the 2x2 block matrices in Gram:
        splitter_1=torch.tensor((1.,0.)).repeat(n_context_points)
        splitter_2=torch.tensor((0.,1.)).repeat(n_context_points)
        
        #Get sum of element (1,1) and (1,2):
        First_split=torch.mv(Gram,splitter_1).view(n_target_points,1,2)
        #Get sum of element (1,2) and (2,2):
        Second_split=torch.mv(Gram,splitter_2).view(n_target_points,1,2)
        #Merge columns to one matrix ->(n_target_points,2,2) - Merge[i,] gives sum of kernel matrices:
        Merge=torch.cat([First_split,Second_split],dim=1)
        
        #-----------
        #Out of Merge - compute the inverse by the formula: 
        # (a b)^{-1} = (1/(ad-cb))* (d,-b)
        # (c d)                     (-b,a)
        #Compute the determinant for the matrix -->shape (n_target_points) (add some noise to ensure stability):
        Determinants=Merge[:,0,0]*Merge[:,1,1]-Merge[:,0,1]*Merge[:,1,0]
        
        #Compute the unnormalized inverses by swapping elements:
        Swap=torch.empty(Merge.size())
        Swap[:,0,1]=-Merge[:,0,1]
        Swap[:,1,0]=-Merge[:,1,0]
        #Swap a and d:
        Swap[:,0,0]=Merge[:,1,1]
        Swap[:,1,1]=Merge[:,0,0]
        #Get the inverses:
        Inverses=Swap/Determinants.view(n_target_points,1,1)
        #Control whether we computed the correct inverse:
        #-----------------------------------------------
        
        #Perform batch-wise multiplication with inverses 
        #(need to reshape vectors to a one-column matrix first and after the multiplication back):
        Interpolate=torch.matmul(Inverses,Interpolate.view(n_target_points,2,1))
    print("After Normalize ", datetime.datetime.today())
    #Return the vector:
    return(Interpolate.view(n_target_points,2))


'''#%% Problem with the kernel smoother for div-free kernel:
X_Context=torch.tensor([[-1.,1.],[1.,1.],[-1.,-1.],[1.,-1.]])
Y_Context=torch.tensor([[1.,-1.],[-1.,-1.],[1.,1.],[-1.,1.]])
X_Target=My_Tools.Give_2d_Grid(min_x=-1,max_x=1,n_x_axis=10)
Smoother=Kernel_Smoother_2d(X_Context,Y_Context,X_Target,normalize=True,l_scale=1,sigma_var=1,kernel_type="rbf",B=None,Ker_project=False)
My_Tools.Plot_Inference_2d(X_Context,Y_Context,X_Target,None,Predict=Smoother,Cov_Mat=None)
plt.plot(X_Context[:,0].numpy(),Y_Context[:,0].numpy())
plt.plot(X_Target[:,0].numpy(),Smoother[:,0].numpy(),c="red")

#%%
#%% Problem with the kernel smoother for div-free kernel:
X_Context=torch.tensor([[-1.,1.],[1.,1.],[-1.,-1.],[1.,-1.]])
Y_Context=torch.tensor([[1.,-1.],[-1.,-1.],[1.,-1.],[-1.,1.]])
X_Target=My_Tools.Give_2d_Grid(min_x=-1,max_x=1,n_x_axis=10)
Smoother=Kernel_Smoother_2d(X_Context,Y_Context,X_Target,normalize=True,l_scale=1,sigma_var=1,kernel_type="rbf",B=None,Ker_project=False)
My_Tools.Plot_Inference_2d(X_Context,Y_Context,X_Target,None,Predict=Smoother,Cov_Mat=None)
plt.plot(X_Context[:,0].numpy(),Y_Context[:,0].numpy())
plt.plot(X_Target[:,0].numpy(),Smoother[:,0].numpy(),c="red")
''' 
#%%
#-----------------------------------------BATCH VERSIONS OF SOME OF THE ABOVE FUNCTION--------------------------
#%%
def Batch_Gram_matrix(X,Y=None,l_scale=1,sigma_var=1, kernel_type="rbf",B=None,Ker_project=False, flattened=False):
    '''
    Input:
    X: torch.tensor
          Shape: (batch_size,n,d)...batch_size...batch size- n number of obs- d...dimension of state space
    Y: torch.tensor or None
          Shape: (batch_size,m,d)...m number of obs, d...dimension of state space 
    l_scale,sigma_var,kernel_type,B,Ker_project: see function "mat_kernel"
    flattened: Boolean - gives shape of output of matrix

    Output:
    Gram_matrix: torch.tensor
                 if flattened shape (batch_size,n*D,m*D)
                 else shape (batch_size,n,m,D,D) (if Y is given (batch_size,n,m,D,D))
                 Block (k,i,j) of size dxd gives Kernel value of i-th X-data point and
                 j-th Y data point of the k-th element of the batch
    '''
    d=X.size(2)
    n=X.size(1)
    batch_size=X.size(0)
    if (B is None):
        B=torch.eye(d)
    if (Y is None):
        Y=X
    m=Y.size(1)
    D=B.size(0)
    if flattened:
        #Create empty matrix:
        Gram_matrix=torch.empty([batch_size,n*D,m*D])
        for k in range(batch_size):
            #Fill matrix block-wise:
            for i in range(n):
                for j in range(m):
                    Gram_matrix[k,D*i:(D*i+D),D*j:(D*j+D)]=mat_kernel(x1=X[k,i,],x2=Y[k,j,],l_scale=l_scale,sigma_var=sigma_var,kernel_type=kernel_type,B=B,Ker_project=Ker_project)
    else:
        #Create empty matrix:
        Gram_matrix=torch.empty([batch_size,n,m,D,D])
        #Fill matrix block-wise:
        for k in range(batch_size):
            for i in range(n):
                for j in range(m):
                    Gram_matrix[k,i,j]=mat_kernel(x1=X[k,i,],x2=Y[k,j,],l_scale=l_scale,sigma_var=sigma_var,kernel_type=kernel_type,B=B,Ker_project=Ker_project)
    return(Gram_matrix)

 #%%   
#A function which performs kernel smoothing for 2d matrix-valued kernels:
#The normalizer for the kernel smoother is a matrix in this case.
#Here: We compute the inverse for the 2d matrix analytically since this is quicker than numerical methods:
def Batch_Kernel_Smoother_2d(X_Context,Y_Context,X_Target,normalize=True,l_scale=1,sigma_var=1,kernel_type="rbf",B=None,Ker_project=False):
    '''
    Inputs: X_Context,Y_Context - torch.tensor -shape (batch_size,n_context_points,2)
            X_Target - torch.tensor - shape (batch_size,n_target_points,2)
            l_scale,sigma_var,kernel_type,B,Ker_project: Kernel parameters - see kernel_mat
    Output:
            Kernel smooth estimates at X_Target 
            torch.tensor - shape - (batch_size,n_target_points,2)
    '''
    #Get the number of context and target points:
    batch_size=X_Context.size(0)
    n_context_points=X_Context.size(1)
    n_target_points=X_Target.size(1)
    #Get the Gram-matrix between the target and the context set 
    #--> shape (batch_size,2*n_target_points,2*n_context_points):
    Gram=Batch_Gram_matrix(X=X_Target,Y=X_Context,l_scale=l_scale,
                     sigma_var=sigma_var,kernel_type=kernel_type,B=B,Ker_project=Ker_project,flattened=True)
    #Get a kernel interpolation for the Target set --> shape (batch_size,n_target_points*2)
    Interpolate=torch.matmul(Gram,Y_Context.view(batch_size,n_context_points*2,1))

    #If wanted, normalize the output:
    if normalize: 
        #Define two vectors as tools to get the sum of the 2x2 block matrices in Gram:
        splitter_1=torch.tensor((1.,0.)).repeat(n_context_points)
        splitter_2=torch.tensor((0.,1.)).repeat(n_context_points)
        
        #Get sum of element (1,1) and (2,1):
        First_split=torch.matmul(Gram,splitter_1).view(batch_size,n_target_points,1,2)
        #Get sum of element (1,2) and (2,2):
        Second_split=torch.matmul(Gram,splitter_2).view(batch_size,n_target_points,1,2)
        
        #Merge columns to one matrix ->(n_target_points,2,2) - Merge[i,] gives sum of kernel matrices:
        Merge=torch.cat([First_split,Second_split],dim=2)
        
        #-----------
        #Out of Merge - compute the inverse by the formula: 
        # (a b)^{-1} = (1/(ad-cb))* (d,-b)
        # (c d)                     (-b,a)
        #Compute the determinant for the matrix -->shape (n_target_points) (add some noise to ensure stability):
        Determinants=Merge[:,:,0,0]*Merge[:,:,1,1]-Merge[:,:,0,1]*Merge[:,:,1,0]
        
        #Compute the unnormalized inverses by swapping elements:
        Swap=torch.empty(Merge.size())
        Swap[:,:,0,1]=-Merge[:,:,0,1]
        Swap[:,:,1,0]=-Merge[:,:,1,0]
        #Swap a and d:
        Swap[:,:,0,0]=Merge[:,:,1,1]
        Swap[:,:,1,1]=Merge[:,:,0,0]
        
        #Get the inverses:
        Inverses=Swap/Determinants.view(batch_size,n_target_points,1,1)
        #Control whether we computed the correct inverse:
        print(torch.matmul(Merge,Inverses))
        #-----------------------------------------------
        
        #Perform batch-wise multiplication with inverses 
        #(need to reshape vectors to a one-column matrix first and after the multiplication back):
        Interpolate=torch.matmul(Inverses,Interpolate.view(batch_size,n_target_points,2,1))
    #Return the vector:
    return(Interpolate.view(batch_size,n_target_points,2))
             
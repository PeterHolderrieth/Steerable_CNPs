#!/usr/bin/env python
# coding: utf-8

# To do:
# 1. Define a Ker_Project in Gram matrix - i.e. a projection on the tangent space of the sphere
# 2. Define GP inference function for Sphere/3d
# 3. Export various data sets in 3d
# In[22]:
#Add current directory to the python path:
import os,sys
sys.path.append(os.getcwd())

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
import math
from numpy import savetxt
import csv
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Own files:
import My_Tools





#HYPERPARAMETERS:
#Set default as double:
torch.set_default_dtype(torch.float)

'''
____________________________________________________________________________________________________________________

----------------------------KERNEL TOOLS -------------------------------------------------------------------------
____________________________________________________________________________________________________________________
'''
#%% This function gives the Gram/Kernel -matrix K(X,Y) of two data sets X and Y"
def Gram_matrix(X,Y=None,l_scale=1,sigma_var=1, kernel_type="rbf",B=None,Ker_project=False,flatten=True):
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
    if B is None:
        B=torch.eye(d).to(X.device)
    #If Y is not given, set to X:
    if Y is None:
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
        '''
        The following computations are based on equation (24) in
        "Kernels for Vector-Valued Functions: a Review" by Alvarez et al
        '''
        #Create a distance matrix:
        X=X.unsqueeze(1).expand(n,m,d)
        Y=Y.unsqueeze(0).expand(n,m,d)
        #Create distance matrix from that --> shape (n,m)
        Dist_mat=torch.sum((X-Y)**2,dim=2)
        #Create the RBF matrix from that --> shape (n,m)
        Gram_RBF=torch.exp(-0.5*Dist_mat/l_scale)/l_scale
        #Reshape for later use:
        Gram_RBF=Gram_RBF.view(n,m,1,1)
        #Get the differences:
        Diff=X-Y
        #Get matrix of outer product --> shape (n,m,d,d)
        Outer_Prod_Mat=torch.matmul(Diff.unsqueeze(3),Diff.unsqueeze(2))
        #Get n*m copies of identity matrices in Rd--> shape (n,m,d,d)
        Ids=torch.eye(d).to(X.device)
        Ids=Ids.view(1,1,d,d)
        Ids=Ids.expand(n,m,d,d)
        #First matrix component for divergence-free kernel-->shape (n,m,d,d)
        Mat_1=Outer_Prod_Mat/l_scale
        #Second matrix component for divergence-free kernel --> shape (n,m,d,d)
        Mat_2=(d-1-Dist_mat.view(n,m,1,1)/l_scale)*Ids
        #Matrix sum of the two matrices:
        A=Mat_1+Mat_2
        #Multiply scalar and matrix part:
        K=Gram_RBF*A
       
    else:
        sys.exit("Unknown kernel type")
    if flatten:
        return(My_Tools.Create_matrix_from_Blocks(K))
    else:
        return(K)


#%% This function gives the Gram/Kernel -matrix K(X,Y) of two data sets X and Y"
def Batch_Gram_matrix(X,Y=None,l_scale=1,sigma_var=1, kernel_type="rbf",B=None,Ker_project=False,flatten=True):
    '''
    Input:
    X: torch.tensor
          Shape: (batch_size,n,d)...n number of obs, d...dimension of state space
    Y: torch.tensor or None
          Shape: (batch_size,m,d)...m number of obs, d...dimension of state space 
    l_scale,sigma_var,kernel_type,B,Ker_project: see function "mat_kernel"

    Output:
    Gram_matrix: torch.tensor
                 Shape (batch_size,n,m,D) (if Y is not given (batch_size,n,n,D))
                 Block i,j of size DxD gives Kernel value of i-th X-data point and
                 j-th Y data point
    '''
    #Get dimension of data space and number of observations from X:
    d=X.size(2)
    n=X.size(1)
    batch_size=X.size(0)
    #If B is not given, set to identity:
    if B is None:
        B=torch.eye(d).to(X.device)
    #If Y is not given, set to X:
    if Y is None:
        Y=X
    #Get number of observations from Y and dimension of B:
    m=Y.size(1)
    D=B.size(0)
    #RBF kernel:
    if kernel_type=="rbf":
        #Expand X,Y along different dimension to get a grid shape:
        X=X.unsqueeze(2).expand(batch_size,n,m,d)
        Y=Y.unsqueeze(1).expand(batch_size,n,m,d)
        #Compute the squared distance matrix:
        Dist_mat=torch.sum((X-Y)**2,dim=3)
        #Compute the RBF kernel from that:
        Gram_RBF=sigma_var*torch.exp(-0.5*Dist_mat/l_scale)
        #Expand B such that it has the right shape:
        B=B.view(1,1,1,D,D)
        B=B.expand(batch_size,n,m,D,D)
        #Reshape RBF Gram matrix:
        Gram_RBF=Gram_RBF.view(batch_size,n,m,1,1)
        #Multiply scalar Gram matrix with the matrix B:
        K=Gram_RBF*B

    elif kernel_type=="dot_product":
        #Get dot product Gram matrix --> shape (batch_size,n,m)
        Gram_one_d=torch.matmul(X,Y.transpose(-1,1))
        #Expand B:
        B=B.view(1,1,1,D,D)
        B=B.expand(batch_size,n,m,D,D)
        #Expand one-dimensional Gram:
        Gram_one_d=Gram_one_d.view(batch_size,n,m,1,1)
        #Multiply with B:
        K=Gram_one_d*B

    elif kernel_type=="div_free":
        '''
        The following computations are based on equation (24) in
        "Kernels for Vector-Valued Functions: a Review" by Alvarez et al
        '''
        #Create a distance matrix:
        X=X.unsqueeze(2).expand(batch_size,n,m,d)
        Y=Y.unsqueeze(1).expand(batch_size,n,m,d)
        #Create distance matrix from that --> shape (batch_size,n,m)
        Dist_mat=torch.sum((X-Y)**2,dim=3)
        #Create the RBF matrix from that --> shape (batch_size,n,m)
        Gram_RBF=torch.exp(-0.5*Dist_mat/l_scale)/l_scale
        #Reshape for later use:
        Gram_RBF=Gram_RBF.view(batch_size,n,m,1,1)
        #Get the differences -->shape (batch_size,n,m,d):
        Diff=X-Y
        #Get matrix of outer product --> shape (batch_size,n,m,d,d)
        Outer_Prod_Mat=torch.matmul(Diff.unsqueeze(4),Diff.unsqueeze(3))
        #Get n*m copies of identity matrices in Rd--> shape (batch_size,n,m,d,d)
        Ids=torch.eye(d).to(X.device)
        Ids=Ids.view(1,1,1,d,d)
        Ids=Ids.expand(batch_size,n,m,d,d)
        #First matrix component for divergence-free kernel-->shape (batch_size,n,m,d,d)
        Mat_1=Outer_Prod_Mat/l_scale
        #Second matrix component for divergence-free kernel --> shape (batch_size,n,m,d,d)
        Mat_2=(d-1-Dist_mat.view(batch_size,n,m,1,1)/l_scale)*Ids
        #Matrix sum of the two matrices:
        A=Mat_1+Mat_2
        #Multiply scalar and matrix part:
        K=Gram_RBF*A
       
    else:
        sys.exit("Unknown kernel type")
    if flatten:
        return(My_Tools.Batch_Create_matrix_from_Blocks(K))
    else:
        return(K)

#A function which performs kernel smoothing for 2d matrix-valued kernels:
#The normalizer for the kernel smoother is a matrix in this case (assuming that it is invertible)
def Kernel_Smoother_2d(X_Context,Y_Context,X_Target,normalize=True,l_scale=1,sigma_var=1,kernel_type="rbf",B=None,Ker_project=False):
    '''
    Inputs: X_Context - torch.tensor -shape (n_context_points,2)
            Y_Context - torch.tensor - shape (n_context_points,D)
            X_Target - torch.tensor - shape (n_target_points,2)
            l_scale,sigma_var,kernel_type,B,Ker_project: Kernel parameters - see Gram_matrix
    Output:
            Kernel smooth estimates at X_Target 
            torch.tensor - shape - (n_target_points,D)
    '''
    #Get the number of context and target points and the dimension of the output space:
    n_context_points=X_Context.size(0)
    n_target_points=X_Target.size(0)
    D=Y_Context.size(1)
    if B is None:
        B=torch.eye(D,device=X_Target.device)
    
    point=datetime.datetime.today()
    #Get the Gram-matrix between the target and the context set --> shape (n_target_points,n_context_points,2,2):
    Gram_Blocks=Gram_matrix(X=X_Target,Y=X_Context,l_scale=l_scale,sigma_var=sigma_var,kernel_type=kernel_type,B=B,Ker_project=Ker_project,flatten=False)
    #print("After Gram matrix: ",datetime.datetime.today()-point)
    point=datetime.datetime.today()
    Gram_Mat=My_Tools.Create_matrix_from_Blocks(Gram_Blocks)
    #Get a kernel interpolation for the Target set and reshape it --> shape (2*n_target_points):
    Interpolate=torch.mv(Gram_Mat,Y_Context.flatten())
    #If wanted, normalize the output:
    if normalize: 
        #Get the column sum of the matrices
        Col_Sum_Mats=Gram_Blocks.sum(dim=1)
        
        #Get the inverses:
        Inverses=Col_Sum_Mats.inverse()
        
        #Perform batch-wise multiplication with inverses 
        #(need to reshape vectors to a one-column matrix first and after the multiplication back):
        Interpolate=torch.matmul(Inverses,Interpolate.view(n_target_points,D,1))
    #print("After normalization: ",datetime.datetime.today()-point)
    point=datetime.datetime.today()
    #Return the vector:
    return(Interpolate.view(n_target_points,D))  

#A batch version of the above function:
def Batch_Kernel_Smoother_2d(X_Context,Y_Context,X_Target,normalize=True,l_scale=1,sigma_var=1,kernel_type="rbf",B=None,Ker_project=False):
    '''
    Inputs: X_Context - torch.tensor -shape (batch_size,n_context_points,2)
            Y_Context - torch.tensor - shape (batch_size,n_context_points,D)
            X_Target - torch.tensor - shape (batch_size,n_target_points,2)
            l_scale,sigma_var,kernel_type,B,Ker_project: Kernel parameters - see Gram_matrix
    Output:
            Kernel smooth estimates at X_Target 
            torch.tensor - shape - (batch_size,n_target_points,D)
    '''
    #Get the number of context and target points and the dimension of the output space:
    n_context_points=X_Context.size(1)
    n_target_points=X_Target.size(1)
    batch_size=X_Context.size(0)
    D=Y_Context.size(2)
    if B is None:
        B=torch.eye(D,device=X_Target.device)

    #Get the Gram-matrix between the target and the context set --> shape (batch_size,n_target_points,n_context_points,D,D):
    Gram_Blocks=Batch_Gram_matrix(X=X_Target,Y=X_Context,l_scale=l_scale,sigma_var=sigma_var,kernel_type=kernel_type,B=B,Ker_project=Ker_project,flatten=False)
    #Reshape --> (batch_size,n_target_points*D,n_context_points*D):
    Gram_Mat=My_Tools.Batch_Create_matrix_from_Blocks(Gram_Blocks)
    #Get a kernel interpolation for the Target set and reshape it --> shape (batch_size,n_target_points*D):
    Interpolate=torch.matmul(Gram_Mat,Y_Context.reshape(batch_size,-1,1)).squeeze()
    #If wanted, normalize the output:
    if normalize: 
        #Get the column sum of the matrices
        Col_Sum_Mats=Gram_Blocks.sum(dim=2)
        
        #Get the inverses:
        Inverses=Col_Sum_Mats.inverse()
        
        #Perform batch-wise multiplication with inverses 
        #(need to reshape vectors to a one-column matrix first and after the multiplication back):
        Interpolate=torch.matmul(Inverses,Interpolate.view(batch_size,n_target_points,D,1))

    #Return the vector:
    return(Interpolate.view(batch_size,n_target_points,D))


# Problem with the kernel smoother for div-free kernel if it is normalizing:
'''
X_Context=torch.tensor([[-1.,1.],[1.,1.],[-1.,-1.],[1.,-1.]])
Y_Context=torch.tensor([[1.,-1.],[-1.,-1.],[1.,1.],[-1.,1.]])
X_Target=My_Tools.Give_2d_Grid(min_x=-1,max_x=1,n_x_axis=10)
Smoother=Kernel_Smoother_2d(X_Context,Y_Context,X_Target,normalize=True,l_scale=1,sigma_var=1,kernel_type="rbf",B=None,Ker_project=False)
My_Tools.Plot_Inference_2d(X_Context,Y_Context,X_Target,None,Predict=Smoother,Cov_Mat=None)
plt.plot(X_Context[:,0].numpy(),Y_Context[:,0].numpy())
plt.plot(X_Target[:,0].numpy(),Smoother[:,0].numpy(),c="red")

X_Context=torch.tensor([[-1.,1.],[1.,1.],[-1.,-1.],[1.,-1.]])
Y_Context=torch.tensor([[1.,-1.],[-1.,-1.],[1.,-1.],[-1.,1.]])
X_Target=My_Tools.Give_2d_Grid(min_x=-1,max_x=1,n_x_axis=10)
Smoother=Kernel_Smoother_2d(X_Context,Y_Context,X_Target,normalize=True,l_scale=1,sigma_var=1,kernel_type="div_free",B=None,Ker_project=False)
My_Tools.Plot_Inference_2d(X_Context,Y_Context,X_Target,None,Predict=Smoother,Cov_Mat=None)
plt.plot(X_Context[:,0].numpy(),Y_Context[:,0].numpy())
plt.plot(X_Target[:,0].numpy(),Smoother[:,0].numpy(),c="red")

'''
#%%
'''
____________________________________________________________________________________________________________________

----------------------------Multi-dimensional Gaussian Processes ---------------------------------------------------------------------
____________________________________________________________________________________________________________________
'''
#%%
#This function samples a multi-dimensional GP with kernel of a type give by the function Gram_matrix
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
#This functions perform GP-inference on the function values at X_Target (so no noise for the target value)
#based on context points X_Context and labels Y_Context:
def GP_inference(X_Context,Y_Context,X_Target,l_scale=1,sigma_var=1, kernel_type="rbf",obs_noise=0.1,B=None,Ker_project=False):
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

#%%
'''
____________________________________________________________________________________________________________________

----------------------------2d Gaussian Processes ---------------------------------------------------------------------
____________________________________________________________________________________________________________________
'''

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
    X=My_Tools.Give_2d_Grid(min_x,max_x,n_x_axis=n_grid_points,flatten=True)
    
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
    
# In[31]:
#%%
'''
____________________________________________________________________________________________________________________

----------------------------Spherical Gaussian Processes ---------------------------------------------------------------------
____________________________________________________________________________________________________________________
'''

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
        Proj_Y=My_Tools.Tangent_Sphere_Projector(X,Y)
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
#%%

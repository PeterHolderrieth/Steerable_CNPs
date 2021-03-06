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
import datetime 
import warnings
from prettytable import PrettyTable

warnings.filterwarnings("ignore", category=UserWarning)

#E(2)-steerable CNNs - library:
from e2cnn import gspaces    
import e2cnn
from e2cnn import nn as G_CNN   

#HYPERPARAMETERS:
#Set default as float:
torch.set_default_dtype(torch.float)


'''
____________________________________________________________________________________________________________________

----------------------------General Tools---------------------------------------------------------------------
____________________________________________________________________________________________________________________
'''
#Tool to split a function in a context and target set (choice is random but size of context set is given):
def rand_target_context_splitter(X,Y,n_context_points):
    '''
    Inputs: X: torch.tensor - shape (batch_size,n,d) - n...number of observations, d...dimension of state space
            Y: torch.tensor - shape (batch_size,n,D) - N...number of observations, D...dimension of label space
            n_context_points: int - size of context set
    Outputs:
        X_Context: torch.tensor - shape (batch_size,n_context_points,d)
        Y_Context: torch.tensor - shape (batch_size,n_context_points,D)
        X_Target:  torch.tensor - shape (batch_size,n-n_context_points,d)
        Y_Target:  torch.tensor - shape (batch_size,n-n_context_points,D)
    '''
    n=X.size(1)
    ind_shuffle=torch.randperm(n)
    X_Context=X[:,ind_shuffle[:n_context_points]]
    Y_Context=Y[:,ind_shuffle[:n_context_points]]
    X_Target=X[:,ind_shuffle[n_context_points:,]]
    Y_Target=Y[:,ind_shuffle[n_context_points:,]]
    return(X_Context,Y_Context,X_Target,Y_Target)


#This function returns the indices of an (n,n)-grid 
#which are in the inner circle of that square grid.
#In other words, those points in square which stay within the square
#after a rotation.
def get_outer_circle_indices(n):
    '''
    Input: n - int - size of square
    Ouput: Ind - torch.tensor
    '''
    x_axis=torch.linspace(start=0,end=n-1,steps=n)
    y_axis=torch.linspace(start=0,end=n-1,steps=n)
    X1,X2=torch.meshgrid(x_axis,y_axis)
    Ind=torch.stack((X1,X2),2)
    Ind=Ind[torch.norm(Ind-(n-1)/2,dim=2)>(n-1)/2].long()
    return(Ind)
def get_inner_circle_indices(n,flat=False):
    '''
    Input: n - int - size of square
    Ouput: Ind - torch.tensor
    '''
    x_axis=torch.linspace(start=0,end=n-1,steps=n)
    y_axis=torch.linspace(start=0,end=n-1,steps=n)
    X1,X2=torch.meshgrid(x_axis,y_axis)
    Ind=torch.stack((X1,X2),2)
    Ind=Ind[torch.norm(Ind-(n-1)/2,dim=2)<(n-1)/2].long()
    if flat:
        Ind=n*Ind[:,0]+Ind[:,1]
    return(Ind)

def bool_inner_circle_indices(n):
    '''
    Input: n - int - size of square
    Ouput: Ind - torch.tensor
    '''
    
    x_axis=torch.linspace(start=0,end=n-1,steps=n)
    y_axis=torch.linspace(start=0,end=n-1,steps=n)
    X1,X2=torch.meshgrid(x_axis,y_axis)
    Ind=torch.stack((X1,X2),2)
    return(torch.norm(Ind-(n-1)/2,dim=2)<=(n-1)/2)
    
def set_outer_circle_zero(X):
    '''
    Input: X - torch.tensor - shape (*,n,n)
    Output: torch.tensor - shape (*,*,n,n) - elements outside of the inner circle in an (n,n)-square set to zero
    '''
    n=X.size(-1)
    dim_batch=len(X.shape)-2
    return(X*bool_inner_circle_indices(n)[(None,)*dim_batch])
#This tool defines a regularization term for learning functions - it regularizes the loss 
#such that prediction functions F with the similiar shape to the ground truth f are preferred against
#functions with the same "distance" to f (same as F) but a different shape.
#Mathematically: we take the difference F-f between the two functions, center it to mean zero
#i.e. take D=F-f-mean(F-f) and then take the the norm of D.
def shape_regularizer(Y_1,Y_2):
    '''
    Input: Y_1,Y_2 - torch.tensor - shape (batch_size,T,*) - T...number of observations, *...shape of space
    Output: float (torch.tensor of dim 0) - Centers Y_1-Y_2 to Cent_Diff and returns the Frobenius norm of Cent_Diff
    '''
    #Compute the difference--> shape (batch_size,T,*)
    Diff=Y_1-Y_2
    #Get means of difference--> shape (batch_size,*)
    Means=Diff.mean(dim=1)
    #Substract it from Diff:
    Cent_Diff=Diff-Means.unsqueeze(1)
    return(torch.sum(Cent_Diff**2,dim=(1,2)))


'''
____________________________________________________________________________________________________________________

----------------------------2d Tools ---------------------------------------------------------------------
____________________________________________________________________________________________________________________
'''   
#A tool to make a 2d_grid:
def give_2d_grid(min_x,max_x,n_x_axis,min_y=None,max_y=None,n_y_axis=None,flatten=True):
    '''
    Input:
        min_x,max_x,min_y,max_y: float - range of x-axis/y-axis
        n_x_axis,n_y_axis: int - number of points per axis
        flatten: Boolean - determines shape of output
    Output:
        torch.tensor - if flatten is True: shape (n_y_axis*n_x_axis,2) 
                                          (element i*n_x_axis+j gives i-th element in y-grid 
                                           and j-th element in  x-grid.
                                           In other words: x is periodic counter and y the stable counter)
                       if flatten is not True: shape (n_y_axis,n_x_axis,2)
    '''
    if min_y is None:
        min_y=min_x
    if max_y is None:
        max_y=max_x
    if n_y_axis is None:
        n_y_axis=n_x_axis
        
    x_grid_vec=torch.linspace(min_x,max_x,n_x_axis)
    y_grid_vec=torch.linspace(min_y,max_y,n_y_axis)
    Y,X=torch.meshgrid(y_grid_vec,x_grid_vec)
    Z=torch.stack((X,Y),2)
    if flatten:
        Z=Z.view(n_y_axis*n_x_axis,2)
    return(Z)

def radial_grid(min,max,n_axis):
    X=give_2d_grid(min,max,n_axis,flatten=False)
    Ind=bool_inner_circle_indices(n_axis)
    return(X[Ind])

#            
#Tool to plot context set, ground truth for target and predictions for target in one plot:
def plot_inference_2d(X_Context,Y_Context,X_Target=None,Y_Target=None,Predict=None,Cov_Mat=None,title="",size_scale=2, ellip_scale=0.8,quiver_scale=15,plot_points=False):
    '''
    Inputs: X_Context,Y_Context: torch.tensor - shape (n_context_points,2) - given context set
            X_Target,Y_Target: torch.tensor - shape (n_target_points,2) - target locations and outputs vectors 
            Predict: torch.tensor - shape (n_target_points,2) - target predictions (i.e. predicting Y_Target)
            Cov_Mat: torch.tensor - shape (n_target_points,2,2) - set of covariance matrices
                                  or - shape (n_target_points,2) - set of variances 
            title: string -  suptitle
    Outputs: None - plots the above tensors in two plots: one plots the means against the context sets, the other one the covariances/variances
    '''
    #Create plots:
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(size_scale*10,size_scale*5))
    plt.gca().set_aspect('equal', adjustable='box')
    fig.subplots_adjust(wspace=0.4)
    #Set titles:
    fig.suptitle(title)
    ax[0].set_title("Context set and predictions")
    ax[1].set_title("Variances")
    
    #Plot context set in blue:
    if plot_points:
        ax[0].scatter(X_Context[:,0],X_Context[:,1],color='black')
    if X_Context is not None and Y_Context is not None:
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


'''
____________________________________________________________________________________________________________________

----------------------------Matrix Tools ---------------------------------------------------------------------
____________________________________________________________________________________________________________________
'''
#A function to get the block diagonal matrix from a matrix:
def get_block_diagonal(X,size=1):
    '''
    Input: X - torch.tensor - shape (n,n)
           size - int - size divides n
    Output: torch.tensor - shape (n/size,size*size) - block diagonals of X
    '''
    m=X.size(0)//size
    Sigma=torch.empty((m,size,size)).to(X.device)
    for i in range(m):
        Sigma[i]=X[(i*size):(i+1)*size,(i*size):(i+1)*size]
    return Sigma

#A function to create a matrix from block matrices (merges the block matrices):
def create_matrix_from_blocks(X):
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

#A function to create a matrix from block matrices (merges the block matrices) (batchwise version of the above function):
def batch_create_matrix_from_blocks(X):
    '''
    Input:
        X - torch.tensor - shape (batch_size,n,m,D_1,D_2)
    Output:
        torch-tensor - shape (batch_size,n*D_1,m*D_2) - block (i,j) of size D_1*D_2 is matrix X[i,j] for i=1,...,n,j=1,...,m
    '''
    batch_size=X.size(0)
    n=X.size(1)
    m=X.size(2)
    D_1=X.size(3)
    D_2=X.size(4)
    M=torch.cat([X[:,:,:,i,:].reshape(batch_size,n,m*D_2) for i in range(D_1)],dim=1)
    ind=torch.cat([torch.arange(i,D_1*n,n,dtype=torch.long) for i in range(n)])
    return(M[:,ind])

#The following function compute the eigenvalue decomposition of a batch 
# of symmetric 2d matrices - represented as vector in R3:
# (torch.symeig is extremely slow on a GPU.)    
def sym_eig_2d(X):
    '''
    Input: X - torch.tensor - shape (n,3) - n batch size
    '''
    #Epsilon to add such that square root differentation is numerically stable:
    eps=1e-5
    #Trace and determinant of the matrix:
    T=X[:,0]+X[:,2]
    D=X[:,0]*X[:,2]-X[:,1]**2
    #Computer lower and upper eigenvalues and stack them:
    L_1=T/2-torch.sqrt(T**2/4-D+eps)
    L_2=T/2+torch.sqrt(T**2/4-D+eps)
    eigen_val=torch.cat([L_1.view(-1,1),L_2.view(-1,1)],dim=1)
    #Get one eigenvector, normalize it and get the second one by rotation of 90 degrees:
    eigen_vec_1=torch.cat([torch.ones((X.size(0),1)).to(X.device),((L_1-X[:,0])/X[:,1]).view(-1,1)],dim=1)
    eigen_vec_1=eigen_vec_1/torch.norm(eigen_vec_1,dim=1).unsqueeze(1)
    eigen_vec_2=torch.cat([-eigen_vec_1[:,1].unsqueeze(1),eigen_vec_1[:,0].unsqueeze(1)],dim=1)
    #Stack the eigenvectors:
    eigen_vec=torch.cat([eigen_vec_1.unsqueeze(2),eigen_vec_2.unsqueeze(2)],dim=2)
    return(eigen_val,eigen_vec)

def batch_sym_eig_2d(X):
    '''
    Input: X - torch.tensor - shape (batch_size,n,3) - n batch size
    '''
    #Epsilon to add such that square root differentation is numerically stable:
    eps=1e-5
    batch_size=X.size(0)
    n=X.size(1)

    #Trace and determinant of the matrix --> shape (batch_size,n)
    T=X[:,:,0]+X[:,:,2]
    D=X[:,:,0]*X[:,:,2]-X[:,:,1]**2
    #Computer lower and upper eigenvalues and stack --> shape (batch_size,n,2)
    L_1=T/2-torch.sqrt(T**2/4-D+eps)
    L_2=T/2+torch.sqrt(T**2/4-D+eps)
    eigen_val=torch.cat([L_1.view(batch_size,n,1),L_2.view(batch_size,n,1)],dim=2)
    #Get one eigenvector, normalize it and get the second one by rotation of 90 degrees:
    eigen_vec_1=torch.cat([torch.ones((batch_size,n,1)).to(X.device),((L_1-X[:,:,0])/X[:,:,1]).view(batch_size,-1,1)],dim=2)
    eigen_vec_1=eigen_vec_1/torch.norm(eigen_vec_1,dim=2).unsqueeze(2)
    eigen_vec_2=torch.cat([-eigen_vec_1[:,:,1].unsqueeze(2),eigen_vec_1[:,:,0].unsqueeze(2)],dim=2)
    #Stack the eigenvectors:
    eigen_vec=torch.cat([eigen_vec_1.unsqueeze(3),eigen_vec_2.unsqueeze(3)],dim=3)
    return(eigen_val,eigen_vec)

#Activation function to get a covariance matrix - apply softplus or other activation functions on eigenvalues:    
def plain_cov_activation_function(X,activ_type="softplus"):
    '''
    Input:
    X- torch.tensor - shape (n,3)
    activ_type - string -type of activation applied on the diagonal matrix
    Output: torch.tensor - shape (n,2,2) - consider X[i] as a symmetric 2x2 matrix
            we compute the eigendecomposition X[i]=UDU^T of that matrix and 
            if s is a an activation function of type activ_type, we apply it componentwise on the diagonal s(D)
            and return Us(D)U^T (in one batch)
    '''
    n=X.size(0)
    eigen_vals,eigen_vecs=sym_eig_2d(X)
    if activ_type=="softplus":
        eigen_vals=1e-5+F.softplus(eigen_vals)
    else: 
        sys.exit("Unknown activation type")
    return(torch.matmul(eigen_vecs, torch.matmul(eigen_vals.diag_embed(), eigen_vecs.transpose(-2, -1))))

#Activation function to get a covariance matrix - apply softplus or other activation functions on eigenvalues:    
def batch_plain_cov_activation_function(X,activ_type="softplus"):
    '''
    Input:
    X- torch.tensor - shape (batch_size,n,3)
    activ_type - string -type of activation applied on the diagonal matrix
    Output: torch.tensor - shape (batch_size,n,2,2) - consider X[j,i] as a symmetric 2x2 matrix
            we compute the eigendecomposition X[j,i]=UDU^T of that matrix and 
            if s is a an activation function of type activ_type, we apply it componentwise on the diagonal s(D)
            and return Us(D)U^T (in one batch)
    '''
    n=X.size(1)
    eigen_vals,eigen_vecs=batch_sym_eig_2d(X)
    if activ_type=="softplus":
        eigen_vals=1e-5+F.softplus(eigen_vals)
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
            This makes the function fully differentiable (however, the gradient is slightly changed for 
            inputs which are close to a diagonal, namely the gradient w.r.t. to x2 cut to zero.)
    '''
    n=X.size(0)
    Out=torch.zeros([n,2,2],device=X.device) 
    below_tol=(torch.abs(X[:,1])<tol)#&(torch.abs(X[:,0]-X[:,2])<tol)
    above_tol=~below_tol
    if any(above_tol):
        Out[above_tol]=plain_cov_activation_function(X[above_tol],activ_type=activ_type)
    
    if activ_type=="softplus":
        Out[below_tol,0,0]=F.softplus(X[below_tol][:,0])
        Out[below_tol,1,1]=F.softplus(X[below_tol][:,2])
    else: 
        sys.exit("Unknown activation type")
    return(Out)

def batch_stable_cov_activation_function(X,activ_type="softplus",tol=1e-7):
    '''
    Input:
            X- torch.tensor - shape (batch_size,n,3)
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
            This makes the function fully differentiable (however, the gradient is slightly changed for 
            inputs which are close to a diagonal, namely the gradient w.r.t. to x2 cut to zero.)
    '''
    n=X.size(1)
    batch_size=X.size(0)
    Out=torch.zeros([batch_size,n,2,2],device=X.device) 
    below_tol=(torch.abs(X[:,:,1])<tol)#&(torch.abs(X[:,0]-X[:,2])<tol)
    above_tol=~below_tol
    if torch.sum(above_tol)>0:
        Out[above_tol]=plain_cov_activation_function(X[above_tol],activ_type=activ_type)
    if activ_type=="softplus":

        Out[below_tol]=F.softplus(torch.stack([X[below_tol][:,0],X[below_tol][:,2]],dim=1)).diag_embed()
        #Out[below_tol,1,1]=F.softplus(X[below_tol][:,2])
    else: 
        sys.exit("Unknown activation type")
    return(Out)

def batch_multivar_log_ll(Means,Covs,data):
    '''
    Input:
        Means - torch.tensor - shape (batch_size,n,D) - Means 
        Covs - torch.tensor - shape (batch_size,n,D,D) - Covariances
        data - torch.tensor - shape (batch_size,n,D) - observed data 
    Output:
        torch.tensor - shape (batch_size,n) - log-likelihoods of observations
    '''
    batch_size,n,D=Means.size()
    Diff=data-Means
    Quad_Term=torch.matmul(Diff.unsqueeze(2),torch.matmul(Covs.inverse(),Diff.unsqueeze(3))).squeeze()
    log_normalizer=-0.5*torch.log(((2*math.pi)**D)*Covs.det())
    log_ll=log_normalizer-0.5*Quad_Term
    return(log_ll)




'''
-------------------------------------------Tools for training -------------------------------------------------
'''
#A class which tracks averages and values over time:
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
'''
-------------------------------------------Tools for training -------------------------------------------------
'''
def count_parameters(model,print_table=False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    if print_table:    
        print(table)
        print(f"Total Trainable Params: {total_params}")
    return total_params

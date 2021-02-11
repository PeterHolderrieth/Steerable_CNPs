#LIBRARIES:
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

#E(2)-steerable CNNs - library:
from e2cnn import gspaces    
from e2cnn import nn as G_CNN   
from e2cnn import group  
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
#noise to add to diagonal to make p.s.d. matrix computation numerically stable:
NOISE=1e-3

#A function to compute the eigenvalue decomposition of a symmetric 2x2 matrix analytically:
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


#Activation function to get a covariance matrix - apply softplus or other activation functions on eigenvalues:    
def unstable_eig_val_cov_activ_func(X,activ_type="softplus"):
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


#Create a fully differentiable version:
def eig_val_cov_coverter(X,activ_type="softplus",tol=1e-7):
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
        Out[above_tol]=unstable_eig_val_cov_activ_func(X[above_tol],activ_type=activ_type)
    if activ_type=="softplus":
        Out[below_tol]=1e-3+F.softplus(torch.stack([X[below_tol][:,0],X[below_tol][:,2]],dim=1)).diag_embed()
    else: 
        sys.exit("Unknown activation type")
    return(Out)

'''
_______________________________________________________________________________________
COVARIANCE CONVERTER
_______________________________________________________________________________________
'''

def cov_activ_func(Pre_Sigma_Grid,dim_cov_est):
    '''
    Pre_Sigma_Grid - torch.Tensor - shape (batch_size,n,dim_cov_est) if dim_cov_est=2,3,4 and (batch_size,n) if dim_cov_est=1
    dim_cov_est - 1,2,3 or 4 - gives type and dimension of covariance converter
    '''
    if dim_cov_est==1:
        Sigma_grid=0.1+0.9*F.sigmoid(Pre_Sigma_Grid).repeat(1,1,2)
        return(Sigma_grid.diag_embed())
    elif dim_cov_est==2:
        Sigma_grid=0.1+0.9*F.sigmoid(Pre_Sigma_Grid)
        return(Sigma_grid.diag_embed())
    elif dim_cov_est==3:
        return(eig_val_cov_coverter(Pre_Sigma_Grid)+torch.tensor([NOISE,NOISE],device=Pre_Sigma_Grid.device).diag_embed()[None,:])
    elif dim_cov_est==4:
        #Consider a vector of size 4 as a 2x2 matrix:
        Pre_Sigma_Grid=Pre_Sigma_Grid.view((Pre_Sigma_Grid.size(0),Pre_Sigma_Grid.size(1),2,2))
        #COmpute A-> A^TA componentwise and add some noise on the diagonal to make it numerically stable:
        return(torch.matmul(Pre_Sigma_Grid.transpose(2,3),Pre_Sigma_Grid)+torch.tensor([NOISE,NOISE],device=Pre_Sigma_Grid.device).diag_embed()[None,:])
    else:
        sys.exit("Error in covariance converter: dimension must be either 1,2,3 or 4.")

'''
_______________________________________________________________________________________
GET FIBER REPRESENTATION FOR THE DIFFERENT COVARIANCE CONVERTERS:
_______________________________________________________________________________________

'''

#The following function gives the fiber representation for the eigen value covariance converter:
def get_eig_val_cov_conv_rep(G_act):
    '''
        Input:
            G_act - instance of e2cnn.gspaces.r2.rot2d_on_r2.Rot2dOnR2 - underlying group
            
        Output:
            psd_rep - instance of e2cnn.group.Representation - group representation of the group representation before the covariance 
            feat_type_pre_rep - instance of G_CNN.FieldType - corresponding field type
    '''
    #Change of basis matrix:
    change_of_basis=np.array([[1,1.,0.],
                          [0.,0.,1.],
                          [1,-1.,0.]])
    
    #Get group order and control:
    N=G_act.fibergroup.order()
    if N<=3 and N!=-1: sys.exit("Group order is not valid.")

    if isinstance(G_act,gspaces.FlipRot2dOnR2):
        irreps=['irrep_0,0','irrep_1,2'] if N>4 else ['irrep_0,0','irrep_1,2','irrep_1,2']
    elif isinstance(G_act,gspaces.Rot2dOnR2):
        irreps=['irrep_0','irrep_2'] if N>4 else ['irrep_0','irrep_2','irrep_2']
    else:
        sys.exit("Error: Unknown group.")

    psd_rep=e2cnn.group.Representation(group=G_act.fibergroup,name="eig_val_rep",irreps=irreps,
                                   change_of_basis=change_of_basis,
                                   supported_nonlinearities=['n_relu'])

    return(psd_rep)

#The following function gives the fiber representation for the different of covariance converters:
def get_pre_cov_rep(G_act,dim_cov_est):
    '''
        G_act - instance of e2cnn.gspaces.r2.rot2d_on_r2.Rot2dOnR2 - underlying group
        dim_cov_est - int - either 1,2,3 or 4  - gives dimension and also the type of the covariance converter
    '''
    if dim_cov_est==1:
        return(G_act.trivial_repr)
    elif dim_cov_est==2:
        return(group.directsum(2*[G_act.trivial_repr]))
    elif dim_cov_est==3:
        return(get_eig_val_cov_conv_rep(G_act))
    elif dim_cov_est==4:
        if isinstance(G_act,gspaces.FlipRot2dOnR2):
            vec_rep=G_act.irrep(1,1)
        elif isinstance(G_act,gspaces.Rot2dOnR2):
            vec_rep=G_act.irrep(1)
        else:
            sys.exit('Error: unknown group.')  
        return(group.directsum(2*[vec_rep]))
    else:
        sys.exit('Error when loading pre covariance representation: dim_cov_est can only be 1,2,3 or 4')

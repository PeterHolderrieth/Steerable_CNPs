#LIBRARIES:
#Tensors:
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

#E(2)-steerable CNNs - librar"y:
from e2cnn import gspaces                                          
from e2cnn import nn as G_CNN   
#import e2cnn

#Plotting in 2d/3d:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
import matplotlib.cm as cm

#Tools:
import datetime
import sys
sys.path.append("../..")

#Own files:
import Kernel_and_GP_tools as GP

#A function to load the GP data which I sampled:
def load_GP_div_free_data(Id,folder='Test_data/2d_GPs/',batch_size=1,share_val_set=0.2,share_test_set=0.2):
    #X=np.load(folder+"GP_data_X"+Id+".npy")
    #Y=np.load(folder+"GP_data_Y"+Id+".npy")
    X=torch.tensor(X,dtype=torch.get_default_dtype())
    Y=torch.tensor(Y,dtype=torch.get_default_dtype())
    n,m,_=X.size()
    for it in range(X.size(0)):
        vec_shuffle=torch.randperm(m)
        X[it]=X[it][vec_shuffle]
        Y[it]=Y[it][vec_shuffle]
    ind_shuffle=torch.randperm(n)
    n_test_points=int(n*share_test_set//1)
    n_val_points=int(n*share_val_set//1)
    test_ind=ind_shuffle[:n_test_points]
    val_ind=ind_shuffle[:(n_test_points+n_val_points)]
    train_ind=ind_shuffle[(n_test_points+n_val_points):]
    
    GP_train_data=utils.TensorDataset(X[train_ind],Y[train_ind])
    GP_val_data=utils.TensorDataset(X[val_ind],Y[val_ind])
    GP_test_data=utils.TensorDataset(X[test_ind],Y[test_ind])
    GP_train_data_loader=utils.DataLoader(GP_train_data,batch_size=batch_size,shuffle=True,drop_last=True)
    GP_test_data_loader=utils.DataLoader(GP_test_data,batch_size=batch_size,shuffle=True,drop_last=True)
    GP_val_data_loader=utils.DataLoader(GP_val_data,batch_size=batch_size,shuffle=True,drop_last=True)
    return(GP_train_data_loader,GP_val_data_loader,GP_test_data_loader)
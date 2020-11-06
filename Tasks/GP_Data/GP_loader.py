#LIBRARIES:
#Tensors:
import math
import torch
import numpy as np

#Tools:
import datetime
import sys

#Own files:
import Kernel_and_GP_tools as GP
import Tasks.GP_Data.GP_dataset as MyDataset

#A function to load the GP data which was sampled earlier:
def load_GP_data_set(data_type,data_set='train',file_path=''):
    if data_type=='div_free':
        if data_set=='test':
            X=np.load(file_path+"GP_div_free/Data/GP_div_free_Test_X.npy")
            Y=np.load(file_path+"GP_div_free/Data/GP_div_free_Test_Y.npy")
        elif data_set=='train':
            X=np.load(file_path+"GP_div_free/Data/GP_div_free_Train_X.npy")
            Y=np.load(file_path+"GP_div_free/Data/GP_div_free_Train_Y.npy")
        elif data_set=='valid':
            X=np.load(file_path+"GP_div_free/Data/GP_div_free_Valid_X.npy")
            Y=np.load(file_path+"GP_div_free/Data/GP_div_free_Valid_Y.npy")
        else:
            sys.exit('Unkown data set. Must be either train, valid or test')

    elif data_type=='curl_free':
        if data_set=='test':
            X=np.load(file_path+"GP_curl_free/Data/GP_curl_free_Test_X.npy")
            Y=np.load(file_path+"GP_curl_free/Data/GP_curl_free_Test_Y.npy")
        elif data_set=='train':
            X=np.load(file_path+"GP_curl_free/Data/GP_curl_free_Train_X.npy")
            Y=np.load(file_path+"GP_curl_free/Data/GP_curl_free_Train_Y.npy")
        elif data_set=='valid':
            X=np.load(file_path+"GP_curl_free/Data/GP_curl_free_Valid_X.npy")
            Y=np.load(file_path+"GP_curl_free/Data/GP_curl_free_Valid_Y.npy")
        else:
            sys.exit('Unkown data set. Must be either train, valid or test')

    elif data_type=='rbf':
        if data_set=='test':
            X=np.load(file_path+"GP_rbf/Data/GP_rbf_Test_X.npy")
            Y=np.load(file_path+"GP_rbf/Data/GP_rbf_Test_Y.npy")
        elif data_set=='train':
            X=np.load(file_path+"GP_rbf/Data/GP_rbf_Train_X.npy")
            Y=np.load(file_path+"GP_rbf/Data/GP_rbf_Train_Y.npy")
        elif data_set=='valid':
            X=np.load(file_path+"GP_rbf/Data/GP_rbf_Valid_X.npy")
            Y=np.load(file_path+"GP_rbf/Data/GP_rbf_Valid_Y.npy")
        else:
            sys.exit('Unkown data set. Must be either train, valid or test')
    
    else:
        sys.exit("Unknown data type. Must be either rbf, div_free or curl_free.")

    #Convert to torch tensor:
    X=torch.tensor(X,dtype=torch.get_default_dtype())
    Y=torch.tensor(Y,dtype=torch.get_default_dtype())

    return(X,Y)

#A function to load a data set:
def give_GP_data_set(Min_n_cont,Max_n_cont,data_type,data_set='train',file_path="",n_total=None,transform=True):
    X,Y=load_GP_data_set(data_type=data_type,data_set=data_set,file_path=file_path)
    return(MyDataset.GPDataset(X,Y,Min_n_cont=Min_n_cont,Max_n_cont=Max_n_cont,n_total=n_total,transform=transform))
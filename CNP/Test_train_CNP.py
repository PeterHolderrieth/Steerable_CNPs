import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
from numpy import loadtxt
from numpy import savetxt

#Import own files:
import Enc_Dec_Models as models
sys.path.append('../')
import My_Tools
import Tasks.GP_Data.GP_div_free_circle.loader as DataLoader
import CNP_Model
import Training

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")

MIN_N_CONT=2
MAX_N_CONT=50
BATCH_SIZE=64
FILEPATH="../Tasks/GP_Data/GP_div_free_circle/"
train_dataset=DataLoader.give_GP_div_free_data_set(MIN_N_CONT,MAX_N_CONT,'train',file_path=FILEPATH)    
dim_X=2 
dim_Y=2
dim_R=128 
hidden_layers_encoder=[128,128,128] 
hidden_layers_decoder=[128,128]
CNP=CNP_Model.ConditionalNeuralProcess(dim_X,dim_Y,dim_Y,dim_R,hidden_layers_encoder,hidden_layers_decoder)
print(My_Tools.count_parameters(CNP,print_table=True))
Training.train_CNP(CNP,train_dataset,train_dataset,data_identifier="GP_circle",device=DEVICE,n_epochs=60,n_iterat_per_epoch=10000)

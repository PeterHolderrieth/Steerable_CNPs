#!/usr/bin/env python
# coding: utf-8 


#LIBRARIES:
#Tensors:
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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
import Steerable_CNP_Models as My_Models



#HYPERPARAMETERS:
torch.set_default_dtype(torch.float)
quiver_scale=15


'''
This file should implement:
- A flexible training function which is able to reload a state of a model and continue training after stopping training.
- A fast way of using big minibatches but still allowing for different number of context points within one minibatch
- A consistent way of defining losses (such that same of number of n_epochs*n_iterat_per epoch should give roughly the same results)
'''

def train_CNP(Steerable_CNP, train_data_loader,val_data_loader, device,
              Max_n_context_points,Min_n_context_points=2,n_epochs=10, n_iterat_per_epoch=10,
                 learning_rate=1e-3, weight_decay=None,shape_reg=None,n_plots=None):
        '''
        Input: 
          Steerable_CNP: Steerable_CNP Module (see above)

          train_data_loader,val_data_loader: torch.utils.data.DataLoader - gives training and validation data
                       every minibatch is a list of length 2 with 
                       1st element: data features - torch.tensor - shape (minibatch_size,n,2)
                       2nd element: data labels   - torch.tensor - shape (minibatch_size,n,2)
          device: instance of torch.device 
          n_epochs: int -number of epochs for training
          n_iterat_per_epoch: int number of training iterations per epoch
          learning_rate: float-learning rate for optimizer
          weight_decay: float- weight_decay for optimizer
          shape_reg: float/None - control shape regularizer (see My_Tools.shape_regularizer)
          n_plots:  if int: total number of plots during training
                    if None: no plots during training
        '''
        '''
        Input: filename - string - name of file - if given, there the model is saved
        Output:
          self.Steerable_CNP is trained (inplace)
          log_ll_vec: torch.tensor
                      Shape (n_epochs)
                      mean log likelihood over one epoch

        Prints:
          Prints self.n_prints-times the mean loss (per minibatch) of the last epoch
        
        Plots (if n_plots is not None):
          We choose a random function and random context before training and plot the development
          of the predictions (mean of the distributions) over the training
        '''
        minibatch_size=train_data_loader.batch_size

        #------------------Tracking training progress ----------------------
        #1.Track training loss:
        train_loss_tracker=torch.zeros(n_epochs)
        #2.Track validation loss:
        valid_loss_tracker=torch.zeros(n_epochs)
        #------------------------------------------------------------------------

        #Define the optimizer and add a weight decay term:
        optimizer=torch.optim.Adam(Steerable_CNP.parameters(),lr=learning_rate,weight_decay=weight_decay)        

        #-------------------EPOCH LOOP ------------------------------------------
        for epoch in range(n_epochs):
            #Track the loss over the epoch:
            loss_epoch=My_Tools.AverageMeter()
            #-------------------------ITERATION IN ONE EPOCH ---------------------
            for it in range(n_iterat_per_epoch):
                #Get the next minibatch and send it to device:
                features, labels=next(iter(train_data_loader))
                features=features.to(device)
                labels=labels.to(device)
                
                #Set the loss to zero:
                loss=torch.tensor(0.0,device=device)
                
                for el in range(minibatch_size):
                    #Sample new training example:
                    n_context_points=torch.randint(size=[],low=Min_n_context_points,high=Max_n_context_points)
                    x_context,y_context,_,_=My_Tools.Rand_Target_Context_Splitter(features[el],
                                                                                       labels[el],
                                                                                         n_context_points)
                    #The target set includes the context set here:
                    Means,Sigmas=Steerable_CNP(x_context,y_context,features[el]) 
                    loss+=Steerable_CNP.loss(labels[el],Means,Sigmas,shape_reg=self.shape_reg)
                
                loss_epoch.update(val=loss.detach().item(),n=self.minibatch_size)
                #Set gradients to zero:
                optimizer.zero_grad()
                #Compute gradients:
                loss.backward()
                
                #DEBUG:
                #Print l-scales:
                #l_scale_tracker[it]=self.Steerable_CNP.encoder.log_l_scale
                #l_scale_grad_tracker[i]=self.Steerable_CNP.encoder.log_l_scale.grad
                
                #Perform optimization step:
                optimizer.step()

                #DEBUG EXPLOSION OF L_SCALE:
                if self.Steerable_CNP.encoder.log_l_scale.item()!=self.Steerable_CNP.encoder.log_l_scale.item():
                    print("Tracker: ", l_scale_tracker[:(it+2)])
                    #print("Gradients :",l_scale_grad_tracker[:(it+2)])
                    print("Features: ", features)
                    print("Labels: ", labels)
                    print("Norm of features: ", torch.norm(features))
                    print("Norm of labels: ", torch.norm(labels))
                    print("Current l scale: ", self.Steerable_CNP.encoder.log_l_scale.item())
                    sys.exit("l scale is NAN")

            #Save the loss:
            loss_vector[epoch]=loss_epoch.avg
            #=----------------PRINT TRAINING EVOLUTION-------------------
            print("Epoch: ",epoch," | training loss: ", loss_epoch.avg," | val accuracy: ")
            if show_plots:
                if (epoch%plot_every==0):
                    self.Steerable_CNP.plot_Context_Target(Plot_x_context,Plot_y_context,Plot_x_target,Plot_y_target)
            #!!!DEBUG:
            #print("Encoder l_scale: ", torch.exp(self.Steerable_CNP.encoder.log_l_scale))
            #print("Encoder log l_scale grad: ", self.Steerable_CNP.encoder.log_l_scale.grad)
            #print("")



        #If a filename is given: save the model and add the date and time to the filename:
        if filename is not None:
            Report={'CNP': Steerable_CNP.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_data_loader': train_data_loader,
                    'val_data_loader': val_data_loader,
                    'n_epochs': n_epochs,
                    'n_iterat_per_epoch': n_iterat_per_epoch,
                    'loss history': loss_vector,
                    'Min n contest points':Min_n_context_points,
                    'Max n context points': Max_n_context_points,
                    'shape_reg': shape_reg}         
            complete_filename=filename+'_'+datetime.datetime.today().strftime('%Y_%m_%d_%H_%M')
            torch.save(Report,complete_filename)
            self.saved_to=complete_filename
        #Return the mean log-likelihood:
        return(self.log_ll_memory)
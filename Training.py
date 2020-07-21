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
- A consistent way of defining losses (such that same of number of n_epochs*n_iterat_per epoch should give roughly the same results)
'''

def train_CNP(Steerable_CNP, train_data_loader,val_data_loader, device,
              Max_n_context_points,Min_n_context_points=2,n_epochs=3, n_iterat_per_epoch=1,
                 learning_rate=1e-3, weight_decay=0.,shape_reg=None,n_plots=None,filename=None,n_val_samples=None):
        '''
        Input: 
          Steerable_CNP: Steerable_CNP Module (see above)

          train_data_loader,val_data_loader: torch.utils.data.DataLoader - gives training and validation data
                       every minibatch is a list of length 2 with 
                       1st element: data features - torch.tensor - shape (minibatch_size,n,2)
                       2nd element: data labels   - torch.tensor - shape (minibatch_size,n,2)
                       !!!We assume that the training set is shuffled along the observations in total
                       but also that the indices of the features and labels are shuffled.
          device: instance of torch.device 
          n_epochs: int -number of epochs for training
          n_iterat_per_epoch: int number of training iterations per epoch
          learning_rate: float-learning rate for optimizer
          weight_decay: float- weight_decay for optimizer
          shape_reg: float/None - control shape regularizer (see My_Tools.shape_regularizer)
          n_plots:  if int: total number of plots during training
                    if None: no plots during training
          filename: string/None - if not None, the training state is saved to a dictionary
          n_val_samples: None/int - if int, every epoch we compute the validation log likelihood
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
        Steerable_CNP=Steerable_CNP.to(device)

        #------------------Tracking training progress ----------------------
        #1.Track training loss and log-ll (if shape_reg=0, this is the same):
        train_loss_tracker=[]
        train_log_ll_tracker=[]
        #2.Track validation loss:
        val_log_ll_tracker=[]
        #------------------------------------------------------------------------

        #Define the optimizer and add a weight decay term:
        optimizer=torch.optim.Adam(Steerable_CNP.parameters(),lr=learning_rate,weight_decay=weight_decay)        

        #-------------------EPOCH LOOP ------------------------------------------
        for epoch in range(n_epochs):
            #Track the loss over the epoch:
            loss_epoch=My_Tools.AverageMeter()
            log_ll_epoch=My_Tools.AverageMeter()
            #-------------------------ITERATION IN ONE EPOCH ---------------------
            for it in range(n_iterat_per_epoch):
                #Get the next minibatch and send it to device:
                features, labels=next(iter(train_data_loader))
                features=features.to(device)
                labels=labels.to(device)
                
                #Set the loss to zero:
                loss=torch.tensor(0.0,device=device)
                
                #Sample new training example:
                n_context_points=torch.randint(size=[],low=Min_n_context_points,high=Max_n_context_points)
                x_context,y_context,_,_=My_Tools.Rand_Target_Context_Splitter(features, labels, n_context_points)
                #The target set includes the context set here:
                Means,Sigmas=Steerable_CNP(x_context,y_context,features) 
                loss,log_ll=Steerable_CNP.loss(labels,Means,Sigmas,shape_reg=shape_reg)

                #Set gradients to zero:
                optimizer.zero_grad()
                #Compute gradients:
                loss.backward()
                #Perform optimization step:
                optimizer.step()

                #Update trackers (n=1 since we have already averaged over the minibatch in the loss):
                loss_epoch.update(val=loss.detach().item(),n=1)
                log_ll_epoch.update(val=log_ll.detach().item(),n=1)

            #Save the loss and log ll on the training set:
            train_loss_tracker.append(loss_epoch.avg)
            train_log_ll_tracker.append(log_ll_epoch.avg)

            if n_val_samples is not None:
              val_log_ll=test_CNP(Steerable_CNP,val_data_loader,device,Min_n_context_points,Max_n_context_points,n_val_samples)
              val_log_ll_tracker.append(val_log_ll)
              print("Epoch: %d | train loss: %.5f | train log ll:  %.5f | val log ll: %.5f"%(epoch,loss_epoch.avg,log_ll_epoch.avg,val_log_ll))

            else:
              print("Epoch: %d | train loss: %.5f | train log ll:  %.5f "%(epoch,loss_epoch.avg,log_ll_epoch.avg))

        #If a filename is given: save the model and add the date and time to the filename:
        if filename is not None:
            complete_filename=filename+'_'+datetime.datetime.today().strftime('%Y_%m_%d_%H_%M')
            Report={'CNP_dict': Steerable_CNP.give_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_data_loader': train_data_loader,
                    'val_data_loader': val_data_loader,
                    'n_iterat_per_epoch': n_iterat_per_epoch,
                    'train_loss_history':   train_loss_tracker,
                    'train_log_ll_history': train_log_ll_tracker,
                    'val_log ll_history': val_log_ll_tracker,
                    'Min_n_context_points':Min_n_context_points,
                    'Max_n_context_points': Max_n_context_points,
                    'shape_reg': shape_reg}   
            torch.save(Report,complete_filename)
        else:
          complete_filename=None
        #Return the model and the loss memory:
        return(Steerable_CNP,train_loss_tracker,complete_filename)

def test_CNP(CNP,val_data_loader,device,Min_n_context,Max_n_context,n_samples=400):
        with torch.no_grad():
            n_iterat=n_samples//val_data_loader.batch_size
            log_ll=torch.tensor(0.0, device=device)
            for i in range(n_iterat):
                    #Get the next minibatch:
                    features, labels=next(iter(val_data_loader))
                    #Send it to the correct device:
                    features=features.to(device)
                    labels=labels.to(device)
                    #Set the loss to zero:
                    n_context_points=torch.randint(size=[],low=Min_n_context,high=Max_n_context)
                    x_context,y_context,_,_=My_Tools.Rand_Target_Context_Splitter(features, labels, n_context_points)
                    #The target set includes the context set here:
                    Means,Sigmas=CNP(x_context,y_context,features) 
                    _, log_ll_it=CNP.loss(labels,Means,Sigmas,shape_reg=None)
                    log_ll+=log_ll_it/n_iterat
        return(log_ll.item())


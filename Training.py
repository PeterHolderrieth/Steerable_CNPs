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

'''
TO DO:
How to save identity of data loader without actually having to save it for every model
'''

def train_CNP(Steerable_CNP, train_dataset,val_dataset, data_identifier,device,minibatch_size=1,n_epochs=3, n_iterat_per_epoch=1,
                 learning_rate=1e-3, weight_decay=0.,shape_reg=None,n_plots=None,n_val_samples=None,filename=None):
        '''
        Input: 
          Steerable_CNP: Steerable_CNP Module (see above)

          train_dataset,val_dataset - instance of Tasks.CNPDataSet - giving train and valid sets
          data_identifier - string - identifier for what data set was used
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
                #Set the loss to zero:
                loss=torch.tensor(0.0,device=device)
                x_context,y_context,x_target,y_target=train_dataset.get_rand_batch(batch_size=minibatch_size,cont_in_target=True)
                #Load data to device:
                x_context=x_context.to(device)
                y_context=y_context.to(device)
                x_target=x_target.to(device)
                y_target=y_target.to(device)

                #DEBUG:
                #The target set includes the context set here:
                Means,Sigmas=Steerable_CNP(x_context,y_context,x_target) 
                loss,log_ll=Steerable_CNP.loss(y_target,Means,Sigmas,shape_reg=shape_reg)

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
              val_log_ll=test_CNP(Steerable_CNP,val_dataset,device,n_val_samples,batch_size=minibatch_size)
              val_log_ll_tracker.append(val_log_ll)
              print("Epoch: %d | train loss: %.5f | train log ll:  %.5f | val log ll: %.5f"%(epoch,loss_epoch.avg,log_ll_epoch.avg,val_log_ll))

            else:
              print("Epoch: %d | train loss: %.5f | train log ll:  %.5f "%(epoch,loss_epoch.avg,log_ll_epoch.avg))

        #If a filename is given: save the model and add the date and time to the filename:
        if filename is not None:
            complete_filename=filename+'_'+datetime.datetime.today().strftime('%Y_%m_%d_%H_%M')
            Report={'CNP_dict': Steerable_CNP.give_dict(),
                    'optimizer': optimizer.state_dict(),
                    'data_identifier': data_identifier,
                    'n_iterat_per_epoch': n_iterat_per_epoch,
                    'train_loss_history':   train_loss_tracker,
                    'train_log_ll_history': train_log_ll_tracker,
                    'val_log ll_history': val_log_ll_tracker,
                    'Min_n_context_points':train_dataset.Min_n_cont,
                    'Max_n_context_points': train_dataset.Max_n_cont,
                    'shape_reg': shape_reg,
                    'n_parameters:': My_Tools.count_parameters(Steerable_CNP),
                    'final_log_ll:': val_log_ll_tracker[-1]}
            torch.save(Report,complete_filename)
        else:
          complete_filename=None
        #Return the model and the loss memory:
        return(Steerable_CNP,train_loss_tracker,complete_filename)

def test_CNP(CNP,val_dataset,device,n_samples=400,batch_size=1):
        with torch.no_grad():
            n_iterat=n_samples//batch_size
            log_ll=torch.tensor(0.0, device=device)
            for i in range(n_iterat):
                    #Get random minibatch:
                    x_context,y_context,x_target,y_target=val_dataset.get_rand_batch(batch_size=batch_size,cont_in_target=False)
                    
                    #Load data to device:
                    x_context=x_context.to(device)
                    y_context=y_context.to(device)
                    x_target=x_target.to(device)
                    y_target=y_target.to(device)

                    #The target set includes the context set here:
                    Means,Sigmas=CNP(x_context,y_context,x_target) 
                    _, log_ll_it=CNP.loss(x_target,Means,Sigmas,shape_reg=None)
                    log_ll+=log_ll_it/n_iterat
        return(log_ll.item())


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
from Equivariance_Tester import equiv_error_model as equiv_error


#HYPERPARAMETERS:
torch.set_default_dtype(torch.float)

'''
TO DO:
How to save identity of data loader without actually having to save it for every model
'''

def train_CNP(CNP, train_dataset,val_dataset, data_identifier,device,minibatch_size=1,n_epochs=3, n_iterat_per_epoch=1,
                 learning_rate=1e-3, weight_decay=0.,shape_reg=None,n_plots=None,n_val_samples=None,filename=None,print_progress=True,G_act=None,feature_in=None,n_equiv_samples=None):
        '''
        Input: 
          CNP: Module of a CNP type accepting context and target sets

          train_dataset,val_dataset - datasets with the function give_rand_batch giving a random batch of context and target set
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
          print_progress - Boolean - indicates whether progress is printed
          G_act - gspaces.gspaces - gspace to track equivariance loss 
          feature_in - G_CNN.FieldType - feature type of input to track equivariance loss 
        '''
        '''
        Input: filename - string - name of file - if given, there the model is saved
        Output:
          CNP is trained (inplace)
          log_ll_vec: torch.tensor
                      Shape (n_epochs)
                      mean log likelihood over one epoch

        Prints:
          Prints n_prints-times the mean loss (per minibatch) of the last epoch
        
        Plots (if n_plots is not None):
          We choose a random function and random context before training and plot the development
          of the predictions (mean of the distributions) over the training
        '''
        CNP=CNP.to(device)

        #------------------Tracking training progress ----------------------
        #1.Track training loss and log-ll (if shape_reg=0, this is the same):
        train_loss_tracker=[]
        train_log_ll_tracker=[]
        #2.Track validation loss:
        val_log_ll_tracker=[]
        if G_act is not None and feature_in is not None and n_equiv_samples is not None:
          equiv_loss_mean_tr=[]
          equiv_loss_mean_norm_tr=[]
          equiv_loss_cov_tr=[]
          equiv_loss_cov_norm_tr=[]
          equiv_loss_mean_val=[]
          equiv_loss_mean_norm_val=[]
          equiv_loss_cov_val=[]
          equiv_loss_cov_norm_val=[]
        #------------------------------------------------------------------------

        #Define the optimizer and add a weight decay term:
        optimizer=torch.optim.Adam(CNP.parameters(),lr=learning_rate,weight_decay=weight_decay)        

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
                Means,Sigmas=CNP(x_context,y_context,x_target) 
                #print("Means sample: ", Means.flatten()[:100])
                #print("Sigmas samples: ", Sigmas.flatten()[:100])
                loss,log_ll=CNP.loss(y_target,Means,Sigmas,shape_reg=shape_reg)

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

            if print_progress:
              if n_val_samples is not None:
                val_log_ll=test_CNP(CNP,val_dataset,device,n_val_samples,batch_size=minibatch_size)
                val_log_ll_tracker.append(val_log_ll)
                print("Epoch: %d | train loss: %.5f | train log ll:  %.5f | val log ll: %.5f"%(epoch,loss_epoch.avg,log_ll_epoch.avg,val_log_ll))

              else:
                print("Epoch: %d | train loss: %.5f | train log ll:  %.5f "%(epoch,loss_epoch.avg,log_ll_epoch.avg))

            if G_act is not None and feature_in is not None and n_equiv_samples is not None:
              train_equiv_loss_it=equiv_error(CNP,train_dataset,G_act,feature_in,device=device,n_samples=n_equiv_samples,batch_size=minibatch_size)
              val_equiv_loss_it=equiv_error(CNP,val_dataset,G_act,feature_in,device=device,n_samples=n_equiv_samples,batch_size=minibatch_size)
              equiv_loss_mean_tr.append(train_equiv_loss_it['loss_mean'])
              equiv_loss_mean_norm_tr.append(train_equiv_loss_it['loss_mean_normalized'])
              equiv_loss_cov_tr.append(train_equiv_loss_it['loss_sigma'])
              equiv_loss_cov_norm_tr.append(train_equiv_loss_it['loss_sigma_normalized'])
              equiv_loss_mean_val.append(val_equiv_loss_it['loss_mean'])
              equiv_loss_mean_norm_val.append(val_equiv_loss_it['loss_mean_normalized'])
              equiv_loss_cov_val.append(val_equiv_loss_it['loss_sigma'])
              equiv_loss_cov_norm_val.append(val_equiv_loss_it['loss_sigma_normalized'])
        
        #If a filename is given: save the model and add the date and time to the filename:
        if filename is not None:
            if G_act is not None and feature_in is not None and n_equiv_samples is not None:
              equiv_loss_train={'loss_mean': equiv_loss_mean_tr, 'loss_mean_norm': equiv_loss_mean_norm_tr,'loss_sigma': equiv_loss_cov_tr,'loss_sigma_norm': equiv_loss_cov_norm_tr}
              equiv_loss_val={'loss_mean': equiv_loss_mean_val, 'loss_mean_norm': equiv_loss_mean_norm_val,'loss_sigma': equiv_loss_cov_val,'loss_sigma_norm': equiv_loss_cov_norm_val}
            else:
              equiv_loss_train=None
              equiv_loss_val=None
            complete_filename=filename+'_'+datetime.datetime.today().strftime('%Y_%m_%d_%H_%M')
            Report={'CNP_dict': CNP.give_dict(),
                    'optimizer': optimizer.state_dict(),
                    'data_identifier': data_identifier,
                    'n_iterat_per_epoch': n_iterat_per_epoch,
                    'train_loss_history':   train_loss_tracker,
                    'train_log_ll_history': train_log_ll_tracker,
                    'val_log ll_history': val_log_ll_tracker,
                    'equiv_loss_train': equiv_loss_train,
                    'equiv_loss_val': equiv_loss_val,
                    'Min_n_context_points': train_dataset.Min_n_cont,
                    'Max_n_context_points': train_dataset.Max_n_cont,
                    'shape_reg': shape_reg,
                    'n_parameters:': My_Tools.count_parameters(CNP)}
            torch.save(Report,complete_filename)
        else:
          complete_filename=None
        #Return the model and the loss memory:
        return(CNP,train_loss_tracker,complete_filename)

def test_CNP(CNP,val_dataset,device,n_samples=400,batch_size=1,n_data_passes=1):
        with torch.no_grad():
            n_obs=val_dataset.n_obs
            n_samples_max=min(n_samples,n_obs)
            n_iterat=max(n_samples_max//batch_size,1)
            log_ll=torch.tensor(0.0, device=device)

            for j in range(n_data_passes):
                ind_list=torch.randperm(n_obs)[:n_samples_max]
                batch_ind_list=[ind_list[j*batch_size:(j+1)*batch_size] for j in range(n_iterat)]

                for it in range(n_iterat):
                    #Get random minibatch:
                    x_context,y_context,x_target,y_target=val_dataset.get_batch(inds=batch_ind_list[it],cont_in_target=False)
                    
                    #Load data to device:
                    x_context=x_context.to(device)
                    y_context=y_context.to(device)
                    x_target=x_target.to(device)
                    y_target=y_target.to(device)

                    #The target set includes the context set here:
                    Means,Sigmas=CNP(x_context,y_context,x_target) 
                    _, log_ll_it=CNP.loss(y_target,Means,Sigmas)
                    log_ll+=log_ll_it/n_iterat
                    
        return(log_ll.item()/n_data_passes)

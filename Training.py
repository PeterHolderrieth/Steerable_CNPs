#!/usr/bin/env python
# coding: utf-8 


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
- 
'''

def trainer(Steerable_CNP,train_data_loader,val_data_loader,Max_n_context_points,Min_n_context_points=2,n_epochs=10,
                 learning_rate=1e-3, n_plots=None,weight_decay=0.0,n_iterat_per_epoch=10,shape_reg=None):
        '''
        Input: 
          Steerable_CNP: Steerable_CNP Module (see above)

          data_loader: torch.utils.data.DataLoader
                       every minibatch is a list of length 2 with 
                       1st element: data features - torch.tensor - shape (minibatch_size,n,2)
                       2nd element: data labels   - torch.tensor - shape (minibatch_size,n,2)
          epochs: int -number of epochs for training
          learning_rate: float-learning rate for optimizer
          weight_decay: float- weight_decay for optimizer
          n_prints: int - total number of printed losses during training
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
        
        #------------------Tracking training progress ----------------------
        #-------------------------------------------------------------------
        #1.Track the mean loss of every epoch:
        loss_vector=torch.zeros(n_epochs)
        #2.Track with plots:
        #Show plots or not? 
        show_plots=(self.n_plots is not None)
        #If yes: pick a random function from the data loader and choose a random subset
        #of context points by saving their indices:
        if n_plots is not None:
            #Create a plot every?:
            plot_every=max(self.n_epochs//self.n_plots,1)
            #Get a random function:
            Plot_X,Plot_Y=next(iter(self.train_data_loader))
            #Number of context points is expected number of context points:
            n_context_points=self.Max_n_context_points//2
            
            #Split:
            Plot_x_context,Plot_y_context,Plot_x_target,Plot_y_target=My_Tools.Rand_Target_Context_Splitter(Plot_X[0],
                                                                                   Plot_Y[0],
                                                                                   n_context_points)
        #3.track test accuracy:
        #----NOT YET IMPLEMENTED -------------
        #------------------------------------------------------------------------

        #Define the optimizer and add a weight decay term:
        optimizer=torch.optim.Adam(self.Steerable_CNP.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)        
        
        #-------------------EPOCH LOOP ------------------------------------------
        
        for epoch in range(self.n_epochs):
            #Track the loss over the epoch:
            loss_epoch=My_Tools.AverageMeter()

            l_scale_tracker=torch.zeros([self.n_iterat_per_epoch],device=self.device)
            #l_scale_grad_tracker=torch.empty(self.n_iterat_per_epoch,device=self.device)

            #-------------------------ITERATION IN ONE EPOCH ---------------------
            #---------------------------------------------------------------------
            for it in range(self.n_iterat_per_epoch):
                #Get the next minibatch:
                features, labels=next(iter(self.train_data_loader))
                #Send it to the correct device:
                features=features.to(self.device)
                labels=labels.to(self.device)
                #Set the loss to zero:
                loss=torch.tensor(0.0,device=self.device)
                #loss_vec=torch.empty(self.minibatch_size) 
                for el in range(self.minibatch_size):
                    #Sample the number of context points uniformly: 
                    n_context_points=torch.randint(size=[],low=Min_n_context_points,high=Max_n_context_points)
                    
                    x_context,y_context,x_target,y_target=My_Tools.Rand_Target_Context_Splitter(features[el],
                                                                                       labels[el],
                                                                                         n_context_points)
                    #The target set includes the context set here:
                    Means,Sigmas=self.Steerable_CNP(x_context,y_context,features[el]) #Otherwise:Means,Sigmas=self.Steerable_CNP(x_context,y_context,x_target)
                    loss+=self.Steerable_CNP.loss(labels[el],Means,Sigmas,shape_reg=self.shape_reg)#/self.minibatch_size #Otherwise:loss+=self.Steerable_CNP.loss(y_target,Means,Sigmas)/self.minibatch_size
                
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
            print("Epoch: ",epoch," | training loss: ", loss_epoch.avg," | val accuracy: "+)
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


###USE torch.summary for summarizing structure of decoder.



class Steerable_CNP_Operator(nn.Module):
    def __init__(self,Steerable_CNP,train_data_loader,test_data_loader,Max_n_context_points,n_epochs=10,
                 learning_rate=1e-3,n_prints=None, n_plots=None,weight_decay=0.0,n_iterat_per_epoch=10,shape_reg=None,filename=None):
        super(Steerable_CNP_Operator, self).__init__()
        '''
        Input: 
          Steerable_CNP: Steerable_CNP Module (see above)

          data_loader: torch.utils.data.DataLoader
                       every minibatch is a list of length 2 with 
                       1st element: data features - torch.tensor - shape (minibatch_size,n,2)
                       2nd element: data labels   - torch.tensor - shape (minibatch_size,n,2)
          epochs: int -number of epochs for training
          learning_rate: float-learning rate for optimizer
          weight_decay: float- weight_decay for optimizer
          n_prints: int - total number of printed losses during training
          n_plots:  if int: total number of plots during training
                    if None: no plots during training
          filename - string - name of file - if given, there the model is saved
        '''

        self.Steerable_CNP=Steerable_CNP
        self.n_epochs=n_epochs
        self.learning_rate=learning_rate
        if n_prints is None:
            self.n_prints=n_epochs
        else:
            self.n_prints=n_prints

        self.n_plots=n_plots
        self.weight_decay=weight_decay
        self.Max_n_context_points=Max_n_context_points
        self.test_data_loader=test_data_loader
        self.train_data_loader=train_data_loader
        self.minibatch_size=train_data_loader.batch_size
        self.n_train_points=len(train_data_loader.dataset)
        self.n_grid_points=train_data_loader.dataset[0][0].size(0)
        self.log_ll_memory=nn.Parameter(torch.zeros(n_epochs,dtype=torch.get_default_dtype()),requires_grad=False)
        self.trained=False
        self.n_iterat_per_epoch=n_iterat_per_epoch
        self.saved_to=None
        #Get the device of the Steerable CNP (here, we assume that all parameters are on a single device):
        self.device=next(Steerable_CNP.parameters()).device
        self.shape_reg=shape_reg
                
    def train(self,filename=None,plot_loss=True):
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
        #Save the number of iterations the optimizer used per epoch:
        #n_iterat_per_epoch=self.n_train_points//self.minibatch_size+self.train_data_loader.drop_last

        #Define the optimizer and add a weight decay term:
        optimizer=torch.optim.Adam(self.Steerable_CNP.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)

        #Track the mean loss of every epoch:
        loss_vector=torch.zeros(self.n_epochs)
        #Print the loss every "track_every" iteration:
        track_every=self.n_epochs//self.n_prints        
        
        #Show plots or not? 
        show_plots=(self.n_plots is not None)
        
        #If yes: pick a random function from the data loader and choose a random subset
        #of context points by saving their indices:
        if show_plots:
            plot_every=max(self.n_epochs//self.n_plots,1)

            #Get a random function:
            Plot_X,Plot_Y=next(iter(self.train_data_loader))
            #Number of context points is expected number of context points:
            n_context_points=self.Max_n_context_points//2
            
            #Split:
            Plot_x_context,Plot_y_context,Plot_x_target,Plot_y_target=My_Tools.Rand_Target_Context_Splitter(Plot_X[0],
                                                                                   Plot_Y[0],
                                                                                   n_context_points)
        
        for epoch in range(self.n_epochs):
            #Track the loss over the epoch:
            loss_epoch=My_Tools.AverageMeter()

            l_scale_tracker=torch.zeros([self.n_iterat_per_epoch],device=self.device)
            #l_scale_grad_tracker=torch.empty(self.n_iterat_per_epoch,device=self.device)
            for it in range(self.n_iterat_per_epoch):
                #Get the next minibatch:
                features, labels=next(iter(self.train_data_loader))
                #Send it to the correct device:
                features=features.to(self.device)
                labels=labels.to(self.device)
                #Set the loss to zero:
                loss=torch.tensor(0.0,device=self.device)
                #loss_vec=torch.empty(self.minibatch_size) 
                for el in range(self.minibatch_size):
                    #Sample the number of context points uniformly: 
                    n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
                    
                    x_context,y_context,x_target,y_target=My_Tools.Rand_Target_Context_Splitter(features[el],
                                                                                       labels[el],
                                                                                         n_context_points)
                    #The target set includes the context set here:
                    Means,Sigmas=self.Steerable_CNP(x_context,y_context,features[el]) #Otherwise:Means,Sigmas=self.Steerable_CNP(x_context,y_context,x_target)
                    loss+=self.Steerable_CNP.loss(labels[el],Means,Sigmas,shape_reg=self.shape_reg)#/self.minibatch_size #Otherwise:loss+=self.Steerable_CNP.loss(y_target,Means,Sigmas)/self.minibatch_size
                
                loss_epoch.update(val=loss.detach().item(),n=self.minibatch_size)
                #Set gradients to zero:
                optimizer.zero_grad()
                #Compute gradients:
                loss.backward()
                #Print l-scales:
                l_scale_tracker[it]=self.Steerable_CNP.encoder.log_l_scale
                #l_scale_grad_tracker[i]=self.Steerable_CNP.encoder.log_l_scale.grad
                #Perform optimization step:
                optimizer.step()

                if self.Steerable_CNP.encoder.log_l_scale.item()!=self.Steerable_CNP.encoder.log_l_scale.item():
                    print("Tracker: ", l_scale_tracker[:(it+2)])
                    #print("Gradients :",l_scale_grad_tracker[:(it+2)])
                    print("Features: ", features)
                    print("Labels: ", labels)
                    print("Norm of features: ", torch.norm(features))
                    print("Norm of labels: ", torch.norm(labels))
                    print("Current l scale: ", self.Steerable_CNP.encoder.log_l_scale.item())
                    sys.exit("l scale is NAN")

            #Track the loss:
            if (epoch%track_every==0):
                print("Epoch: ",epoch," | Loss: ", loss_epoch.avg)
                print("Encoder l_scale: ", torch.exp(self.Steerable_CNP.encoder.log_l_scale))
                print("Encoder log l_scale grad: ", self.Steerable_CNP.encoder.log_l_scale.grad)
                print("")
            
            if show_plots:
                if (epoch%plot_every==0):
                    self.Steerable_CNP.plot_Context_Target(Plot_x_context,Plot_y_context,Plot_x_target,Plot_y_target)
            
            #Save loss and compute gradients:
            loss_vector[epoch]=loss_epoch.avg
        
        self.log_ll_memory=nn.Parameter(-loss_vector.detach(),requires_grad=False)
        
        #Set trained to True:
        self.trained=True
        #If a filename is given: save the model and add the date and time to the filename:
        if filename is not None:
            complete_filename="Trained_Models/"+filename+'_'+datetime.datetime.today().strftime('%Y_%m_%d_%H_%M')
            torch.save(self.state_dict(),complete_filename)
            self.saved_to=complete_filename
        #Return the mean log-likelihood:
        return(self.log_ll_memory)
    
    def plot_log_ll_memory(self):
        plt.plot(self.log_ll_memory)
        plt.xlabel("Iteration")
        plt.ylabel("Log-likelihood")
    
    #A function which tests the ConvCNP by plotting the predictions:
    def plot_test(self,x_context,y_context,x_target=None,y_target=None,GP_parameters=None,title=""):
            plt.figure(plt.gcf().number+1)
            #plt.title(filename + "Trained model")
            self.Steerable_CNP.plot_Context_Target(x_context,y_context,x_target,y_target,title=title)
            if GP_parameters is not None:
                plt.figure(plt.gcf().number+1)
                Means_GP,Cov_Mat_GP,Var_GP=GP.GP_inference(x_context,y_context,x_target, **GP_parameters)
                Cov_Mat_GP=My_Tools.Get_Block_Diagonal(Cov_Mat_GP,size=2)
                My_Tools.Plot_Inference_2d(x_context,y_context,x_target,y_target,Predict=Means_GP,Cov_Mat=Cov_Mat_GP,title="GP inference")
    
    def plot_test_random(self,n_samples=4,GP_parameters=None):
        for i in range(n_samples):
            X,Y=next(iter(self.test_data_loader))
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            x_context,y_context,x_target,y_target=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
            self.plot_test(x_context,y_context,x_target,y_target,GP_parameters=GP_parameters)
    #A function which tests the model - i.e. it returns the average
    #log-likelihood on the test data:
    def test(self,n_samples=400):
        with torch.no_grad():
            n_iterat=n_samples//self.minibatch_size
            log_ll=torch.tensor(0.0,device=self.device)
            for i in range(n_iterat):
                #Get the next minibatch:
                    features, labels=next(iter(self.test_data_loader))
                    #Send it to the correct device:
                    features=features.to(self.device)
                    labels=labels.to(self.device)
                    #Set the loss to zero:
                    loss=torch.tensor(0.0,device=self.device)
                    #loss_vec=torch.empty(self.minibatch_size) 
                    for i in range(self.minibatch_size):
                        #Sample the number of context points uniformly: 
                        n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
                        
                        x_context,y_context,x_target,y_target=My_Tools.Rand_Target_Context_Splitter(features[i],
                                                                                        labels[i],
                                                                                            n_context_points)
                        #Predict the target set:
                        Means,Sigmas=self.Steerable_CNP(x_context,y_context,x_target) #Otherwise:Means,Sigmas=self.Steerable_CNP(x_context,y_context,x_target)
                        log_ll-=self.Steerable_CNP.loss(y_target,Means,Sigmas)/n_samples
        return(log_ll.item())

    def test_equivariance_encoder(self,n_samples=1,plot=True,inner_circle=True):
        '''
        Input: n_samples - int - number of context, target samples to consider
               plot - Boolean - indicates whether plots are generated for every group transformation
        Output: For every group element, it computes the "group equivariance error" of the encoder, i.e.
                the difference between the embedding of the transformed context set and the transformed 
                embedding of the non-transformed context set.
                returns: loss - float - mean aggregrated loss per sample
        '''
        #Loss summation:
        loss=torch.tensor(0.0)
        for i in range(n_samples):
            #Get random mini batch:
            X,Y=next(iter(self.test_data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,_,_=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
            #Get embedding:
            Embedding=self.Steerable_CNP.encoder(x_context,y_context)
            #Get geometric version of embedding:
            geom_Embedding=G_CNN.GeometricTensor(Embedding,self.Steerable_CNP.feature_emb)
            #Go over all (test) group elements:
            for g in self.Steerable_CNP.G_act.testing_elements:
                #Get matrix representation of g:
                M=torch.tensor(self.Steerable_CNP.feature_in.representation(g),dtype=torch.get_default_dtype())
                #Transform the context set:
                trans_x_context=torch.matmul(x_context,M.t())
                trans_y_context=torch.matmul(y_context,M.t())
                #Get embedding of transformed context:
                Embedding_trans=self.Steerable_CNP.encoder(trans_x_context,trans_y_context)
                #Get transformed embedding (of non-transformed context set)
                trans_Embedding=geom_Embedding.transform(g).tensor
                #Get distance/error between the two (in theory, it should be zero)
                loss_it=torch.norm(Embedding_trans-trans_Embedding)
                #If wanted, control difference only on the inner cycle:
                if inner_circle:
                    n=Embedding_trans.size(3)
                    ind=My_Tools.get_outer_circle_indices(n)
                    for i,j in ind:
                        Embedding_trans[0,:,i,j]=0
                        trans_Embedding[0,:,i,j]=0
                #Get distance/error between the two (in theory, it should be zero)
                loss_it=torch.norm(Embedding_trans-trans_Embedding)
                #Plot the embedding if wanted:
                if plot:
                    #Get the title:
                    title="Embedding of transformed context| Group: "+self.Steerable_CNP.G_act.name+ "  |  Element: "+str(g)+" | loss: "+str(loss_it.item())
                    self.Steerable_CNP.encoder.plot_embedding(Embedding_trans,trans_x_context,trans_y_context,title=title)
                    title="Transformed Embedding of orig. context| Group: "+self.Steerable_CNP.G_act.name+ "  |  Element: "+str(g)+" | loss: "+str(loss_it.item())
                    self.Steerable_CNP.encoder.plot_embedding(trans_Embedding,trans_x_context,trans_y_context,title=title)
                #Add to aggregated loss:
                loss=loss+loss_it
        #Divide aggregated loss by the number of samples:
        return(loss/n_samples)
            
    def test_equivariance_decoder(self,n_samples=1,plot=True,inner_circle=True):
        '''
        Input: n_samples - int - number of context, target samples to consider
        Output: For every group element, it computes the "group equivariance error" of the decoder, i.e.
                the difference between the decoder output of the transformed embedding and the transformed 
                decoder output of the non-transformed embedding.
                returns: loss - float - mean aggregrated loss per sample
        '''
        #Hyperparameter for size of plots:
        size_scale=1
        
        #Float to save mean aggregrated loss:
        loss=torch.tensor(0.0)
        for i in range(n_samples):
            #Get random mini batch:
            X,Y=next(iter(self.test_data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,_,_=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
            
            #Get embedding and version:
            Emb = self.Steerable_CNP.encoder(x_context,y_context)
            geom_Emb = G_CNN.GeometricTensor(Emb, self.Steerable_CNP.feature_emb)
            
            #Get output from decoder and geometric version:
            Out=self.Steerable_CNP.decoder(Emb)
            geom_Out = G_CNN.GeometricTensor(Out, self.Steerable_CNP.feature_out)
            
            #Get grid of encoder:
            grid=self.Steerable_CNP.encoder.grid
            
            #Go over all group (testing) elements:
            for g in self.Steerable_CNP.G_act.testing_elements:
                
                #Transform embedding:
                geom_Emb_transformed= geom_Emb.transform(g)
                Emb_transformed=geom_Emb_transformed.tensor
                
                #Get output of transformed embedding and geometric version:
                Out_transformed = self.Steerable_CNP.decoder(Emb_transformed)
                
                #Get transformed output:
                transformed_Out= geom_Out.transform(g).tensor
                #Set difference to zero out of the inner circle:
                if inner_circle:
                    n=Out_transformed.size(3)
                    ind=My_Tools.get_outer_circle_indices(n)
                    for i,j in ind:
                        transformed_Out[0,:,i,j]=0
                        Out_transformed[0,:,i,j]=0
                
                #Get the difference:        
                Diff=transformed_Out-Out_transformed
                normalizer=torch.norm(Out_transformed)
                #Get iteration loss and add:
                loss_it=torch.norm(Diff)/normalizer
                loss=loss+loss_it
                #If wanted, plot mean rotations:
                if plot:
                    Means_transformed=Out_transformed.squeeze()[:2].permute(dims=(2,1,0)).reshape(-1,2)
                    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(size_scale*10,size_scale*5))
                    plt.gca().set_aspect('equal', adjustable='box')
                    ax.quiver(grid[:,0],grid[:,1],Means_transformed[:,0].detach(),Means_transformed[:,1].detach(),scale=quiver_scale)
                    #Get the title:
                    title="Decoder Output | Group: "+self.Steerable_CNP.G_act.name+ "  |  Element: "+str(g)+" | loss: "+str(loss_it.item())
                    ax.set_title(title)
        #Get mean aggregrated loss:
        return(loss/n_samples)
        
    def test_equivariance_target_smoother(self,n_samples=1):
        '''
        Input: n_samples - int - number of context, target samples to consider
        Output: For every group element, it computes the "group equivariance error" of the target smoother, i.e.
                the difference between the target smoothing of the transformed decoder output and the transformed target 
                and the target smoothing of the decoder output and the transformed target 
                returns: loss - float - mean aggregrated loss per sample
                NOTE: THIS FUNCTION ONLY CONTROLS EQUIVARIANCE OF THE MEANS, NOT OF THE VARIANCE (since the prediction of variances is not equivariant)
        '''
        loss_Means=torch.tensor(0.0)
        loss_Sigmas=torch.tensor(0.0)
        for i in range(n_samples):
            #Get random mini batch:
            X,Y=next(iter(self.test_data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,x_target,_=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
            #Get embedding:
            Emb = self.Steerable_CNP.encoder(x_context,y_context)
            #Get output of embedding and geometric version:
            Out=self.Steerable_CNP.decoder(Emb)
            geom_Out=G_CNN.GeometricTensor(Out, self.Steerable_CNP.feature_out)
            
            #Get smoothed means on target:
            Means,_=self.Steerable_CNP.target_smoother(x_target,Out.squeeze())
            
            normalizer=torch.norm(Means)
            
            for g in self.Steerable_CNP.G_act.testing_elements:
                #Get representation on the output:
                M=torch.tensor(self.Steerable_CNP.feature_in.representation(g),dtype=torch.get_default_dtype())
                #Transform means, target and output:
                trans_Means=torch.matmul(Means,M.t())
                trans_x_target=torch.matmul(x_target,M.t())
                trans_geom_Out=geom_Out.transform(g)
                
                #Get means on transformed target and Output:
                Means_trans,_=self.Steerable_CNP.target_smoother(trans_x_target,
                                                           trans_geom_Out.tensor.squeeze())
                #Get current loss and add it to the aggregrated loss:
                loss_it=torch.norm(Means_trans-trans_Means)
                loss_Means=loss_Means+loss_it
        #Get mean aggregrated loss:
        return(loss_Means/n_samples,loss_Sigmas/n_samples)
    
    def test_equivariance_model(self,n_samples=1,plot=True,title=""):
        '''
        Input: n_samples - int - number of context, target samples to consider
        Output: For every group element, it computes the "group equivariance error" of the model, i.e.
                the difference between the model output of the transformed context and target set and the transformed 
                model output of the non-transformed context and target set.
                returns: loss - float - mean aggregrated loss per sample
        '''
        #Get loss:
        loss=torch.tensor(0.0)
        for i in range(n_samples):
            #Get random mini batch:
            X,Y=next(iter(self.test_data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,x_target,_=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
            #Get means and variances:
            Means,Sigmas=self.Steerable_CNP.forward(x_context,y_context,x_target)
            #Go over all group (testing) elements:
            for g in self.Steerable_CNP.G_act.testing_elements:
                #Get input representation of g and transform context:
                M_in=torch.tensor(self.Steerable_CNP.feature_in.representation(g),dtype=torch.get_default_dtype())
                trans_x_context=torch.matmul(x_context,M_in.t())
                trans_y_context=torch.matmul(y_context,M_in.t())
                #Get output representation of g and transform target (here output representation on means is same as input):
                M_out=torch.tensor(self.Steerable_CNP.feature_in.representation(g),dtype=torch.get_default_dtype())
                trans_x_target=torch.matmul(x_target,M_out.t())
                
                #Get means and variances of transformed context and transformed target:
                Means_trans,Sigmas_trans=self.Steerable_CNP.forward(trans_x_context,trans_y_context,trans_x_target)
                #Get transformed  means and variances:
                trans_Means=torch.matmul(Means,M_out.t())
                #???TRANS_SIGMA???
                #Compute the error and add to aggregrated loss:
                it_loss=torch.norm(Means_trans-trans_Means)
                loss=loss+it_loss
                #If wanted plot the inference:
                if plot:
                    sup_title=title+"Group: "+self.Steerable_CNP.G_act.name+ "  |  Element: "+str(g)+"| Loss "+str(it_loss.detach().item())
                    My_Tools.Plot_Inference_2d(trans_x_context,trans_y_context,trans_x_target,
                                           Y_Target=None,Predict=Means_trans.detach(),Cov_Mat=Sigmas_trans.detach(),title=sup_title)
        #Get mean aggregrated loss:
        return(loss/n_samples)


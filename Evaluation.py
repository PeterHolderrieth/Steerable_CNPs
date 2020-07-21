#!/usr/bin/env python
# coding: utf-8 

#%%

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
import Training



#HYPERPARAMETERS:
torch.set_default_dtype(torch.float)
quiver_scale=15

'''
TO DO:
Implement Equivariance tests numerically (see Spherical/Steerable CNP paper), validation loss tests,
Implement Equivariance plots (stabilized)
'''

class Steerable_CNP_Evaluater(nn.Module):
    def __init__(self,dictionary,G_act,feature_in,feature_out):
        super(Steerable_CNP_Evaluater, self).__init__()
        '''
        Input: dictionary - obtained from train_CNP

        '''
        self.dictionary=dictionary
        self.Steerable_CNP=My_Models.Steerable_CNP.create_model_from_dict(dictionary['CNP_dict'])
        self.Max_n_context_points=dictionary['Max_n_context_points']
        self.Min_n_context_points=dictionary['Min_n_context_points']
        self.train_data_loader=dictionary['train_data_loader']
        self.val_data_loader=dictionary['val_data_loader']
        self.train_loss=dictionary['train_loss_history']   
        self.train_log_ll=dictionary['train_log_ll_history']
        self.val_log_ll=dictionary['val_log ll_history']
        self.shape_reg=dictionary['shape_reg']
        self.G_act=G_act
        self.feature_in=feature_in
        self.feature_sigma=feature_sigma        
    
    def plot_loss_memory(self):
        fig,ax=plt.subplots(ncols=2,nrows=1)
        ax[0].set(xlabel='Iteration', ylabel='log ll',title='Loss over Iteration')
        ax[1].set(xlabel='Iteration', ylabel='log ll',title='Log LL over Iteration')
        ax[0].plot(self.train_loss)
        ax[1].plot(self.train_log_ll,label="Train set")
        if self.val_log_ll is not None:
            ax[1].plot(self.val_log_ll,label="Valid set")
        ax[1].legend()

    #A function which tests the ConvCNP by plotting the predictions:
    def plot_test(self,x_context,y_context,x_target=None,y_target=None,GP_parameters=None,title=""):
        '''
        Input: x_context,y_context -torch.tensor - shape (1,n,2)
               x_target,y_target - torch.tensor - shape (1,n,2)
               GP_parameters - dictionary - parameters of GP to compare to
        '''
        plt.figure(plt.gcf().number+1)
        self.Steerable_CNP.plot_Context_Target(x_context,y_context,x_target,y_target,title=title)
        if GP_parameters is not None:
            plt.figure(plt.gcf().number+1)
            Means_GP,Cov_Mat_GP,Var_GP=GP.GP_inference(x_context[0],y_context[0],x_target[0], **GP_parameters)
            Cov_Mat_GP=My_Tools.Get_Block_Diagonal(Cov_Mat_GP,size=2)
            My_Tools.Plot_Inference_2d(x_context[0],y_context[0],x_target[0],y_target[0],Predict=Means_GP,Cov_Mat=Cov_Mat_GP,title="GP inference")
    
    def plot_test_random(self,n_samples=4,GP_parameters=None):
        for i in range(n_samples):
            #Get one single example:
            X,Y=next(iter(self.val_data_loader))
            X=X[0].unsqueeze(0)
            Y=Y[0].unsqueeze(0)
            #Split in context target:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            x_context,y_context,x_target,y_target=My_Tools.Rand_Target_Context_Splitter(X,Y,n_context_points)
            self.plot_test(x_context,y_context,x_target,y_target,GP_parameters=GP_parameters)

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
            x_context,y_context,_,_=My_Tools.Rand_Target_Context_Splitter(X,Y,n_context_points)
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
    
    def equiv_error_model(self,n_batches):
        '''
        Input:  n_batches - int - number of minibatches to consider
        Output: For every group element, it computes the "group equivariance error" of the model, i.e.
                the difference between the model output of the transformed context and target set and the transformed 
                output of the non-transformed context and target set divided by the norm 
                returns: loss - float - mean aggregrated loss per sample
        '''
        #Get loss:
        loss_mean=torch.tensor(0.0)
        loss_sigma=torch.tensor(0.0)

        for i in range(n_batches):
            #Get random mini batch:
            X,Y=next(iter(self.val_data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,x_target,_=My_Tools.Rand_Target_Context_Splitter(X,Y,n_context_points)
            #Get means and variances:
            Means,Sigmas=self.Steerable_CNP.forward(x_context,y_context,x_target)
            normalizer_mean=torch.sum(Means**2,dim=(1,2))
            normalizer_sigma=torch.sum(Sigmas**2,dim=(1,2,3))

            #Go over all group (testing) elements:
            for g in self.G_act.testing_elements:
                #Get input representation of g and transform context:
                M_in=torch.tensor(self.feature_in.representation(g),dtype=torch.get_default_dtype())
                trans_x_context=torch.matmul(x_context,M_in.t())
                trans_y_context=torch.matmul(y_context,M_in.t())
                trans_x_target=torch.matmul(x_target,M_in.t())

                #Get output representation of g and transform target (here output representation on means is same as input):
                #M_sigma=torch.tensor(self.feature_in.representation(g),dtype=torch.get_default_dtype())
                
                #Get means and variances of transformed context and transformed target:
                Means_trans,Sigmas_trans=self.Steerable_CNP.forward(trans_x_context,trans_y_context,trans_x_target)
                #Get transformed  means and variances:
                trans_Means=torch.matmul(Means,M_in.t())
                #trans_Sigmas=torch.matmul(M_out,torch.matmul(Sigmas_trans,M_out.t()))
                #Compute the error and add to aggregrated loss and take the average over the batch:
                loss_mean+=(torch.sum((Means_trans-trans_Means)**2,dim=(1,2))/normalizer_mean.unsqueeze(1)).mean()
                #loss_sigma+=(torch.sum((trans_Sigmas-Sigmas_trans)**2,dim=(1,2,3))/normalizer_sigma.unsqueeze(1)).mean()
        
        #Get mean aggregrated loss:
        return(loss_mean/n_batches,loss_sigma/n_batches)
    
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
            X,Y=next(iter(self.val_data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,x_target,_=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
            #Get means and variances:
            Means,Sigmas=self.Steerable_CNP.forward(x_context,y_context,x_target)
            #Go over all group (testing) elements:
            for g in self.Steerable_CNP.G_act.testing_elements:
                #Get input representation of g and transform context:
                M_in=torch.tensor(self.feature_in.representation(g),dtype=torch.get_default_dtype())
                trans_x_context=torch.matmul(x_context,M_in.t())
                trans_y_context=torch.matmul(y_context,M_in.t())
                #Get output representation of g and transform target (here output representation on means is same as input):
                M_out=torch.tensor(self.feature_in.representation(g),dtype=torch.get_default_dtype())
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

Encoder=My_Models.Steerable_Encoder()
Decoder=My_Models.Cyclic_Decoder(fib_reps=[[1,0],[1,-1],[1,0]],kernel_sizes=[5,7],N=4,non_linearity=['NormReLU'])
G_act=Decoder.G_act
feature_in=G_CNN.FieldType(G_act,[G_act.irrep(1)])
feature_sigma=G_CNN.FieldType(G_act,[G_act.trivial_repr])
CNP=My_Models.Steerable_CNP(encoder=Encoder,decoder=Decoder,dim_cov_est=1)
GP_train_data_loader,GP_test_data_loader=GP.load_2d_GP_data(Id="37845",batch_size=3)
if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
_,_,filename=Training.train_CNP(CNP, GP_train_data_loader,GP_test_data_loader, device=device,
              Max_n_context_points=50,n_epochs=3,n_iterat_per_epoch=1,
              filename="Test_CNP",n_val_samples=50)
CNP_dict=torch.load(filename)
Evaluater=Steerable_CNP_Evaluater(CNP_dict,G_act,feature_in,feature_sigma)
Evaluater.plot_loss_memory()
GP_parameters={'l_scale':1,'sigma_var':1, 'kernel_type':"div_free",'obs_noise':1e-4,'B':None,'Ker_project':False}
Evaluater.plot_test_random(GP_parameters=GP_parameters)
print(Evaluater.equiv_error_model(n_batches=10))
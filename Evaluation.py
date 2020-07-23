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
IMPLEMENT SUMMARY PRINT OF MODEL 
'''

class Steerable_CNP_Evaluater(nn.Module):
    def __init__(self,dictionary,G_act,in_repr):
        super(Steerable_CNP_Evaluater, self).__init__()
        '''
        Input: dictionary - obtained from train_CNP    
               G_act      - instance of e2cnn.gspaces - underlying G-space 
               in_repr    - in_repr - instance of e2cnn.group.irrep - input representation

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

        #Save the G-space:
        self.G_act=G_act
        #Save the feature type of the input, the embedding and the output of the decoder:
        self.feature_in=G_CNN.FieldType(G_act,[in_repr])
        self.feature_emb=G_CNN.FieldType(G_act,[G_act.trivial_repr,in_repr])

        self.feature_mean_grid=self.feature_in #Feature type on out of decoder/grid
        if self.Steerable_CNP.dim_cov_est==1:
            sigma_grid_rep=G_act.trivial_repr
        else:
            sigma_grid_rep,_=My_Tools.get_pre_psd_rep(G_act)
        self.feature_sigma_grid=G_CNN.FieldType(G_act,[sigma_grid_rep])
        self.feature_out_grid=G_CNN.FieldType(G_act,[in_repr,sigma_grid_rep])

    
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

    def equiv_error_encoder(self,n_samples=1,inner_circle=False,plot_trans=False,trans_plot=False,plot_stable=False):
        '''
        Input: n_samples - int - number of context, target samples to consider
        Output: For every group element, it computes the "group equivariance error" of the encoder, i.e.
                the difference between the embedding of the transformed context set and the transformed 
                embedding of the non-transformed context set divided by the norm of the vector.
                returns: loss - float - mean aggregrated loss per sample
        '''
        #If n_samples is too large, do not plot:
        if n_samples>4:
            plot_trans=False
            plot_stable=False
            trans_plot=False

        #Initialize container for loss, number of batches to consider and number of group (testing) elements:
        loss=torch.tensor(0.0)
        loss_normalized=torch.tensor(0.0)
        n_batches=max(n_samples//self.val_data_loader.batch_size,1)
        n_testing_elements=len(list(self.G_act.testing_elements))

        for i in range(n_batches):
            #Get random mini batch:
            X,Y=next(iter(self.val_data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,_,_=My_Tools.Rand_Target_Context_Splitter(X,Y,n_context_points)
            #Get embedding:
            Embedding=self.Steerable_CNP.encoder(x_context,y_context)
            #Get geometric version of embedding:
            geom_Embedding=G_CNN.GeometricTensor(Embedding,self.feature_emb)

            #Get the mean l1 norm per batch:
            normalizer=torch.abs(Embedding).sum(1).mean([1,2])

            #Go over all (testing) group elements:
            for g in self.G_act.testing_elements:
                #Get matrix representation of g and transform context:
                M=torch.tensor(self.feature_in.representation(g),dtype=torch.get_default_dtype())                #Transform the context set:
                trans_x_context=torch.matmul(x_context,M.t())
                trans_y_context=torch.matmul(y_context,M.t())

                #Get embedding of transformed context:
                Embedding_trans=self.Steerable_CNP.encoder(trans_x_context,trans_y_context)
                #Get transformed embedding (of non-transformed context set)
                trans_Embedding=geom_Embedding.transform(g).tensor

                Diff=Embedding_trans-trans_Embedding
                #If wanted, control difference only on the inner cycle:
                if inner_circle: Diff=My_Tools.set_outer_circle_zero(Diff) 
                #Get distance/error between the two (in theory, it should be zero)
                loss_it=torch.abs(Diff).sum(1).mean()
                loss_it_normalized=(torch.abs(Diff).sum(1).mean([1,2])/normalizer).mean(0)
                
                #Add to aggregated loss (need to normalize by number of testing elements):
                loss=loss+loss_it/n_testing_elements
                loss_normalized=loss_normalized+loss_it_normalized/n_testing_elements
                
                #Plot the embedding of the transformed context:
                if plot_trans:
                    title="Embedding of transformed context| Group: "+self.G_act.name+ "  |  Element: "+str(g)+" | loss: "+str(loss_it.item())
                    self.Steerable_CNP.encoder.plot_embedding(Embedding_trans[0],trans_x_context[0],trans_y_context[0],title=title)
                
                #Plot the transformed embedding:
                if trans_plot:    
                    title="Transformed Embedding of orig. context| Group: "+self.G_act.name+ "  |  Element: "+str(g)+" | loss: "+str(loss_it.item())
                    self.Steerable_CNP.encoder.plot_embedding(trans_Embedding[0],trans_x_context[0],trans_y_context[0],title=title)
                
                #Plot the embedding of the transformed context transformed back again:
                if plot_stable:
                    geom_trans_Embedding=G_CNN.GeometricTensor(Embedding_trans,self.feature_emb)
                    back_Embedding_trans=geom_trans_Embedding.transform(self.G_act.fibergroup.inverse(g)).tensor
                    if inner_circle: back_Embedding_trans=My_Tools.set_outer_circle_zero(back_Embedding_trans)
                    title="Embedding of transformed context transformed back| Group: "+self.G_act.name+ "  |  Element: "+str(g)+" | loss: "+str(loss_it.item())
                    self.Steerable_CNP.encoder.plot_embedding(back_Embedding_trans[0],x_context[0],y_context[0],title=title)
                    
        #Divide aggregated loss by the number of samples:
        return(loss/n_batches,loss_normalized/n_batches)
            
    def equiv_error_decoder(self,n_samples=1,inner_circle=True):
        '''
        Input: n_samples - int - number of context, target samples to consider
        Output: For every group element, it computes the "group equivariance error" of the decoder, i.e.
                the difference between the decoder output of the transformed embedding and the transformed 
                decoder output of the non-transformed embedding.
                returns: loss - float - mean aggregrated loss per sample
        '''
        #Initialize container for loss, number of batches to consider and number of group (testing) elements:
        loss=torch.tensor(0.0)
        loss_normalized=torch.tensor(0.0)
        n_batches=max(n_samples//self.val_data_loader.batch_size,1)
        n_testing_elements=len(list(self.G_act.testing_elements))

        for i in range(n_batches):
            #Get random mini batch:
            X,Y=next(iter(self.val_data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,_,_=My_Tools.Rand_Target_Context_Splitter(X,Y,n_context_points)
            
            #Get embedding and version:
            Emb = self.Steerable_CNP.encoder(x_context,y_context)
            geom_Emb = G_CNN.GeometricTensor(Emb, self.feature_emb)
            
            #Get output from decoder and geometric version:
            Out=self.Steerable_CNP.decoder(Emb)
            geom_Out = G_CNN.GeometricTensor(Out, self.feature_out_grid)
            
            #Get the normalizer:
            normalizer=torch.abs(Out).sum(1).mean([1,2])

            #Get grid of encoder:
            grid=self.Steerable_CNP.encoder.grid
            
            #Go over all group (testing) elements:
            for g in self.G_act.testing_elements:
                #Transform embedding:
                Emb_transformed= geom_Emb.transform(g).tensor
                
                #Get output of transformed embedding and geometric version:
                Out_transformed = self.Steerable_CNP.decoder(Emb_transformed)
                
                #Get transformed output:
                transformed_Out= geom_Out.transform(g).tensor    

                #Get the difference and if wanted set it to zero outside of the inner circle:    
                Diff=transformed_Out-Out_transformed
                if inner_circle: Diff=My_Tools.set_outer_circle_zero(Diff)

                #Get error - mean l1 norm:
                loss_it=torch.abs(Diff).sum(1).mean()
                loss_it_normalized=(torch.abs(Diff).sum(1).mean([1,2])/normalizer).mean(0)
                
                #Add to aggregated loss (need to normalize by number of testing elements):
                loss=loss+loss_it/n_testing_elements
                loss_normalized=loss_normalized+loss_it_normalized/n_testing_elements

        #Get mean aggregrated loss:
        return(loss/n_batches,loss_normalized/n_batches)
        
    def equiv_error_target_smoother(self,n_samples=1):
        '''
        Input: n_samples - int - number of context, target samples to consider
        Output: For every group element, it computes the "group equivariance error" of the target smoother, i.e.
                the difference between the target smoothing of the transformed decoder output and the transformed target 
                and the target smoothing of the decoder output and the transformed target 
                returns: loss list - [float,float] - 
        '''

        #Initialize container for loss, number of batches to consider and number of group (testing) elements:
        loss_mean=torch.tensor(0.0)
        loss_mean_normalized=torch.tensor(0.0)
        loss_sigma=torch.tensor(0.0)
        loss_sigma_normalized=torch.tensor(0.0)

        n_batches=max(n_samples//self.val_data_loader.batch_size,1)
        n_testing_elements=len(list(self.G_act.testing_elements))

        for i in range(n_batches):
            #Get random mini batch:
            X,Y=next(iter(self.val_data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,x_target,_=My_Tools.Rand_Target_Context_Splitter(X,Y,n_context_points)
            #Get embedding:
            Emb = self.Steerable_CNP.encoder(x_context,y_context)
            #Get output of embedding and geometric version:
            Out=self.Steerable_CNP.decoder(Emb)
            geom_Out=G_CNN.GeometricTensor(Out, self.feature_out_grid)
            
            #Get smoothed means on target:
            Means,Sigmas=self.Steerable_CNP.target_smoother(x_target,Out)
            #Get squared norm per batch element as a normalizer:
            normalizer_mean=torch.abs(Means).sum(2).mean(1)
            normalizer_sigma=torch.abs(Sigmas).sum([2,3]).mean(1)

            for g in self.G_act.testing_elements:
                #Get representation on the output:
                M=torch.tensor(self.feature_in.representation(g),dtype=torch.get_default_dtype())
                #Transform means, target and output:
                trans_Means=torch.matmul(Means,M.t())
                trans_Sigmas=torch.matmul(M,torch.matmul(Sigmas,M.t()))
                trans_x_target=torch.matmul(x_target,M.t())
                trans_geom_Out=geom_Out.transform(g)
                
                #Get means on transformed target and Output:
                Means_trans,Sigmas_trans=self.Steerable_CNP.target_smoother(trans_x_target,
                                                           trans_geom_Out.tensor.squeeze())
                #Get the difference:
                Diff_means=Means_trans-trans_Means
                Diff_sigmas=Sigmas_trans-trans_Sigmas

                #Get the loss for the current iteration:
                loss_mean_it=torch.abs(Diff_means).sum(2).mean([0,1])
                loss_sigma_it=torch.abs(Diff_sigmas).sum([2,3]).mean([0,1])

                loss_mean_normalized_it=(torch.abs(Diff_means).sum(2).mean(1)/normalizer_mean).mean()
                loss_sigma_normalized_it=(torch.abs(Diff_sigmas).sum([2,3]).mean(1)/normalizer_sigma).mean()

                #Add to aggregated loss (need to normalize by number of testing elements):
                loss_mean=loss_mean+loss_mean_it/n_testing_elements
                loss_sigma=loss_sigma+loss_sigma_it/n_testing_elements

                loss_mean_normalized=loss_mean_normalized+loss_mean_normalized_it/n_testing_elements
                loss_sigma_normalized=loss_sigma_normalized+loss_sigma_normalized_it/n_testing_elements

        out_dict={'loss_mean': loss_mean.item()/n_batches,'loss_mean_normalized': loss_mean_normalized.item()/n_batches,
                  'loss_sigma': loss_sigma.item()/n_batches, 'loss_sigma_normalized': loss_sigma_normalized.item()/n_batches}
        #Get mean aggregrated loss:
        return(out_dict)
    
    def equiv_error_model(self,n_samples=10,plot_trans=False,trans_plot=False,plot_stable=False,title=""):
        '''
        Input:  n_samples - int - number of data samples to consider
        Output: For every group element, it computes the "group equivariance error" of the model, i.e.
                the difference between the model output of the transformed context and target set and the transformed 
                output of the non-transformed context and target set divided by the norm 
                returns: loss - float - mean aggregrated loss per sample
        '''
        #If n_samples is too large, do not plot:
        if n_samples>4:
            plot_trans=False
            plot_stable=False
            trans_plot=False

        #Initialize container for loss, number of batches to consider and number of group (testing) elements:
        loss_mean=torch.tensor(0.0)
        loss_mean_normalized=torch.tensor(0.0)
        loss_sigma=torch.tensor(0.0)
        loss_sigma_normalized=torch.tensor(0.0)

        n_batches=max(n_samples//self.val_data_loader.batch_size,1)
        n_testing_elements=len(list(self.G_act.testing_elements))

        for i in range(n_batches):
            #Get random mini batch:
            X,Y=next(iter(self.val_data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,x_target,_=My_Tools.Rand_Target_Context_Splitter(X,Y,n_context_points)
            #Get means and variances:
            Means,Sigmas=self.Steerable_CNP.forward(x_context,y_context,x_target)
            #Get squared norm per batch element as a normalizer:
            normalizer_mean=torch.abs(Means).sum(2).mean(1)
            normalizer_sigma=torch.abs(Sigmas).sum([2,3]).mean(1)

            #Go over all group (testing) elements:
            for g in self.G_act.testing_elements:
                #Get input representation of g and transform context:
                M=torch.tensor(self.feature_in.representation(g),dtype=torch.get_default_dtype())
                trans_x_context=torch.matmul(x_context,M.t())
                trans_y_context=torch.matmul(y_context,M.t())
                trans_x_target=torch.matmul(x_target,M.t())
                
                #Get means and variances of transformed context and transformed target:
                Means_trans,Sigmas_trans=self.Steerable_CNP.forward(trans_x_context,trans_y_context,trans_x_target)
                #Get transformed  means and variances:
                trans_Means=torch.matmul(Means,M.t())
                trans_Sigmas=torch.matmul(M,torch.matmul(Sigmas_trans,M.t()))

                #Get the difference:
                Diff_means=Means_trans-trans_Means
                Diff_sigmas=Sigmas_trans-trans_Sigmas

                #Get the loss for the current iteration:
                loss_mean_it=torch.abs(Diff_means).sum(2).mean([0,1])
                loss_sigma_it=torch.abs(Diff_sigmas).sum([2,3]).mean([0,1])

                loss_mean_normalized_it=(torch.abs(Diff_means).sum(2).mean(1)/normalizer_mean).mean()
                loss_sigma_normalized_it=(torch.abs(Diff_sigmas).sum([2,3]).mean(1)/normalizer_sigma).mean()

                #Add to aggregated loss (need to normalize by number of testing elements):
                loss_mean=loss_mean+loss_mean_it/n_testing_elements
                loss_sigma=loss_sigma+loss_sigma_it/n_testing_elements

                loss_mean_normalized=loss_mean_normalized+loss_mean_normalized_it/n_testing_elements
                loss_sigma_normalized=loss_sigma_normalized+loss_sigma_normalized_it/n_testing_elements
                
                if plot_trans:
                    sup_title=title+"Output of transformed input | "+"Group: "+self.G_act.name+ "  |  Element: "+str(g)+"| Loss mean: "+str(loss_mean_it.detach().item())\
                        +"| Loss sigma: "+str(loss_sigma_it.detach().item())
                    My_Tools.Plot_Inference_2d(trans_x_context[0],trans_y_context[0],trans_x_target[0],
                                           Y_Target=None,Predict=Means_trans[0].detach(),Cov_Mat=Sigmas_trans[0].detach(),title=sup_title)

                if plot_stable:
                    back_Means_trans=torch.matmul(Means_trans,M)
                    back_Sigmas_trans=torch.matmul(M.t(),torch.matmul(Sigmas_trans,M))
                    sup_title=title+"Back transformed output of transformed input | "+"Group: "+self.G_act.name+ "  |  Element: "+str(g)+"| Loss mean: "+str(loss_mean_it.detach().item())\
                        +"| Loss sigma: "+str(loss_sigma_it.detach().item())
                    My_Tools.Plot_Inference_2d(x_context[0],y_context[0],x_target[0],
                                           Y_Target=None,Predict=back_Means_trans[0].detach(),Cov_Mat=back_Sigmas_trans[0].detach(),title=sup_title)

        out_dict={'loss_mean': loss_mean.item()/n_batches,'loss_mean_normalized': loss_mean_normalized.item()/n_batches,
                  'loss_sigma': loss_sigma.item()/n_batches, 'loss_sigma_normalized': loss_sigma_normalized.item()/n_batches}
        #Get mean aggregrated loss:
        return(out_dict)


'''
Encoder=My_Models.Steerable_Encoder(l_scale=0.4,x_range=[-4,4],n_x_axis=20)
Decoder=My_Models.Cyclic_Decoder(hidden_fib_reps=[[1,-1],[1,-1]],kernel_sizes=[5,7,9],dim_cov_est=3,N=16,non_linearity=['NormReLU'])
G_act=Decoder.G_act
in_repr=G_act.irrep(1)
CNP=My_Models.Steerable_CNP(encoder=Encoder,decoder=Decoder,dim_cov_est=3)
GP_train_data_loader,GP_test_data_loader=GP.load_2d_GP_data(Id="37845",batch_size=3)

if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

_,_,filename=Training.train_CNP(CNP, GP_train_data_loader,GP_test_data_loader, device=device,
              Max_n_context_points=50,n_epochs=5,n_iterat_per_epoch=3,
              filename="Test_CNP",n_val_samples=50)

CNP_dict=torch.load("Test_CNP_2020_07_23_16_16")
Evaluater=Steerable_CNP_Evaluater(CNP_dict,G_act,in_repr)
#Evaluater.plot_loss_memory()
GP_parameters={'l_scale':1,'sigma_var':1, 'kernel_type':"div_free",'obs_noise':1e-4,'B':None,'Ker_project':False}
#Evaluater.plot_test_random(GP_parameters=GP_parameters)
print("Equiv. error:", Evaluater.equiv_error_model(n_samples=1,plot_stable=True))
'''


# %%

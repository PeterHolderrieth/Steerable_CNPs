#!/usr/bin/env python
# coding: utf-8

# To do:
# 1. Structure the class encoder and ConvCNP better: Allow for variable CNN to be defined
# (Is it necessary that the height and width of output feature map is the same the input height and width? Otherwise,
# it gets a mess)
# 2. Change the architecture such that it allows for minibatches of data sets (so far only: minibatch size is one)
# 3. Show in an example with plot that equivariance is not fulfilled (maybe one before training, one after traing)
# |

# In[1]:


#LIBRARIES:
#Tensors:
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

#E(2)-steerable CNNs - library:
from e2cnn import gspaces                                          
from e2cnn import nn as G_CNN        


#Plotting in 2d/3d:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
import matplotlib.cm as cm

#Tools:
import datetime

#Own files:
import Kernel_and_GP_tools as GP
import My_Tools
import ConvCNP_Models as ConvCNP


# In[2]:


#HYPERPARAMETERS:
#Set default as double:
torch.set_default_dtype(torch.float)

#%% Define the geometric/steerable version of the embedding:
class Geom_ConvCNP_Encoder(ConvCNP.ConvCNP_Encoder):
    def __init__(self,feature_in, G_act,x_range,y_range=None,n_x_axis=10,n_y_axis=None,kernel_dict={'kernel_type':"rbf"},normalize=True):
        self.feature_in=feature_in
        self.feature_emb=G_CNN.FieldType(G_act, [G_act.trivial_repr,G_act.irrep(1)])
        super(Geom_ConvCNP_Encoder, self).__init__(x_range=x_range,y_range=y_range,n_x_axis=n_x_axis,
             n_y_axis=n_y_axis,kernel_dict=kernel_dict,normalize=normalize)
    def forward(self,X,Y):
        Embedding=super(Geom_ConvCNP_Encoder,self).forward(X,Y)
        return(G_CNN.GeometricTensor(Embedding, self.feature_emb))
    
#%% Define the geometric/steerable version of the ConvCNP:
class Geom_ConvCNP(ConvCNP.ConvCNP):
   def __init__(self,encoder,feature_in, G_act,decoder,kernel_dict_out={'kernel_type':"rbf"},normalize_output=True,feature_out=None):       
       super(Geom_ConvCNP,self).__init__(encoder=encoder,decoder=decoder,kernel_dict_out=kernel_dict_out,normalize_output=normalize_output)
       
       self.G_act=G_act
       self.feature_in=feature_in
       self.feature_emb=G_CNN.FieldType(G_act, [G_act.trivial_repr,G_act.irrep(1)])
       self.feature_out=feature_in
        
   def forward(self,X_context,Y_context,X_target):
       #Y_target=super(Geom_ConvCNP,self).forward(X_context,Y_context,X_target)
       #1.Context Set -> Embedding (via Encoder) --> shape (3,self.encoder.n_y_axis,self.encoder.n_x_axis):
        Embedding=self.encoder(X_context,Y_context)
        
        #Make a geometric embedding out of the embedding:
        Geom_Embedding=G_CNN.GeometricTensor(Embedding, self.feature_emb)
        
        #2.Embedding ->Feature Map (via CNN) 
        Final_Feature_Map=self.decoder(Geom_Embedding).tensor.squeeze()

        #Split into mean and variance and "make variance positive" with softplus:
        return(self.target_smoother(X_target,Final_Feature_Map))

#%%
class Geom_ConvCNP_Operator(ConvCNP.ConvCNP_Operator):
    def __init__(self,Geom_ConvCNP,data_loader,Max_n_context_points,n_epochs=10,
                 learning_rate=1e-3,n_prints=None, n_plots=None,weight_decay=0.0,n_iterat_per_epoch=10):
        super(Geom_ConvCNP_Operator, self).__init__(Geom_ConvCNP,data_loader,Max_n_context_points,n_epochs=10,
                 learning_rate=1e-3,n_prints=None, n_plots=None,weight_decay=0.0,n_iterat_per_epoch=10)
    
    
    def test_equivariance_encoder(self,n_samples=1,plot=True):
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
            X,Y=next(iter(self.data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,_,_=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
            #Get embedding:
            Embedding=self.ConvCNP.encoder(x_context,y_context)
            #Get geometric version of embedding:
            geom_Embedding=G_CNN.GeometricTensor(Embedding,self.ConvCNP.feature_emb)
            #Go over all (test) group elements:
            for g in self.ConvCNP.G_act.testing_elements:
                #Get matrix representation of g:
                M=torch.tensor(self.ConvCNP.feature_in.representation(g),dtype=torch.get_default_dtype())
                #Transform the context set:
                trans_x_context=torch.matmul(x_context,M.t())
                trans_y_context=torch.matmul(y_context,M.t())
                #Get embedding of transformed context:
                Embedding_trans=self.ConvCNP.encoder(trans_x_context,trans_y_context)
                #Get transformed embedding (of non-transformed context set)
                trans_Embedding=geom_Embedding.transform(g).tensor
                #Get distance/error between the two (in theory, it should be zero)
                loss_it=torch.norm(Embedding_trans-trans_Embedding)
                #Get the title:
                title="Group: "+self.ConvCNP.G_act.name+ "  |  Element: "+str(g)+" | loss: "+str(loss_it.item())
                #Plot the embedding if wanted:
                if plot:
                    self.ConvCNP.encoder.plot_embedding(Embedding_trans,trans_x_context,trans_y_context,title=title)
                #Add to aggregated loss:
                loss=loss+loss_it
        #Divide aggregated loss by the number of samples:
        return(loss/n_samples)
        
    def test_equivariance_decoder(self,n_samples=1,plot=True):
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
            X,Y=next(iter(self.data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,_,_=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
            #Get embedding:
            Emb = self.ConvCNP.encoder(x_context,y_context)
            #Get geometric version of embedding:
            geom_Emb = G_CNN.GeometricTensor(Emb, self.ConvCNP.feature_emb)
            #Get output from decoder:
            Out=self.ConvCNP.decoder(geom_Emb)
            #Get grid of encoder:
            grid=self.ConvCNP.encoder.grid
            #Go over all group (testing) elements:
            for g in self.ConvCNP.G_act.testing_elements:
                #Transform embedding:
                geom_Emb_transformed= geom_Emb.transform(g)
                #Get output of transformed embedding:
                Out_transformed = self.ConvCNP.decoder(geom_Emb_transformed)
                #Get transformed output:
                transformed_Out= Out.transform(g)
                #Get iteration loss:
                loss_it=torch.norm(transformed_Out.tensor-Out_transformed.tensor)
                loss=loss+loss_it
                #If wanted, plot mean rotations:
                if plot:
                    Means_transformed=Out_transformed.tensor.squeeze()[:2].permute(dims=(2,1,0)).reshape(-1,2)
                    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(size_scale*10,size_scale*5))
                    plt.gca().set_aspect('equal', adjustable='box')
                    ax.quiver(grid[:,0],grid[:,1],Means_transformed[:,0].detach(),Means_transformed[:,1].detach())
                    #Get the title:
                    title="Decoder Output | Group: "+self.ConvCNP.G_act.name+ "  |  Element: "+str(g)+" | loss: "+str(loss_it.item())
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
        loss_Vars=torch.tensor(0.0)
        for i in range(n_samples):
            #Get random mini batch:
            X,Y=next(iter(self.data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,x_target,_=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
            #Get embedding:
            Emb = self.ConvCNP.encoder(x_context,y_context)
            #Get geometric version of embedding:
            geom_Emb = G_CNN.GeometricTensor(Emb, self.ConvCNP.feature_emb)
            Out=self.ConvCNP.decoder(geom_Emb)
            #Get smoothed means on target:
            Means,_=self.ConvCNP.target_smoother(x_target,Out.tensor.squeeze())
            for g in self.ConvCNP.G_act.testing_elements:
                #Get representation on the output:
                M=torch.tensor(self.ConvCNP.feature_out.representation(g),dtype=torch.get_default_dtype())
                #Transform means, target and output:
                trans_Means=torch.matmul(Means,M.t())
                trans_x_target=torch.matmul(x_target,M.t())
                trans_Out=Out.transform(g)
                #Get means on transformed target and Output:
                Means_trans,_=self.ConvCNP.target_smoother(trans_x_target,
                                                           trans_Out.tensor.squeeze())
                #Get current loss and add it to the aggregrated loss:
                loss_it=torch.norm(Means_trans-trans_Means)
                loss_Means=loss_Means+loss_it
        #Get mean aggregrated loss:
        return(loss_Means/n_samples,loss_Vars/n_samples)
    
    def test_equivariance_model(self,n_samples=1,plot=True):
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
            X,Y=next(iter(self.data_loader))
            #Get random number context points:
            n_context_points=torch.randint(size=[],low=2,high=self.Max_n_context_points)
            #Get random split in context and target set:
            x_context,y_context,x_target,_=My_Tools.Rand_Target_Context_Splitter(X[0],Y[0],n_context_points)
            #Get means and variances:
            Means,Vars=self.ConvCNP.forward(x_context,y_context,x_target)
            #Go over all group (testing) elements:
            for g in self.ConvCNP.G_act.testing_elements:
                #Get input representation of g and transform context:
                M_in=torch.tensor(self.ConvCNP.feature_in.representation(g),dtype=torch.get_default_dtype())
                trans_x_context=torch.matmul(x_context,M_in.t())
                trans_y_context=torch.matmul(y_context,M_in.t())
                #Get output representation of g and transform target:
                M_out=torch.tensor(self.ConvCNP.feature_out.representation(g),dtype=torch.get_default_dtype())
                trans_x_target=torch.matmul(x_target,M_out.t())
                
                #Get means and variances of transformed context and transformed target:
                Means_trans,Vars_trans=self.ConvCNP.forward(trans_x_context,trans_y_context,trans_x_target)
                #Get transformed  means and variances:
                trans_Means=torch.matmul(Means,M_out.t())
                #Compute the error and add to aggregrated loss:
                it_loss=torch.norm(Means_trans-trans_Means)
                loss=loss+it_loss
                #If wanted plot the inference:
                if plot:
                    title="Group: "+self.ConvCNP.G_act.name+ "  |  Element: "+str(g)+"| Loss "+str(it_loss.detach().item())
                    My_Tools.Plot_Inference_2d(trans_x_context,trans_y_context,trans_x_target,
                                           Y_Target=None,Predict=Means_trans.detach(),Cov_Mat=Vars_trans.detach(),title=title)
        #Get mean aggregrated loss:
        return(loss/n_samples)

        
#%%      
G_act = gspaces.Rot2dOnR2(N=4)
feat_type_in=G_CNN.FieldType(G_act, [G_act.irrep(1)])
feat_type_emb=G_CNN.FieldType(G_act, [G_act.trivial_repr,G_act.irrep(1)])
feat_type_hid_1=G_CNN.FieldType(G_act, [G_act.regular_repr])
feat_type_hid_2=G_CNN.FieldType(G_act, [G_act.regular_repr])
feat_type_out=G_CNN.FieldType(G_act, [G_act.irrep(1),G_act.trivial_repr,G_act.trivial_repr])
geom_decoder=G_CNN.SequentialModule(G_CNN.R2Conv(feat_type_emb, feat_type_out, kernel_size=5,padding=2))

'''geom_decoder = G_CNN.SequentialModule(G_CNN.R2Conv(feat_type_emb, feat_type_hid_1, kernel_size=5,padding=2),
                              G_CNN.ReLU(feat_type_hid_1),
                              G_CNN.R2Conv(feat_type_hid_1, feat_type_hid_2, kernel_size=7,padding=3),
                              G_CNN.ReLU(feat_type_hid_2),
                              G_CNN.R2Conv(feat_type_hid_2, feat_type_out, kernel_size=3,padding=1))

'''
grid_dict={'x_range':[-3,3],'y_range':[-3,3],'n_x_axis':5,'n_y_axis':5}
encoder=ConvCNP.ConvCNP_Encoder(**grid_dict,normalize=True)
geom_convcnp=Geom_ConvCNP(feature_in=feat_type_in,G_act=G_act,encoder=encoder,decoder=geom_decoder)
#%%
GP_data_loader=GP.load_2d_GP_data(Id="37845",batch_size=3)
GP_parameters={'l_scale':1,'sigma_var':1, 'kernel_type':"div_free",'obs_noise':1e-4,'B':None,'Ker_project':False}
#%%
Geom_CNP_Operator=Geom_ConvCNP_Operator(geom_convcnp,data_loader=GP_data_loader,Max_n_context_points=20,n_epochs=1,n_plots=5,n_iterat_per_epoch=2)

#%%Equivariance tests:
print(Geom_CNP_Operator.test_equivariance_model(n_samples=1))
#Geom_CNP_Operator.test_equivariance_model(n_samples=1)

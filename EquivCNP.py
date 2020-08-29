#LIBRARIES:
#Tensors:

import numpy as np
import math
import torch
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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#Own files:
import Kernel_and_GP_tools as GP
import My_Tools
import EquivDeepSets 
from Cov_Converter import cov_converter
import Decoder_Models as models
import Architectures
import Tasks.GP_Data.GP_div_free_circle.loader as DataLoader

#HYPERPARAMETERS and set seed:
torch.set_default_dtype(torch.float)


'''
-------------------------------------------------------------------------
--------------------------Equivariant CNP CLASS----------------------------
-------------------------------------------------------------------------
'''     
class EquivCNP(nn.Module):
    def __init__(self, encoder, decoder,dim_cov_est=3, dim_context_feat=2,
                         l_scale=1.,normalize_output=True,kernel_dict_out={'kernel_type':"rbf"}):
        '''
        Inputs:
            encoder - instance of EquivDeepSets.EquivDeepSets class above
            decoder - nn.Module - takes input (batch_size,dim_context_feat+1,height,width) and gives (batch_size,2+dim_cov_est,height,width) 
            dim_context_feat - int - dimension of the features of the context set (usually 2)
            l_scale - float - gives initialisation for learnable length parameter
            normalize_output  - Boolean - indicates whether kernel smoothing is performed with normalizing
            kernel_dict_out - gives parameters for kernel smoother of output
        '''
        #-----------------------SAVING OF PARAMETERS ----------------------------------
        super(EquivCNP, self).__init__()
        #Initialse the encoder:
        self.encoder=encoder
        #Decoder and save the type (Convolutional, Steerable and if Steerable which group)
        self.decoder=decoder
        self.decoder_type=decoder.__class__.__name__
        #Get the parameters for kernel smoother for the target set:
        self.log_l_scale_out=nn.Parameter(torch.log(torch.tensor(l_scale,dtype=torch.get_default_dtype())),requires_grad=True)
        #Get the other kernel parameters for the kernel smoother for the target set (others are fixed):
        self.kernel_dict_out=kernel_dict_out
        #Save whether output is normalized (i.e. kernel smoothing is performed with normalizing):
        self.normalize_output=normalize_output
        #Save the dimension of the covariance estimator of the last layer:
        self.dim_cov_est=dim_cov_est
        self.dim_context_feat=dim_context_feat
        #-----------------------SAVING of PARAMETERS FINISHED---------------------------------


        #--------------------CONTROL OF PARAMETERS -------------------------
        #So far, the dimension of the covariance estimator has to be either 1 or 3 
        #(i.e. number of output channels either 3 or 5):
        if not any(dim_cov_est==dim for dim in [1,2,3,4]): sys.exit("Dim_cov_est must be either 1,2,3 or 4.")
        if 'l_scale' in kernel_dict_out: sys.exit("Encoder error: l scale is variable and not fixed")
        if not isinstance(self.normalize_output,bool): sys.exit("Normalize output has to be boolean.")
        if not isinstance(l_scale,float): sys.exit("l_scale initialization has to be a float.")
        if not isinstance(encoder,EquivDeepSets.EquivDeepSets): sys.exit("Enoder is not correct.")
        if not isinstance(decoder, nn.Module): sys.exit("Decoder has to be nn.Module")
        #--------------------END CONTROL OF PARAMETERS----------------------
        '''
        #-------------------CONTROL WHETHER DECODER ACCEPTS AND RETURNS CORRECT SHAPES----
        test_input=torch.randn([5,1+self.dim_context_feat,encoder.n_y_axis,encoder.n_x_axis])  
        test_output=decoder(test_input)
        if len(test_output.shape)!=4 or test_output.size(0)!=test_input.size(0) or test_output.size(2)!=encoder.n_y_axis or test_output.size(3)!=encoder.n_x_axis:
            sys.exit("Decoder error: shape of output is not correct.")
        if (self.dim_cov_est+2)!=test_output.size(1):sys.exit("Number of output channels!=2+dim of cov estimation.")
        #-------------------END CONTROL WHETHER DECODER ACCEPTS AND RETURNS CORRECT SHAPES----
        '''
    #Define the function which maps the output of the decoder to
    #predictions on the target set based on kernel smoothing, i.e. the predictions on 
    #the target set are obtained by kernel smoothing of these points on the grid of encoder
    def target_smoother(self,X_target,Final_Feature_Map):
        '''
        Input: X_target - torch.tensor- shape (batch_size,n_target,2)
               Final_Feature_Map- torch.tensor - shape (batch_size,self.dim_cov_est+2,self.encoder.n_y_axis,self.encoder.n_x_axis)
        Output: Predictions on X_target - Means_target - torch.tensor - shape (batch_size,n_target,2)
                Covariances on X_target - Covs_target - torch.tensor - shape (batch_size,n_target,2,2)
        '''
        batch_size=X_target.size(0)
        #-----------SPLIT FINAL FEATURE MAP INTO MEANS AND COVARIANCE PARAMETERS----------
        #Reshape the Final Feature Map:
        Resh_Final_Feature_Map=Final_Feature_Map.permute(dims=(0,2,3,1)).reshape(batch_size,self.encoder.n_y_axis*self.encoder.n_x_axis,
                                                            self.dim_cov_est+2)
        #Split into mean and parameters for covariance:
        Means_grid=Resh_Final_Feature_Map[:,:,:2]
        Pre_Activ_Covs_grid=Resh_Final_Feature_Map[:,:,2:]
        #----------END SPLIT FINAL FEATURE MAP INTO MEANS AND COVARIANCE PARAMETERS----------

        #-----------APPLY ACITVATION FUNCTION ON COVARIANCES---------------------
        Covs_grid=cov_converter(Pre_Activ_Covs_grid,dim_cov_est=self.dim_cov_est)
        #-----------END APPLY ACITVATION FUNCTION ON COVARIANCES---------------------

        #-----------APPLY KERNEL SMOOTHING --------------------------------------
        #Set the lenght scale (clamp for numerical stability):
        l_scale=torch.exp(torch.clamp(self.log_l_scale_out,max=5.,min=-5.))
        #Create a batch-version of the grid (need shape (batch_size,n,2)):
        expand_grid=self.encoder.grid.unsqueeze(0).expand(batch_size,self.encoder.grid.size(0),2)
        #Means on Target Set (via Kernel smoothing) --> shape (batch_size,n_target,2):
        Means_target=GP.Batch_Kernel_Smoother_2d(X_Context=expand_grid,
                                          Y_Context=Means_grid,
                                           X_Target=X_target,normalize=self.normalize_output,
                                           l_scale=l_scale,**self.kernel_dict_out)
        
        #Create flattened version (needed for target smoother):
        Covs_grid_flat=Covs_grid.view(batch_size,self.encoder.n_y_axis*self.encoder.n_x_axis,-1)
        #3.Get covariances on target set--> shape (batch_size,n_target,4):
        Covs_target_flat=GP.Batch_Kernel_Smoother_2d(X_Context=expand_grid,
                                            Y_Context=Covs_grid_flat,
                                          X_Target=X_target,normalize=self.normalize_output,
                                          l_scale=l_scale,kernel_type="rbf")                                 
        #Reshape covariance matrices to proper matrices --> shape (batch_size,n_target,2,2):
        Covs_target=Covs_target_flat.view(batch_size,X_target.size(1),2,2)
        #-----------END APPLY KERNEL SMOOTHING --------------------------------------
        return(Means_target, Covs_target)

    #Define the forward pass of ConvCNP: 
    def forward(self,X_context,Y_context,X_target):
        '''
        Inputs:
            X_context: torch.tensor - shape (batch_size,n_context,2)
            Y_context: torch.tensor - shape (batch_size,n_context,2)
            X_target: torch.tensor - shape (batch_size,n_target,2)
        Outputs:
            Means_target: torch.tensor - shape (batch_size,n_target,2) - mean of predictions
            Sigmas_target: torch.tensor -shape (batch_size,n_target,2) - scale of predictions
        '''
        #1.Context Set -> Embedding (via Encoder) --> shape (batch_size,3,self.encoder.n_y_axis,self.encoder.n_x_axis):
        Embedding=self.encoder(X_context,Y_context)
        #print('Embedding: ', Embedding[0,1:].flatten()[:100])
        #2.Embedding ->Feature Map (via CNN) --> shape (batch_size,2+self.dim_cov_est,self.encoder.n_y_axis,self.encoder.n_x_axis):
        Final_Feature_Map=self.decoder(Embedding)
        #print('Final_Feature_Map: ', Final_Feature_Map.flatten()[:100])
        #Smooth the output:
        Means_target,Sigmas_target=self.target_smoother(X_target,Final_Feature_Map)
        #Sigmas_target=Sigmas_target.clamp(min=1e-1,max=10.)
        return(Means_target,Sigmas_target)
        
    def plot_Context_Target(self,X_Context,Y_Context,X_Target,Y_Target=None,title=""):
        '''
            Inputs: X_Context, Y_Context, X_Target: torch.tensor - shape (batch_size,n_context/n_target,2) 
                    Y_Target: torch.tensor - shape (n_context_points,2) - ground truth
            Output: None - plots predictions 
        
        '''
        #Get predictions:
        Means,Covs=self.forward(X_Context,Y_Context,X_Target)
        #Plot predictions against ground truth:
        for i in range(X_Context.size(0)):
            My_Tools.Plot_Inference_2d(X_Context[i],Y_Context[i],X_Target[i],Y_Target[i],Predict=Means[i].detach(),Cov_Mat=Covs[i].detach(),title=title)
    
    def loss(self,Y_Target,Predict,Covs,shape_reg=None):
        '''
            Inputs: Y_Target: torch.tensor - shape (batch_size,n,2) - Target set locations and vectors
                    Predict: torch.tensor - shape (batch_size,n,2) - Predictions of Y_Target at X_Target
                    Covs: torch.tensor - shape (batch_size,n,2,2) - covariance matrices of Y_Target at X_Target
                    shape_reg: float/None - if float gives the weight of the shape_regularizer term (see My_Tools.shape_regularizer)
            Output: -log_ll+shape_reg*shape_diff: log_ll is the log-likelihood at Y_Target given the parameters Predict and Covs
                                                  shape_diff is the "shape difference" (interpreted here as the variance
                                                  of the difference Prdict-Y_Target computed by My_Tools.shape_regularizer)
        '''
        log_ll_vec=My_Tools.batch_multivar_log_ll(Means=Predict,Covs=Covs,Data=Y_Target)
        log_ll=log_ll_vec.mean()
        if shape_reg is not None: 
            loss=-log_ll+shape_reg*My_Tools.shape_regularizer(Y_1=Y_Target,Y_2=Predict)
        else: 
            loss=-log_ll
        return(loss,log_ll)

    #Two functions to save the model in a dictionary:
    #1.Create a dictionary:
    def give_dict(self):
        dictionary={
            'encoder_dict': self.encoder.give_dict(),
            'decoder_dict': self.decoder.give_model_dict(),
            'decoder_class': self.decoder_type,
            'log_l_scale_out': self.log_l_scale_out.detach().item(),
            'normalize_output': self.normalize_output,
            'dim_context_feat': self.dim_context_feat,
            'dim_cov_est': self.dim_cov_est,
            'kernel_dict_out': self.kernel_dict_out
        }
        return(dictionary)
    #2.Save the dictionary in a file:
    def save_model_dict(self,filename):
        torch.save(self.give_dict(),f=filename)

    #Two functions to load the model from a dictionary:
    #1.Create model from dictionary:
    def create_model_from_dict(dictionary):
        '''
        Input: dictionary - dict - parameters to load into EquivCNP class (including weights and biases for decoder and encoder)
        Output: instance of EquivCNP with parameters as specified in dictionary
        '''
        #Load Encoder:
        Encoder=EquivDeepSets.EquivDeepSets(**dictionary['encoder_dict'])
        #Load Decoder (depending on type of decoder use different functions):
        if dictionary['decoder_class']=="EquivDecoder":
            Decoder=Architectures.EquivDecoder.create_model_from_dict(dictionary['decoder_dict'])
        elif dictionary['decoder_class']=="CNNDecoder":
            Decoder=Architectures.CNNDecoder.create_model_from_dict(dictionary['decoder_dict'])
        else:
            sys.exit("Unknown decoder type.")

        #Create model:
        Model=EquivCNP(encoder=Encoder,
                        decoder=Decoder,
                        dim_cov_est=dictionary['dim_cov_est'], 
                        kernel_dict_out=dictionary['kernel_dict_out'],
                        dim_context_feat=dictionary['dim_context_feat'],
                        l_scale=math.exp(dictionary['log_l_scale_out']), 
                        normalize_output=dictionary['normalize_output'])
        return(Model)

    #2. Load dictionary and from dictionary load model:
    def load_model_from_dict(filename):
        '''
        Input: filename - string -location of dictionary
        Output: instance of EquivCNP with parameters as specified in dictionary at path "filename"
        '''
        dictionary=torch.load(f=filename)
        print(list(dictionary.keys()))
        return(EquivCNP.create_model_from_dict(dictionary))

'''
DIM_COV_EST=3
encoder=EquivDeepSets.EquivDeepSets(x_range=[-10,10],n_x_axis=50)
decoder=models.get_CNNDecoder('little',dim_cov_est=DIM_COV_EST,dim_features_inp=2)
equivcnp=EquivCNP(encoder,decoder,DIM_COV_EST,dim_context_feat=2)

FILEPATH="Tasks/GP_Data/GP_div_free_circle/"
Dataset=DataLoader.give_GP_div_free_data_set(5,50,'train',file_path=FILEPATH)
n_samples=3
for i in range(n_samples):
    X_c,Y_c,X_t,Y_t=Dataset.get_rand_batch(batch_size=10)
    Means,Covs=equivcnp(X_c,Y_c,X_t)
'''

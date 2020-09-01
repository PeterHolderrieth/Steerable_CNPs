

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

#Tools:
import datetime
import sys

#!!!!!!!!THIS FUNCTION ASSUMES THAT THE INPUT AND OUTPUT FEATURES TYPES ARE THE SAME!!:
def equiv_error_model(CNP,dataset,G_act,feature_in,device,n_samples=10,batch_size=1,n_data_passes=1):
        '''
        Input:  CNP - a CNP-like which takes context sets and target sets
                dataset - a dataset giving random batches of context and target sets
                feature_in - G_CNN.FieldType - gives field type of input features
                n_samples - int - number of data samples to consider
                batch_size - batch size to use for samples
                n_data_passes - number of times one passes over the data set (every iteration is random because of random split in context and target sets.)

        Output: For every group element, it computes the "group equivariance error" of the model, i.e.
                the difference between the model output of the transformed context and target set and the transformed 
                output of the non-transformed context and target set divided by the norm 
                returns: loss - float - mean aggregrated loss per sample

                Specifically the error is computed by  mean(|Psi(g.f)(x)-g.Psi(f)(x)| for x in x_target)/mean(|Psi(f)(x)| for x in x_target)
        '''
        with torch.no_grad():
                #Initialize container for loss, number of batches to consider and number of group (testing) elements:
                loss_mean=torch.tensor(0.0)
                loss_mean_normalized=torch.tensor(0.0)
                loss_sigma=torch.tensor(0.0)
                loss_sigma_normalized=torch.tensor(0.0)

                #Save the number of batches:
                n_obs=dataset.n_obs
                n_samples_max=min(n_samples,n_obs)
                n_iterat=max(n_samples_max//batch_size,1)
                n_testing_elements=len(list(G_act.testing_elements))

                for j in range(n_data_passes):
                        ind_list=torch.randperm(n_obs)[:n_samples_max]
                        batch_ind_list=[ind_list[j*batch_size:(j+1)*batch_size] for j in range(n_iterat)]

                        for it in range(n_iterat):
                                #Get random mini batch:
                                x_context,y_context,x_target,y_target=dataset.get_batch(inds=batch_ind_list[it],cont_in_target=False)
                                x_context=x_context.to(device)
                                y_context=y_context.to(device)
                                x_target=x_target.to(device)
                                y_target=y_target.to(device)
                        
                                #Get means and variances:
                                Means,Sigmas=CNP.forward(x_context,y_context,x_target)
                                #Get squared norm per batch element as a normalizer:
                                normalizer_mean=torch.abs(Means).sum(2).mean(1)
                                normalizer_sigma=torch.abs(Sigmas).sum([2,3]).mean(1)

                                #Go over all group (testing) elements:
                                for g in G_act.testing_elements:
                                        #Get input representation of g and transform context:
                                        M=torch.tensor(feature_in.representation(g),dtype=torch.get_default_dtype(),device=device)
                                        trans_x_context=torch.matmul(x_context,M.t())
                                        trans_y_context=torch.matmul(y_context,M.t())
                                        trans_x_target=torch.matmul(x_target,M.t())
                                        
                                        #Get means and variances of transformed context and transformed target:
                                        Means_trans,Sigmas_trans=CNP.forward(trans_x_context,trans_y_context,trans_x_target)
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

                #Normalize over the number of samples taken:
                normalizer=n_data_passes*n_iterat
                #Save error in dictionary:
                out_dict={'loss_mean': loss_mean.item()/normalizer,'loss_mean_normalized': loss_mean_normalized.item()/normalizer,
                  'loss_sigma': loss_sigma.item()/normalizer, 'loss_sigma_normalized': loss_sigma_normalized.item()/normalizer}

        return(out_dict)

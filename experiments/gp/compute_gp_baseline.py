import torch
import torch.utils.data as utils
import torch.nn.functional as F
import numpy as np

import sys
import argparse
import datetime
sys.path.append('../..')

#Own files:
import kernel_and_gp_tools as GP
import my_utils
import tasks.gp.gp_loader as dataLoader

#Set device:
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.set_defaults(
    data=None,
    N_SAMPLES=None,
    BATCH_SIZE=30,
    N_data_PASSES=30)

#Arguments for task:
ap.add_argument("-data", "--data", type=str, required=True,help="data to use.")
ap.add_argument("-n_passes", "--N_data_PASSES", type=int, required=False,help="Number of data passes.")
ap.add_argument("-n_samples", "--N_SAMPLES", type=int, required=False,help="Number of data samples (only not None for debugging).")
ap.add_argument("-batch", "--BATCH_SIZE", type=int, required=False,help="Batch size.")


#Pass the arguments:
ARGS = vars(ap.parse_args())


#Compute the log-ll of the GP posterior on the data by sampling:
def compute_gp_log_ll(GP_parameters,dataset,device,n_samples=None,batch_size=1,n_data_passes=1):
        with torch.no_grad():
            n_obs=dataset.n_obs
            if n_samples is None: 
                n_samples=n_obs
            n_samples_max=min(n_samples,n_obs)
            n_iterat=max(n_samples_max//batch_size,1)
            log_ll=torch.tensor(0.0, device=device)

            for j in range(n_data_passes):
                ind_list=torch.randperm(n_obs)[:n_samples_max]
                batch_ind_list=[ind_list[j*batch_size:(j+1)*batch_size] for j in range(n_iterat)]

                for it in range(n_iterat):
                    #Get random minibatch:
                    x_context,y_context,x_target,y_target=dataset.get_batch(inds=batch_ind_list[it],cont_in_target=False)
                    
                    #Load data to device:
                    x_context=x_context.to(device)
                    y_context=y_context.to(device)
                    x_target=x_target.to(device)
                    y_target=y_target.to(device)

                    #The target set includes the context set here:
                    Means_list=[]
                    Sigmas_list=[]
                    for b in range(batch_size):
                        Means,Sigmas,_=GP.gp_inference(x_context[b],y_context[b],x_target[b],**GP_parameters)
                        Means_list.append(Means)
                        Sigmas=my_utils.get_block_diagonal(Sigmas,size=2)
                        Sigmas_list.append(Sigmas)
                    Means=torch.stack(Means_list,dim=0)
                    Sigmas=torch.stack(Sigmas_list,dim=0)
                    log_ll_it=my_utils.batch_multivar_log_ll(Means,Sigmas,y_target)
                    log_ll+=log_ll_it.mean()/n_iterat
                                        
        return(log_ll.item()/n_data_passes)


#Fixed hyperparameters:
FILEPATH="../../tasks/gp/"
MIN_N_CONT=5
MAX_N_CONT=50

dataSET=dataLoader.give_gp_data_set(MIN_N_CONT,MAX_N_CONT,ARGS['data'],'test',file_path=FILEPATH)                 

if ARGS['data']=='rbf':
    GP_parameters={'l_scale':5.,
    'sigma_var': 10., 
    'kernel_type':"rbf",
    'obs_noise':0.02}

elif ARGS['data']=='div_free':
    GP_parameters={'l_scale':5.,
    'sigma_var': 10., 
    'kernel_type':"div_free",
    'obs_noise':0.02}

elif ARGS['data']=='curl_free':
    GP_parameters={'l_scale':5.,
    'sigma_var': 10., 
    'kernel_type':"curl_free",
    'obs_noise':0.02}
else: 
    sys.exit("Unknown data type.")

print("Start time:", datetime.datetime.today())
#Run:
log_ll=compute_gp_log_ll(GP_parameters,dataSET,DEVICE,ARGS['N_SAMPLES'],ARGS['BATCH_SIZE'],ARGS['N_data_PASSES'])

#Print:
print("Mean log-likelihood on validation data set:")
print(log_ll)
print("End time: ", datetime.datetime.today())
print("GP data set with kernel: ", ARGS['data'])
print("Number of data passes: ", ARGS['N_data_PASSES'])

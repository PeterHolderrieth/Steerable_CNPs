import torch
import torch.utils.data as utils
import torch.nn.functional as F
import numpy as np

import sys
import argparse
import datetime
sys.path.append('../..')

#Own files:
import Kernel_and_GP_tools as GP
import My_Tools
import Tasks.ERA5.ERA5_Dataset as Dataset
import itertools 

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
    N_SAMPLES=None,
    BATCH_SIZE=30,
    N_DATA_PASSES=1,
    DATA_SET='train')

#Arguments for task:
ap.add_argument("-n_passes", "--N_DATA_PASSES", type=int, required=False,help="Number of data passes.")
ap.add_argument("-n_samples", "--N_SAMPLES", type=int, required=False,help="Number of data samples (only not None for debugging).")
ap.add_argument("-batch", "--BATCH_SIZE", type=int, required=False,help="Batch size.")
ap.add_argument("-data", "--DATA_SIZE", type=str, required=True,help="Size of data set. 'big' or 'small'.")
ap.add_argument("-mode","--DATA_SET",type=str,required=False,help="Type of data set: 'train', 'val', or 'test'")
ap.add_argument("-lscale", "--LSCALE", type=float, required=True,help="L scale of kernel.")
ap.add_argument("-sigma", "--SIGMA", type=float, required=True,help="Sigma scale of kernel.")
ap.add_argument("-noise", "--NOISE", type=float, required=True,help="Noise scale of kernel.")

#Pass the arguments:
ARGS = vars(ap.parse_args())


#Compute the log-ll of the GP posterior on the data by sampling:
def Compute_GP_log_ll(GP_parameters,dataset,device,n_samples=None,batch_size=1,n_data_passes=1):
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
                    B=torch.eye(4).to(device)
                    for b in range(batch_size):
                        Means,Sigmas,_=GP.GP_inference(x_context[b],y_context[b],x_target[b],**GP_parameters,B=B)
                        Means=Means[:,2:]
                        Means_list.append(Means)
                        Sigmas=My_Tools.Get_Block_Diagonal(Sigmas,size=4)
                        Sigmas=Sigmas[:,2:,2:]
                        Sigmas_list.append(Sigmas)
                    Means=torch.stack(Means_list,dim=0)
                    Sigmas=torch.stack(Sigmas_list,dim=0)

                    log_ll_it=My_Tools.batch_multivar_log_ll(Means,Sigmas,y_target)
                    log_ll+=log_ll_it.mean()/n_iterat
                                        
        return(log_ll.item()/n_data_passes)


#Fixed hyperparameters:
MIN_N_CONT=5
MAX_N_CONT=50

if ARGS['DATA_SIZE']=='small':
        PATH_TO_TRAIN_FILE="../../Tasks/ERA5/ERA5_US/Data/Train_Small_ERA5_US.nc"
        PATH_TO_VAL_FILE="../../Tasks/ERA5/ERA5_US/Data/Valid_Small_ERA5_US.nc"
        PATH_TO_TEST_FILE="../../Tasks/ERA5/ERA5_US/Data/Test_Small_ERA5_US.nc"
elif ARGS['DATA_SIZE']=='big':
        PATH_TO_TRAIN_FILE="../../Tasks/ERA5/ERA5_US/Data/Train_Big_ERA5_US.nc"
        PATH_TO_VAL_FILE="../../Tasks/ERA5/ERA5_US/Data/Valid_Big_ERA5_US.nc"
        PATH_TO_TEST_FILE="../../Tasks/ERA5/ERA5_US/Data/Test_Big_ERA5_US.nc"
        PATH_TO_TEST_CHINA_FILE="../../Tasks/ERA5/ERA5_China/Data/Test_Big_ERA5_China.nc"
else:
    sys.exit("Unknown data set.")

if ARGS['DATA_SET']=='train':
    dataset=Dataset.ERA5Dataset(PATH_TO_TRAIN_FILE,MIN_N_CONT,MAX_N_CONT,place='US',normalize=True,circular=True)
elif ARGS['DATA_SET']=='val':
    dataset=Dataset.ERA5Dataset(PATH_TO_VAL_FILE,MIN_N_CONT,MAX_N_CONT,place='US',normalize=True,circular=True)
elif ARGS['DATA_SET']=='test':
    dataset=Dataset.ERA5Dataset(PATH_TO_TEST_FILE,MIN_N_CONT,MAX_N_CONT,place='US',normalize=True,circular=True)
elif ARGS['DATA_SET']=='testChina':
    dataset=Dataset.ERA5Dataset(PATH_TO_TEST_CHINA_FILE,MIN_N_CONT,MAX_N_CONT,place='China',normalize=True,circular=True)
else:                                                                                                                                                                                                                  sys.exit("Unknown train mode.")

GP_parameters={'l_scale':ARGS['LSCALE'],
'sigma_var': ARGS['SIGMA'], 
'kernel_type':"rbf",
'obs_noise':ARGS['NOISE']}

print("GP parameters: ", GP_parameters)
print("Start time:", datetime.datetime.today())   
log_ll=Compute_GP_log_ll(GP_parameters,dataset,DEVICE,ARGS['N_SAMPLES'],ARGS['BATCH_SIZE'],ARGS['N_DATA_PASSES'])
print("Mean log-likelihood on data set:")
print(log_ll)


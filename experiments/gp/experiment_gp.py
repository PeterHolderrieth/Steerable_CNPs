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
import argparse
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append('../../')

#Own files:
import kernel_and_gp_tools as GP
import my_utils
import equiv_encoder 
import training
from cov_activ_func import cov_activ_func
import decoder_models as models
import architectures
import steercnp
import cnp.cnp_model as CNP_Model
import cnp.cnp_architectures as CNP_architectures
import tasks.gp.gp_loader as dataLoader

#HYPERPARAMETERS and set seed:
torch.set_default_dtype(torch.float)

'''
SET DEVICE:
'''
LIST_NAMES=["regular_small",
        "regular_middle",
        "regular_big",
        "regular_huge",
        "irrep_small",
        "irrep_middle",
        "irrep_big",
        "irrep_huge"
        ]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")


# Construct the argument parser
ap = argparse.ArgumentParser()
ap.set_defaults(
    BATCH_SIZE=30,
    N_EPOCHS=3,
    PRINT_PROGRESS=True,
    N_ITERAT_PER_EPOCH=1,
    LEARNING_RATE=1e-4, 
    DIM_COV_EST=4,
    N_VAL_SAMPLES=None,
    N_EVAL_SAMPLES=None,
    LENGTH_SCALE_OUT=5.,
    LENGTH_SCALE_IN=7.,
    TESTING_GROUP=None,
    N_EQUIV_SAMPLES=None,
    SHAPE_REG=None,
    N_data_PASSES=1,
    SEED=1997,
    FILENAME=None,
    DIV_FREE=False)

#Arguments for task:
ap.add_argument("-data", "--data", type=str, required=True,help="data set to use: rbf, div_free or curl_free")


#Arguments for architecture:
ap.add_argument("-G", "--GROUP", type=str, required=True,help="Group")
ap.add_argument("-A", "--ARCHITECTURE", type=str, required=True,help="Decoder architecture.")
ap.add_argument("-cov", "--DIM_COV_EST", type=int, required=False,help="Dimension of covariance estimation.")
ap.add_argument("-div", "--DIV_FREE", type=bool, required=False,help="Indicates whether to use divergence-free kernel at the output.")

#Arguments for training:
ap.add_argument("-batch", "--BATCH_SIZE", type=int, required=False,help="Batch size.")
ap.add_argument("-lr", "--LEARNING_RATE", type=float,required=False,help="Learning rate.")
ap.add_argument("-epochs", "--N_EPOCHS", type=int, required=False,help="Number of epochs.")
ap.add_argument("-it", "--N_ITERAT_PER_EPOCH", type=int, required=False,help="Number of iterations per epoch.")
ap.add_argument("-file", "--FILENAME", type=str, required=False,help="Number of iterations per epoch.")
ap.add_argument("-l", "--LENGTH_SCALE_IN", type=float, required=False,help="Length scale for encoder.")
ap.add_argument("-seed","--SEED", type=int, required=False, help="Seed for randomness.")
ap.add_argument("-shape","--SHAPE_REG", type=float, required=False, help="Shape Regularizer")
ap.add_argument("-continue","--CONTINUE",type=str,required=False,help="File to continue training")
#Arguments for tracking:
ap.add_argument("-n_val", "--N_VAL_SAMPLES", type=int, required=False,help="Number of validation samples.")
ap.add_argument("-track", "--PRINT_PROGRESS", type=bool, required=False,help="Print output?")
ap.add_argument("-n_eval", "--N_EVAL_SAMPLES", type=int, required=False,help="Number of evaluation samples after training.")
ap.add_argument("-n_test", "--N_TEST_data_PASSES", type=int, required=False,help="Number of data passes after training.")
ap.add_argument("-n_equiv_val", "--N_EQUIV_SAMPLES", type=int, required=False,help="Number of samples to evaluate equivariance error.")
ap.add_argument("-test_G", "--TESTING_GROUP", type=str, required=False, help="Group with respect to which equivariance is tested.")
ap.add_argument("-passes", "--N_data_PASSES", type=int, required=False, help="Passes through data used for evaluation.") 

#Pass the arguments:
ARGS = vars(ap.parse_args())

#Set the seed:
torch.manual_seed(ARGS['SEED'])
np.random.seed(ARGS['SEED'])

#Fixed hyperparameters:
X_RANGE=[-10,10]
N_X_AXIS=30
MIN_N_CONT=5
MAX_N_CONT=50
FILEPATH="../../tasks/gp/"
data_IDENTIFIER="gp"



train_dataset=dataLoader.give_gp_data_set(MIN_N_CONT,MAX_N_CONT,ARGS['data'],'train',file_path=FILEPATH)                 
val_dataset=dataLoader.give_gp_data_set(MIN_N_CONT,MAX_N_CONT,ARGS['data'],'valid',file_path=FILEPATH)                 

print()
print("Time: ", datetime.datetime.today())
print("Group:", ARGS['GROUP'])
print('Model type:', ARGS['ARCHITECTURE'])
#Define the encoder:
encoder=equiv_encoder.EquivEncoder(x_range=X_RANGE,n_x_axis=N_X_AXIS,l_scale=ARGS['LENGTH_SCALE_IN'])

#Define the correct encoder:
if ARGS['GROUP']=='CNP':
    if ARGS['CONTINUE'] is not None:
        train_dict=torch.load(ARGS['CONTINUE'])
        CNP_dict=train_dict['CNP_dict']
        CNP=CNP_Model.ConditionalNeuralProcess.create_model_from_dict(CNP_dict)
        print("Reloaded CNP model from training dict.")
    else:
        CNP=CNP_architectures.give_cnp_architecture(ARGS['ARCHITECTURE'])
else:
    if ARGS['GROUP']=='C16':
        decoder=models.get_C16_Decoder(ARGS['ARCHITECTURE'],dim_cov_est=ARGS['DIM_COV_EST'],context_rep_ids=[1])
    elif ARGS['GROUP']=='D4':
        decoder=models.get_D4_Decoder(ARGS['ARCHITECTURE'],dim_cov_est=ARGS['DIM_COV_EST'],context_rep_ids=[[1,1]])
    elif ARGS['GROUP']=='D8':
        decoder=models.get_D8_Decoder(ARGS['ARCHITECTURE'],dim_cov_est=ARGS['DIM_COV_EST'],context_rep_ids=[[1,1]])
    elif ARGS['GROUP']=='SO2':
        decoder=models.get_SO2_Decoder(ARGS['ARCHITECTURE'],dim_cov_est=ARGS['DIM_COV_EST'],context_rep_ids=[1])
    elif ARGS['GROUP']=='C4':
        decoder=models.get_C4_Decoder(ARGS['ARCHITECTURE'],dim_cov_est=ARGS['DIM_COV_EST'],context_rep_ids=[1])
    elif ARGS['GROUP']=='C8':
        decoder=models.get_C8_Decoder(ARGS['ARCHITECTURE'],dim_cov_est=ARGS['DIM_COV_EST'],context_rep_ids=[1])
    elif ARGS['GROUP']=='CNN':
        decoder=models.get_CNNDecoder(ARGS['ARCHITECTURE'],dim_cov_est=ARGS['DIM_COV_EST'],dim_features_inp=2) 
    else:
        sys.exit("Unknown architecture type.")
    
    if ARGS['DIV_FREE']:
        print("Used div free kernel in the output")
        CNP=steercnp.SteerCNP(encoder,decoder,ARGS['DIM_COV_EST'],dim_context_feat=2,l_scale=ARGS['LENGTH_SCALE_OUT'],
                            kernel_dict_out={'kernel_type':"div_free"},normalize_output=False)
    else:
        CNP=steercnp.SteerCNP(encoder,decoder,ARGS['DIM_COV_EST'],dim_context_feat=2,l_scale=ARGS['LENGTH_SCALE_OUT'])

#If equivariance is wanted, create the group and the fieldtype for the equivariance:
if ARGS['TESTING_GROUP']=='D4':
    G_act=gspaces.FlipRot2dOnR2(N=4)
    feature_in=G_CNN.FieldType(G_act,[G_act.irrep(1,1)])
elif ARGS['TESTING_GROUP']=='C16':
    G_act=gspaces.Rot2dOnR2(N=16)
    feature_in=G_CNN.FieldType(G_act,[G_act.irrep(1)])
else:
    G_act=None
    feature_in=None


print("Number of parameters: ", my_utils.count_parameters(CNP,print_table=False))

CNP,_,_=training.train_cnp(CNP,
                           train_dataset=train_dataset,
                           val_dataset=val_dataset,
                           data_identifier=data_IDENTIFIER,
                           device=DEVICE,
                           minibatch_size=ARGS['BATCH_SIZE'],
                           n_epochs=ARGS['N_EPOCHS'],
                           n_iterat_per_epoch=ARGS['N_ITERAT_PER_EPOCH'],
                           learning_rate=ARGS['LEARNING_RATE'],
                           shape_reg=ARGS['SHAPE_REG'],
                           n_val_samples=ARGS['N_VAL_SAMPLES'],
                           print_progress=ARGS['PRINT_PROGRESS'],
                           filename=ARGS['FILENAME'],
                           n_equiv_samples=ARGS['N_EQUIV_SAMPLES'],
                           G_act=G_act,
                           feature_in=feature_in
                           )

print("Time finished with training: ", datetime.datetime.today())


#Final evaluation on validation data set:
if ARGS['N_EVAL_SAMPLES'] is not None:
    eval_log_ll=training.test_cnp(CNP,val_dataset,DEVICE,n_samples=ARGS['N_EVAL_SAMPLES'],batch_size=ARGS['BATCH_SIZE'],n_data_passes=ARGS['N_data_PASSES'])
    print("Final log ll:", eval_log_ll)
    print()

#Final evaluation on test data set:
if ARGS['N_TEST_data_PASSES'] is not None:
    test_dataset=dataLoader.give_gp_data_set(MIN_N_CONT,MAX_N_CONT,ARGS['data'],'test',file_path=FILEPATH)                 
    test_log_ll=training.test_cnp(CNP,test_dataset,DEVICE,n_samples=test_dataset.n_obs,batch_size=ARGS['BATCH_SIZE'],n_data_passes=ARGS['N_TEST_data_PASSES'])
    print("Final test log ll:", test_log_ll)
    print("Time finished with testing: ", datetime.datetime.today())

print()


import torch
import argparse
import sys
import datetime
sys.path.append("../../")


import training
import steercnp
import cnp.cnp_model as CNP_Model
import tasks.ERA5.era5_dataset as dataset

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")

ap = argparse.ArgumentParser()
ap.set_defaults(
    TYPE='SteerCNP',
    N_data_PASSES=1,
    PLACE='US',
    BATCH_SIZE=30)

#Arguments for architecture:
ap.add_argument("-file", "--FILE", required=True, type=str)
ap.add_argument("-batch", "--BATCH", required=False,type=int)
ap.add_argument("-p", "--N_data_PASSES", required=False,type=int)
ap.add_argument("-type", "--TYPE", required=False,type=str)
ap.add_argument("-place","--PLACE", required=False,type=str)

#Pass arguments:
ARGS = vars(ap.parse_args())

train_dict=torch.load(ARGS['FILE'],map_location=torch.device('cpu'))

if ARGS['TYPE']=="SteerCNP":
    CNP=SteerCNP.SteerCNP.create_model_from_dict(train_dict['CNP_dict'])
else:
    CNP=CNP_Model.ConditionalNeuralProcess.create_model_from_dict(train_dict['CNP_dict'])

CNP=CNP.to(DEVICE)

if ARGS['PLACE']=='US':
    PATH_TO_TEST_FILE="../../tasks/ERA5/ERA5_US/data/Test_Big_ERA5_US.nc"
    print("Use US data.")
elif ARGS['PLACE']=='China':
    PATH_TO_TEST_FILE="../../tasks/ERA5/ERA5_China/data/Test_Big_ERA5_China.nc"
    print("Use China data.")
else:
    sys.exit("Unknown place.")

MIN_N_CONT=2
MAX_N_CONT=50
test_dataset=dataset.ERA5Dataset(PATH_TO_TEST_FILE,MIN_N_CONT,MAX_N_CONT,place=ARGS['PLACE'],normalize=True,circular=True)

log_ll=training.test_cnp(CNP,test_dataset,DEVICE,n_samples=test_dataset.n_obs,batch_size=ARGS['BATCH_SIZE'],n_data_passes=ARGS['N_data_PASSES'])
print("Filename: ", ARGS['FILE'])
print("Time: ", datetime.datetime.today())
print("Place: ", ARGS['PLACE'])
print("Test log-likelihood: ", log_ll)
print("Number of samples: ", ARGS['N_data_PASSES'])

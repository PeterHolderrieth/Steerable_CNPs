#Best so far: {'l_scale': 10.0, 'sigma_var': 0.1, 'kernel_type': 'rbf', 'obs_noise': 0.01}

lscale=10.
sigma=0.1
noise=0.01  
data=big
mode='test'

python Run_GP.py -data $data -lscale $lscale -sigma $sigma -noise $noise -mode $mode

mode='testChina'

python Run_GP.py -data $data -lscale $lscale -sigma $sigma -noise $noise -mode $mode 


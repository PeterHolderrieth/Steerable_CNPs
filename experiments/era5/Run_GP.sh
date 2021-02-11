#Best so far: {'l_scale': 15.0, 'sigma_var': 0.25, 'kernel_type': 'rbf', 'obs_noise': 0.01}
lscale=15.
sigma=0.25
noise=0.01  
data=big
mode='test'
passes=5

python Run_GP.py -data $data -lscale $lscale -sigma $sigma -noise $noise -mode $mode -n_passes $passes

mode='testChina'

python Run_GP.py -data $data -lscale $lscale -sigma $sigma -noise $noise -mode $mode -n_passes $passes


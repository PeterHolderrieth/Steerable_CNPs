#Best so far: {'l_scale': 15.0, 'sigma_var': 0.25, 'kernel_type': 'rbf', 'obs_noise': 0.01}

for lscale in 15.
do
    for sigma in 0.15 0.2 0.25 0.28 0.35 
    do
        for noise in 0.007 0.009 0.01 0.011 0.013  
        do 
             python Run_GP.py -data small -lscale $lscale -sigma $sigma -noise $noise
        done
    done
done

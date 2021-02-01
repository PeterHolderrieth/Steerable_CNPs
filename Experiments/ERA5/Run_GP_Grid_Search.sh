#Best so far: {'l_scale': 10.0, 'sigma_var': 0.1, 'kernel_type': 'rbf', 'obs_noise': 0.01}
for lscale in 9. 10.
do
    for sigma in 0.05 0.1 0.25 0.5 1.
    do
        for noise in 0.005 0.01 0.02 0.05 0.1 
        do 
             python Run_GP.py -data small -lscale $lscale -sigma $sigma -noise $noise
        done
    done
done

#Best so far: {'l_scale': 15.0, 'sigma_var': 0.25, 'kernel_type': 'rbf', 'obs_noise': 0.01}
for lscale in 14. 14.5 15. 15.5 16.
do
    for sigma in 0.25
    do
        for noise in 0.01
        do 
             python Run_GP.py -data small -lscale $lscale -sigma $sigma -noise $noise
        done
    done
done

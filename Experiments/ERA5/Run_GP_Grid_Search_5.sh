for lscale in 10.
do
    for sigma in 0.11 0.12 0.13
    do
        for noise in 0.007 0.009 0.01 0.011 0.013  
        do 
             python Run_GP.py -data small -lscale $lscale -sigma $sigma -noise $noise
        done
    done
done
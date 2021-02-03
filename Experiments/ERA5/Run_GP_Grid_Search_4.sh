for lscale in 10.
do
    for sigma in 0.07 0.08 0.09 0.1 
    do
        for noise in 0.007 0.009 0.01 0.011 0.013  
        do 
             python Run_GP.py -data small -lscale $lscale -sigma $sigma -noise $noise
        done
    done
done
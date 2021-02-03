for lscale in 14.2 14.7 14.9 15. 15.1 15.3 15.7
do
    for sigma in 0.01 
    do
        for noise in 0.01 
        do 
             python Run_GP.py -data small -lscale $lscale -sigma $sigma -noise $noise
        done
    done
done

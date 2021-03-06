#Get the number of iterations, epochs and tests needed!
epochs=30
it=1500
n_test=15
n_val=100
track=True

for seed in 1 2 3 4 5  
do 
    python experiment_gp.py -lr 5e-4 -batch 5 -epochs $epochs -it $it -track True -G C16 -A regular_big -n_test $n_test -n_val $n_val -l 3.  -seed $seed -data rbf > results/C16/C16CNP_rbf_${seed}.txt
done  
  

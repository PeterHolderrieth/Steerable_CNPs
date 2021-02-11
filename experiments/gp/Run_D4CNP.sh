#Get the number of iterations, epochs and tests needed!
epochs=30
it=1500
n_test=15
n_val=100
track=True



for seed in 1 2 3 4 5 
do 
    python experiment_gp.py -lr 5e-4 -epochs 30 -it $it -track True -G D4 -A regular_big -n_test $n_test -n_val $n_val -l 3. -batch 5 -cov 4 -seed $seed -data rbf > results/D4/D4CNP_rbf_${seed}.txt
done 

for seed in 1 2 3 4 5 
do 
    python experiment_gp.py -lr 5e-4 -epochs 30 -it $it -track True -G D4 -A regular_big -n_test $n_test -n_val $n_val -l 3. -batch 5 -cov 4 -seed $seed -data div_free > results/D4/D4CNP_div_free_${seed}.txt
done 

for seed in 1 2 3 4 5 
do 
    python experiment_gp.py -lr 5e-4 -epochs 30 -it $it -track True -G D4 -A regular_big -n_test $n_test -n_val $n_val -l 3. -batch 5 -cov 4 -seed $seed -data curl_free > results/D4/D4CNP_curl_free_${seed}.txt
done 

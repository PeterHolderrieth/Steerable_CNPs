#Get the number of iterations, epochs and tests needed!
epochs=30
it=1500
n_test=5
n_val=100
track=True



for seed in 1 2 3 4 5 
do 
    python experiment_gp.py -lr 5e-4 -epochs $epochs -it $it -track True -G C4 -A regular_huge -n_test $n_test -n_val $n_val -l 3. -cov 4  -seed $seed -data rbf > Results/C4/C4CNP_rbf_${seed}.txt
done 

for seed in 1 2 3 4 5 
do 
    python experiment_gp.py -lr 5e-4 -epochs $epochs -it $it -track True -G C4 -A regular_huge -n_test $n_test -n_val $n_val -l 3. -cov 4  -seed $seed -data div_free > Results/C4/C4CNP_div_free_${seed}.txt
done 

for seed in 1 2 3 4 5 
do 
    python experiment_gp.py -lr 5e-4 -epochs $epochs -it $it -track True -G C4 -A regular_huge -n_test $n_test -n_val $n_val -l 3. -cov 4  -seed $seed -data curl_free > Results/C4/C4CNP_curl_free_${seed}.txt
done 

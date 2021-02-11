#Get the number of iterations, epochs and tests needed!
epochs=3
it=1
n_test=1
n_val=1
track=True



for seed in 1 2 3 4 5 
do 
    python experiment_gp.py -lr 5e-4 -epochs $epochs -it $it -track True -G C4 -A regular_huge -n_test $n_test -n_val $n_val -l 3. -cov 4  -seed $seed -data rbf 
done 

for seed in 1 2 3 4 5 
do 
    python experiment_gp.py -lr 5e-4 -epochs $epochs -it $it -track True -G C4 -A regular_huge -n_test $n_test -n_val $n_val -l 3. -cov 4  -seed $seed -data div_free 
done 

for seed in 1 2 3 4 5 
do 
    python experiment_gp.py -lr 5e-4 -epochs $epochs -it $it -track True -G C4 -A regular_huge -n_test $n_test -n_val $n_val -l 3. -cov 4  -seed $seed -data curl_free 
done 

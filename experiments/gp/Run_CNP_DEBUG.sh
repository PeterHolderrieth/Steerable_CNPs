#Get the number of iterations, epochs and tests needed!
epochs=3
it=1
n_val=1
track=True
n_test=1


for seed in 1 2 3 4 5 
do
    python experiment_gp.py -lr 1e-4 -epochs $epochs -A thin -it $it -track $track -G CNP -n_test $n_test -n_val $n_val -l 3. -batch 60  -seed $seed -data rbf 
done 

for seed in 1 2 3 4 5 
do
    python experiment_gp.py -lr 1e-4 -epochs $epochs -A thin -it $it -track $track -G CNP -n_test $n_test -n_val $n_val -l 3. -batch 60  -seed $seed -data div_free 
done 

for seed in 1 2 3 4 5 
do
    python experiment_gp.py -lr 1e-4 -epochs $epochs -A thin -it $it -track $track -G CNP -n_test $n_test -n_val $n_val -l 3. -batch 60  -seed $seed -data curl_free 
done 


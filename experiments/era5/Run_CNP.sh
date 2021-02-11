epochs=30
it=4000
n_val=100
n_test_US=20
n_test_China=20

for seed in 1 2 3 4 5
do 
    python experiment_era5.py -lr 1e-4 -epochs $epochs -A thin -it $it -track True -G CNP  -n_val $n_val -l 3. -batch 15 -data big -seed $seed -n_test_US $n_test_US -n_test_China $n_test_China > results/CNP/CNP_${seed}.txt
done



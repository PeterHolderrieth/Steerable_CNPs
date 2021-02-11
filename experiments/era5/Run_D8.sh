epochs=30
it=1500
n_test_US=20
n_test_China=20
n_val=100

for seed in 1 2 3 4 5
do 
	python experiment_era5.py -lr 5e-4 -epochs $epochs -it $it -track True -G D8 -batch 5 -A regular_middle -n_val $n_val -n_test_US $n_test_US -n_test_China $n_test_China -l 3. -cov 4 -data big -seed $seed > Results/D8/D8_${seed}.txt
done



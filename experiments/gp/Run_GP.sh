n_passes=5
batch=30
python compute_gp_baseline.py -data rbf -n_passes $n_passes -batch $batch >> Results/GP/gp_rbf_1.txt
python compute_gp_baseline.py -data div_free -n_passes $n_passes -batch $batch >> Results/GP/gp_div_free_1.txt
python compute_gp_baseline.py -data curl_free -n_passes $n_passes -batch $batch >> Results/GP/gp_curl_free_1.txt

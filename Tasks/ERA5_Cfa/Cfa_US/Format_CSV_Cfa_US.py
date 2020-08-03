import pandas as pd
old_filename="Data/ERA5_Cfa_US_unformated.csv"
new_filename="Data/ERA5_Cfa_US.csv"
#Split the longitude , latitude and the value in different columns:
with open(old_filename, 'r') as f_in, open(new_filename, 'w') as f_out:
    head=next(f_in).replace(" ","")
    f_out.write(head)
    [f_out.write(','.join(line.split()) + '\n') for line in f_in if line[:8]!="Latitude"]
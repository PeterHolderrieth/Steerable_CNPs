import sys
import pandas as pd

filename_old=sys.argv[1]
filename_new=sys.argv[2]

data=pd.read_csv(filename_old,delimiter=",")

#Permute columns:
cols=data.columns.to_list()
cols_perm=cols[:2]+cols[3:]+[cols[2]]
data=data[cols_perm]


#Get a list of data frames per variable:
#List of shortnames:
shortnames=["z","sp","2t","10u","10v","100u","100v"]
data_list_single_var=[]
for var in shortnames:
    #Extract data for certain variable and reset index:
    data_single_var=data[data.shortName==var].reset_index(drop=True)
    #Rename the "value" column to the name of the variable:
    data_single_var.rename(columns={'Value': var },inplace=True)
    #Drop the short name:
    data_single_var.drop(columns=["shortName"],inplace=True) 
    #Append to the list:
    data_list_single_var.append(data_single_var)

#Merge data:
merged_data=data_list_single_var[0]
for i in range(1,len(shortnames)):
    merged_data=pd.merge(merged_data,data_list_single_var[i])

#Control that merged data has the correct number of rows:
n_rows_merged_data=len(merged_data.index)
n_rows_data=len(data.index)
n_control=int(n_rows_data/len(shortnames))

if n_control!=n_rows_merged_data:
    sys.exit("Error when processing data: Numbers of rows of unprocessed and processed do not fit.")
    print("Filename old: ", filename_old)

#Save the file:
merged_data.to_csv(filename_new,index=False)

#Control with reloaded data:
reloaded_data=pd.read_csv(filename_new,delimiter=",")
EPS=1e-5
diff_sum=(merged_data-reloaded_data).abs().sum().sum()
if diff_sum>1e-4:
    sys.exit("Error when reloading file: it seems that the sum of differences is fairly large.")

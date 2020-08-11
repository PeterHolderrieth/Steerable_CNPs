import pandas as pd
import sys 
import numpy as np

#pd.options.display.max_columns = None
#pd.options.display.max_rows = None

filename_to_split=sys.argv[1]
fileloc_new=sys.argv[2]
filename_without_time=sys.argv[3]

LAT_LOW=30.
LAT_HIGH=40.
STEP_LAT=0.25
LONG_WEST=-96.
LONG_EAST=-86.
STEP_LONG=0.25
VAR_LIST=['wind_10m_east','wind_10m_north']

def get_grid(min_x1,max_x1,step_x1,min_x2,max_x2,step_x2):
    X_1=np.arange(min_x1,max_x1+step_x1/2,step_x1)
    X_2=np.arange(min_x2,max_x2+step_x2/2,step_x2)
    n_x2=X_2.shape
    n_x1=X_1.shape
    X_1=X_1.repeat(n_x2)
    X_2=np.tile(X_2,n_x1)
    return(np.stack([X_1,X_2],axis=1))


#Load data frame:
df=pd.read_pickle(filename_to_split)

#Swap Longitude and Latitutde columns and east and north components such that x1 corresponds to longitutde and x2 to latitude:
col_list=list(df)
col_list=[col_list[0]]+[col_list[2]]+[col_list[1]]+col_list[3:6]+[col_list[7]]+[col_list[6]]+[col_list[9]]+[col_list[8]]
df=df.reindex(columns=col_list).reset_index(drop=True)
#Sort:
df.sort_values(by=['datetime','Longitude','Latitude'], axis=0, inplace=True)

#Set the index to be datetime and don't drop:
df.set_index(keys=['datetime'], drop=False,inplace=True)
#Get all times:
datetimes=df['datetime'].unique().tolist()

#time=pd.to_datetime(datetimes[0])
#df_single_time=df.loc[df['datetime']==time].copy()
#df_single_time.drop(columns=['datetime'],inplace=True)
#df_single_time[['Longitude','Latitude']].to_pickle(fileloc_new+"Grid_df.pickle")


#Get the control 
grid_df_control=pd.read_pickle(fileloc_new+"Grid_df.pickle")
X_control=grid_df_control.to_numpy()

for time_int in datetimes:
    time=pd.to_datetime(time_int)
    df_single_time=df.loc[df['datetime']==time].copy()
    df_single_time.drop(columns=['datetime'],inplace=True)
    time_str=time.strftime(format=("%Y_%m_%d_%H"))
    filename=time_str+filename_without_time
    X=df_single_time[['Longitude','Latitude']].to_numpy()
    diff_latitude=(X!=X_control).any()
    if diff_latitude:
        print("Last Example: ", time_str)
        sys.exit("ERROR: The latitude is not the same.")
    else:
        #Choose variables:
        df_single_time=df_single_time[VAR_LIST].reset_index(drop=True)
        df_single_time.to_pickle(fileloc_new+time_str+filename_without_time)
print("Finished one file.")

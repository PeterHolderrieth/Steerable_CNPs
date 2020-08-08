import pandas as pd
import sys 
#pd.options.display.max_columns = None
#pd.options.display.max_rows = None

filename_to_split=sys.argv[1]
fileloc_new=sys.argv[2]
filename_without_time=sys.argv[3]

#Load data frame:
df=pd.read_pickle(filename_to_split)
# sort the dataframe
df.sort_values(by=['datetime','Latitude','Longitude'], axis=0, inplace=True)
# set the index to be this and don't drop
df.set_index(keys=['datetime'], drop=False,inplace=True)
#Get all times:
datetimes=df['datetime'].unique().tolist()

#Extract latitude and longitude:
df_control=df.loc[df['datetime']==pd.to_datetime(datetimes[0])]
X_control=df_control[['Latitude','Longitude']].to_numpy()

for time_int in datetimes:
    time=pd.to_datetime(time_int)
    df_single_time=df.loc[df['datetime']==time].copy()
    df_single_time.drop(columns=['datetime'],inplace=True)
    time_str=time.strftime(format=("%Y_%m_%d_%H"))
    filename=time_str+filename_without_time
    X=df_single_time[['Latitude','Longitude']].to_numpy()
    diff_latitude=(X!=X_control).any()
    if diff_latitude:
        print("Last Example: ", time_str)
        sys.exit("ERROR: The latitude is not the same.")
    else:
        df_single_time.drop(columns=['Latitude','Longitude'],inplace=True)
        df_single_time.to_pickle(fileloc_new+time_str+filename_without_time)
# get a list of names
#names=df['datetime'].unique().tolist()
# now we can perform a lookup on a 'view' of the dataframe
#joe = df.loc[df.name=='joe']

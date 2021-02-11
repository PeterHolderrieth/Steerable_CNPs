#Bash command file to process GRIB file data:


MIN=1980
MAX=2018
FILEnoYEAR="_ERA5_China"

for (( c=$MIN; c<=$MAX; c++ ))
do  
    echo "Year: $c"
    ls -l --block-size=M data/"$c"${FILEnoYEAR}.grib

    #Create an unformatted CSV file:
    grib_get_data -p dataDate,dataTime,validityDate,validityTime,shortName data/"$c"${FILEnoYEAR}.grib > data/"$c"${FILEnoYEAR}_unformatted.csv
    echo "Saved unformatted csv file"

    ls -l --block-size=M data/"$c"${FILEnoYEAR}_unformatted.csv

    #Format the csv file to a proper table giving per row one measurement:
    python ../pre_processing/build_value_table_era5.py data/"$c"${FILEnoYEAR}_unformatted.csv data/"$c"${FILEnoYEAR}_per_measurement.csv
    echo "Formatted CSV file to proper table."

    #Remove unformatted csv file:
    rm data/"$c"${FILEnoYEAR}_unformatted.csv
    echo "Removed unformatted csv file"

    ls -l --block-size=M data/"$c"${FILEnoYEAR}_per_measurement.csv

    #Reshape the data table from "one row per measurement" to "one row per time point"
    python ../pre_processing/build_data_table_era5.py data/"$c"${FILEnoYEAR}_per_measurement.csv data/"$c"${FILEnoYEAR}_per_time.csv

    #Remove per measurement file:
    rm data/"$c"${FILEnoYEAR}_per_measurement.csv
    echo "Removed per_measurement file"

    ls -l --block-size=M data/"$c"${FILEnoYEAR}_per_time.csv

    #Compress and process the data table to a pickle file:
    python ../pre_processing/compress_data_table_era5.py data/"$c"${FILEnoYEAR}_per_time.csv data/"$c"${FILEnoYEAR}.pickle

    #Remove uncompressed file:
    rm data/"$c"${FILEnoYEAR}_per_time.csv
    echo "Removed per_time file"

    ls -l --block-size=M data/"$c"${FILEnoYEAR}.pickle

done

echo "Process completed"


Julia Signell
2016-01-19

*********************** DO NOT ALTER THESE FILES ***************************

This folder contains the Version1 files for Charlotte. These files differ from
the raw in that they are mildly processed so that they all have the same traits. 

- named: City_<radar code>_<gage or radar>_YEAR.csv
- comma separated
- column order matches City_<radar code>_lat_lon_YEAR.csv file order
- all timesteps for the year are included in file (for gage)
- units are mm accumulated at the end of each time step
- yearly files include all of year (not off by 6 hours or anything)
- timezone is UTC
- missing values in datafiles are filled with -99
- missing values in City_<radar code>_lat_lon_YEAR.csv files are filled with -999, -999

If you have qualms about datapoints and want to change values to -99, do this 
in the Version2 files. 

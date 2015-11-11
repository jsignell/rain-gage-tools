from raingage import *

class RadarGage:
    '''
    Base class for RainGage files
    '''
    def __init__(self, df_file=None, path='.', year=None, name="Philadelphia_raingage_{YEAR}_NAN_filled.dat",
                 freq='15min', per_hour=4, ngages=24, units='mm', lat_lon_file="RG_lon_lat.txt"):
        '''
        If you choose to give a year, you still need to give a path to the
        parent directory and a general name for the files. 
        '''
        self.freq = freq
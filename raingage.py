from math import cos, pi
import string
import posixpath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def choose_group(df, interval, gage=None, m=None, h=None):
    if gage is not None and type(gage) is not list and type(gage) is not tuple:
        gage = [gage]
    if m in range(1,13,1):
        interval='seasonal'
    elif h in range(0,24,1):
        interval='diurnal'
    if interval is 'seasonal':
        if gage is None:
            group = df.mean(axis=1).groupby(df.index.month)
        elif set(gage) <= set(df.columns):
            group = df[gage].groupby(df[gage].index.month)
    elif interval is 'diurnal':
        if gage is None:
            group = df.mean(axis=1).groupby(df.index.hour)
        elif set(gage) <= set(df.columns):
            group = df[gage].groupby(df[gage].index.hour)

    if m is None and h is None:
        return group
    elif m in range(1,13,1):
        for name, df in group:
            if name == m:
                group1 = df.groupby(df.index.hour)
    elif h in range(0,24,1):
        for name, df in group:
            if name == h:
                group1 = df.groupby(df.index.month) 
    return group1


class RainGage:
    '''
    Base class for RainGage files
    '''
    def __init__(self, df_file=None, path='.', year=None, name="Philadelphia_raingage_{YEAR}_NAN_filled.dat"):
        '''
        If you choose to give a year, you still need to give a path to the
        parent directory and a general name for the files. 
        '''
        self.freq = '15min'
        self.per_hour = 4
        self.ngages = 24
        self.path = path.replace('\\','/')
        if not path.endswith('/'):
            self.path = self.path + '/'
        
        if df_file is not None:
            self.name = df_file
            self.df = pd.read_csv(self.path+self.name, 
                                  delim_whitespace=True, na_values='-99', 
                                  index_col=0, parse_dates=True)
            self.year = '{first_year} to {last_year}'.format(first_year=self.df.index[0].year,
                                                             last_year=self.df.index[-1].year)
        elif year is None:
            self.name = name
            self.year = [int(s) for s in self.name.split('_') if s.isdigit()][0]
            self.get_df()
        elif type(year) is int:
            self.name = name.format(YEAR=year)
            self.year = year
            self.get_df()
        elif type(year) is list or type(year) is tuple:
            self.get_files(year, name)
            self.year = '{first_year} to {last_year}'.format(first_year=year[0],
                                                             last_year=year[-1])
        else:
            print "Error: incorrect init of RainGage"

        self.rate = self.df*self.per_hour
        self.reset_thresh()
        self.get_wet()

    def get_files(self, year, name):
        f = [name.format(YEAR=y) for y in year]
        self.name = f
        df_list = []
        for f in self.name:
            self.name = f
            self.get_df()
            df_list.append(self.df)    
        self.df = pd.concat(df_list)
    
    def show(self):
        print self.path
        print self.name
        print self.year
        print self.freq
        print self.per_hour
    
    def get_RG_lon_lat(self, filename="RG_lon_lat.txt"):
        RG_lon_lat = pd.read_csv(self.path+filename, delim_whitespace=True, 
                                 header=None, names=['RG', 'lon', 'lat'])
        RG_lon_lat.index.name = 'order_in_file'
        RG_lon_lat['RG'] = 'RG' + RG_lon_lat['RG'].apply(str)
        RG_lon_lat.index = RG_lon_lat['RG']
        RG_lon_lat['Y'] = RG_lon_lat['lat']*110.574
        RG_lon_lat['X'] = RG_lon_lat['lon']*111.320*(RG_lon_lat['lat']*pi/180).apply(cos)
        RG_lon_lat['X'] = RG_lon_lat['X'] - min(RG_lon_lat['X'])
        RG_lon_lat['Y'] = RG_lon_lat['Y'] - min(RG_lon_lat['Y'])
        self.RG_lon_lat = RG_lon_lat
    
    def get_df(self): 
        dates = ['year','month','day','hour','minute']
        names = dates[:]
        RG = ['RG' + str(i) for i in range(1,self.ngages+1,1)]
        [names.append(i) for i in RG]
        
        dateparse = lambda x: pd.datetime.strptime(x, '%Y %m %d %H %M')
        self.df = pd.read_csv(self.path+self.name, delim_whitespace=True, header=None, names=names,
                              na_values = '-99',
                              parse_dates={'date_time': dates},
                              date_parser=dateparse, index_col=[0])
    
    def reset_thresh(self): 
        self.thresh = min([i for i in self.df[self.df.columns[0]] if i > 0])
        
    def get_wet(self):
        self.wet = self.df >= self.thresh
    
    def get_wettest(self, time_step='15min', path='SVG_data'):
        df = self.df.resample(time_step, how='sum',label='right', closed='right').dropna(how='all')
        
        if not hasattr(self,'thresh'):
            self.get_thresh()

        wet = df >= self.thresh
        df = df.drop(wet[wet.sum(axis=1) != 24].index)
        wettest = df.sum(axis=1).sort_values()

        list_of_series = []
        for t in wettest.tail(5).index:
            list_of_series.append(df.loc[t])
        
        if not hasattr(self,'RG_lon_lat'):
            self.get_RG_lon_lat()
 
        self.wettest = self.RG_lon_lat.join(list_of_series)
        self.wettest.to_csv(path, index=False)

    def get_storm(self, storm_day='2013-08-13', time_step='15min', path='SVG_data'):
        df = self.df.resample(time_step, how='sum',label='right', closed='right').dropna(how='all')

        if not hasattr(self,'thresh'):
            self.get_thresh()

        wet = df >= self.thresh
        df = df.drop(wet[wet.sum(axis=1) != 8].index)
        storm = df[storm_day].transpose()

        if not hasattr(self,'RG_lon_lat'):
            self.get_RG_lon_lat()

        self.storm = self.RG_lon_lat.join(storm)
        for col in self.storm:
            if col not in self.RG_lon_lat.columns:
                storm[col] = self.storm[col].replace(0, np.nan)
        storm.to_csv(path, index=False)

    def create_title(self, title, time_step='15min', interval='seasonal',
                     gage=None, month=None, hour=None):
        if gage is not None:
            title = '{g}: '.format(g=gage)+title
        if month is not None:
            title = title + ' for Month {m} of'.format(m=month)
        elif hour is not None:
            title = title + ' for Hour {h} of'.format(h=hour) 
        elif interval is 'seasonal':
            title = title + ' for Months of'
        elif interval is 'diurnal':
            title = title + ' for Hours of'
        full_title = (title +' {year}').format(ts=time_step, year=self.year)
        self.title = full_title
        
    def get_prob_wet(self, time_step='15min', interval='seasonal', show_all=False,
                     gage=None, month=None, hour=None, look_closer=None, lc=None):
        fig = plt.figure(figsize=(16,6))
        ax = fig.add_subplot(111)
        self.wet = self.wet.resample(time_step, how='sum',label='right', closed='right')>=1

        if show_all is False:
            self.group = choose_group(self.wet, interval, gage, month, hour)
            if look_closer is None:
                self.group.mean().plot(kind='bar', ax=ax)
            elif look_closer is not None:
                self.group.mean()[look_closer].plot(kind='bar', ax=ax)
            elif lc is not None:
                self.group.mean()[look_closer].loc[lc].plot(kind='bar', ax=ax)

        else:
            if gage is None or len(gage) > 1:
                gage = None
                show_all = self.wet.mean(axis=1).groupby([self.wet.index.hour, self.wet.index.month]).mean().unstack()
            elif len(gage) is 1:
                show_all = self.wet[gage].groupby([self.wet.index.hour, self.wet.index.month]).mean().unstack() 
            show_all.index.name='hours'
            show_all.columns.name='months'
            self.show_all = show_all
            if month is None:
                show_all.plot(kind='bar', ax=ax)
            else:
                if gage is None:
                    if hour is None:
                        show_all[month].plot(kind='bar', ax=ax)
                    else:
                        show_all[month].loc[hour].plot(kind='bar', ax=ax)
                else:
                    sliceit = zip(gage*len(month), month)
                    if hour is None:
                        show_all[sliceit].plot(kind='bar', ax=ax)
                    else:
                        show_all[sliceit].loc[hour].plot(kind='bar', ax=ax)

        ax.set_ylabel("Probability of wet {ts}".format(ts=time_step))
        self.create_title('Probability of wet {ts}',time_step, interval, gage, month, hour)
        plt.title(self.title)

    def get_boxplots(self, time_step='15min', interval='seasonal', 
                     gage=None, month=None, hour=None, look_closer=None):
        self.group = choose_group(self.df, interval, gage, month, hour)
        wet_rates = []
        labels = []
        for name, df in self.group:
            if look_closer is not None:
                if name not in look_closer:
                    continue
            elif interval is 'diurnal':
                if name%2 == 1:  # if the hour is odd
                    continue
            df = df.resample(time_step, how='mean', label='right', closed='right')
            df = df[df.T >= self.thresh]  # only keep wet days
            df = df * self.per_hour  # Make it a rain rate
            wet_rates.append(df.values)
            labels.append(name)
        fig = plt.figure(figsize=(len(wet_rates)*4/3,4))
        plt.boxplot(wet_rates, sym='', whis=[10,90], meanline=True, labels=labels)
        plt.yscale('linear')
        plt.ylabel('Rain Rate (mm/hr)')
        title = '{ts} Rain Rate Distribution (excluding dry {ts})'
        self.create_title(title, time_step, interval, gage, month, hour)
        plt.title(self.title)

    def get_distribution(self, time_step='15min', interval='seasonal', 
                         gage=None, month=None, hour=None, look_closer=None):
        self.group = choose_group(self.df, interval, gage, month, hour)
        labels = []
        foo = []
        for name, df in self.group:
            if look_closer is not None:
                if name not in look_closer:
                    continue
            else:
                if name%4 != 0:  # if the hour isn't divisible by 4
                    continue
            df = df.resample(time_step, how='mean',label='right',closed='right')
            df = df[df.T >= self.thresh]  # only keep wet days
            df = df * self.per_hour  # Make it a rain rate
            foo.append(df.quantile(np.arange(0,1.001,.001)))
            labels.append(name)
        quan = pd.DataFrame(foo).transpose()
        quan.columns = labels
        self.quantiles = quan
        
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(211)
        self.quantiles.plot(logy=True,ax=ax)
        ax.set_ylabel('Rain Rate (mm/hr)') 
        title = '{ts} Rain Rate Distribution (excluding dry {ts})'
        self.create_title(title, time_step, interval, gage, month, hour)
        plt.title(self.title)
        
        ax1 = fig.add_subplot(212)
        self.quantiles.plot(xlim=(0.9,1),ax=ax1, legend=None)
        ax1.set_ylabel('Rain Rate (mm/hr)') 
        
        
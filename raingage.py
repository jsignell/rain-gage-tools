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

        self.reset_thresh()

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
    
    def get_RG_lon_lat(self):
        RG_lon_lat = pd.read_csv(self.path+"RG_lon_lat.txt", delim_whitespace=True, 
                                 header=None, names=['RG', 'lon', 'lat'])
        RG_lon_lat.index.name = 'order_in_file'
        RG_lon_lat['RG'] = 'RG' + RG_lon_lat['RG'].apply(str)
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
    
    def create_title(self, title, interval='seasonal',
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
        full_title = (title +' {year}').format(ts='15min', year=self.year)
        self.title = full_title
        
    def get_prob_wet(self, interval='seasonal', show_all=False,
                     gage=None, month=None, hour=None, look_closer=None, lc=None):
        fig = plt.figure(figsize=(16,6))
        ax = fig.add_subplot(111)

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

        ax.set_ylabel("Probability of wet 15min")
        self.create_title('Probability of wet 15min', interval, gage, month, hour)
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
            df = df.resample('15min')
            df = df[df.T >= self.thresh]  # only keep wet days
            df = df * self.per_hour  # Make it a rain rate
            wet_rates.append(df.values)
            labels.append(name)
        fig = plt.figure(figsize=(len(wet_rates)*4/3,4))
        plt.boxplot(wet_rates, sym='', whis=[10,90], meanline=True, labels=labels)
        plt.yscale('linear')
        plt.ylabel('Rain Rate (mm/hr)')
        title = '{ts} Rain Rate Distribution (excluding dry {ts})'
        self.create_title(title, interval, gage, month, hour)
        plt.title(self.title)

    def get_distribution(self, time_step='15min', interval='seasonal', 
                         gage=None, month=None, hour=None, look_closer=None):
        self.group = choose_group(self.df, interval, gage, month, hour)
        wet_rates = []
        labels = []
        foo = []
        for name, df in self.group:
            if look_closer is not None:
                if name not in look_closer:
                    continue
            elif interval is 'diurnal':
                if name%2 == 1:  # if the hour is odd
                    continue
            df = df.resample('15min')
            df = df[df.T >= self.thresh]  # only keep wet days
            df = df * self.per_hour  # Make it a rain rate
            foo.append(df.quantile(np.arange(0,1.001,.001)))
            labels.append(name)
        quan = pd.DataFrame(foo).transpose()
        quan.columns = labels
        self.quantiles = quan
        
        fig = plt.figure(figsize=(len(wet_rates)*4/3,4))
        ax = fig.add_subplot(111)
        self.quantiles.plot(logy=True,ax=ax, legend=None)
        ax.set_ylabel('Rain Rate (mm/hr)') 
        title = '{ts} Rain Rate Distribution (excluding dry {ts})'
        self.create_title(title, interval, gage, month, hour)
        plt.title(self.title)
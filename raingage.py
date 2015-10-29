import string
import posixpath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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

        self.df.months = self.df.groupby(self.df.index.month)
        self.months = self.df.months.groups.keys()
        self.get_gage_mean()
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
    
    def get_gage_mean(self):
        self.gage_mean = self.df.mean(axis=1)
        self.gage_mean.months = self.gage_mean.groupby(self.gage_mean.index.month)
        self.gage_mean.hours = self.gage_mean.groupby(self.gage_mean.index.hour)
    
    def reset_thresh(self): 
        self.thresh = min([i for i in self.df[self.df.columns[0]] if i > 0])
        
    def get_wet(self):
        self.wet = self.gage_mean >= self.thresh
        self.wet.months = self.wet.groupby(self.wet.index.month)
        self.wet.hours = self.wet.groupby(self.wet.index.hour)
        
    def get_prob_wet(self, interval='seasonal'):
        fig = plt.figure(figsize=(16,6))
        ax = fig.add_subplot(111)
        
        if interval is 'seasonal':
            self.wet.months.mean().plot(kind='bar',legend=None, ax=ax)
            ax.set_xlabel("Months of {year}".format(year=self.year))

        elif interval is 'diurnal':
            self.wet.hours.mean().plot(kind='bar',legend=None, ax=ax)
            ax.set_xlabel("Hours of {year}".format(year=self.year))
        
        elif interval is 'all' or interval in self.months:
            bar = []
            monthly = [df[1] for df in self.wet.months]
            for i, df in enumerate(monthly):
                bar.append(df.groupby(df.index.hour).mean())
            wet_hours_months = pd.DataFrame(bar).transpose()
            wet_hours_months.columns = self.months
            self.wet.hours.months = wet_hours_months
            if interval in self.months:
                self.wet.hours.months[interval].plot(kind='bar', ax=ax)
                ax.set_xlabel("Hours of {interval}-{year}".format(interval=interval, year=self.year))
            else:
                self.wet.hours.months.plot(kind='bar', ax=ax)
                ax.set_xlabel("Hours of {year}".format(year=self.year))

        ax.set_ylabel("Probability of wet 15min")
        if not hasattr(self, 'prob_wet'):
            self.prob_wet = {}
        self.prob_wet.update({interval: fig})

    def get_boxplots(self, df_list, fig, time_step='15min', interval='seasonal'):
        foo = []
        for i, df in enumerate(df_list):
            df = df.resample(time_step)
            df = df[df.T >= self.thresh]  # only keep wet days
            df = df * self.per_hour  # Make it a rain rate
            if interval is 'seasonal':
                df.name = df.index[0].month
            if interval is 'diurnal' or interval in self.months:
                df.name = df.index[0].hour
            quan = df.quantile(np.arange(0,1.01,.01))
            nrows = 1
            ncols = len(df_list)
            ax = fig.add_subplot(nrows, ncols, i+1)
            df.plot(kind='box', sym='', whis=[10,90], meanline=True, logy=True, sharey=True, ax=ax)
            if i == 0:
                ax.set_ylabel('Rain Rate (mm/hr)')
            foo.append(quan)
 
        if not hasattr(self, 'boxplots'):
            self.boxplots = {}
        if interval not in self.boxplots.keys():
            self.boxplots.update({interval: {time_step: fig}})
        self.boxplots[interval].update({time_step: fig})

        if not hasattr(self, 'quantiles'):
            self.quantiles = {}
        if interval not in self.quantiles.keys():
            self.quantiles.update({interval: {time_step: pd.DataFrame(foo).transpose()}})
        self.quantiles[interval].update({time_step: pd.DataFrame(foo).transpose()})

    def get_distribution(self, time_step='15min', interval='seasonal'):
        fig = plt.figure(figsize=(16,4))
        ax = fig.add_subplot(111)
        self.quantiles[interval][time_step].plot(logy=True,ax=ax, legend=None)
        ax.set_ylabel('Rain Rate (mm/hr)')

        if not hasattr(self, 'distribution'):
            self.distribution = {}
        if interval not in self.distribution.keys():
            self.distribution.update({interval: {time_step:fig}})
        self.distribution[interval].update({time_step: fig})
    
    def get_quantiles(self, time_step='15min', interval='seasonal'):
        '''
        interval={'seasonal', 'diurnal', int referring to specific month}
        
        '''
        fig = plt.figure(figsize=(16,4))

        if interval is 'seasonal':
            df_list = [df[1] for df in self.gage_mean.months]

        if interval is 'diurnal':
            df_list = [i[1] for i in self.gage_mean.hours if i[0]%2==0]
            
        if interval in self.months:
            df = [df[1] for df in self.gage_mean.months][interval-self.months[0]]
            df_list = [i[1] for i in df.groupby(df.index.hour) if i[0]%2==0]
            fig.suptitle('{time_step} Rain Rate Distribution (excluding dry {time_step}) {interval}-{year}'.format(
                         time_step=time_step, interval=interval, year=self.year), fontsize=20)
            
        if interval not in self.months:
            fig.suptitle('{time_step} Rain Rate Distribution (excluding dry {time_step}) for {year}'.format(
                         time_step=time_step, year=self.year), fontsize=20)
        self.get_boxplots(df_list, fig, time_step, interval=interval)
        self.get_distribution(time_step, interval)
    
    def plot_lots(self):
        self.get_wet()
        self.get_prob_wet('seasonal')
        self.get_prob_wet('diurnal')
        self.get_prob_wet('all')
        self.get_quantiles(interval='seasonal')
        self.get_quantiles(interval='seasonal', time_step='1H')
        self.get_quantiles(interval='seasonal', time_step='1D')
        self.get_quantiles(interval='diurnal')
        self.get_quantiles(interval='diurnal', time_step='1H')
 
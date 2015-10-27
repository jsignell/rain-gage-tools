import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class RainGage:
    '''
    Base class for RainGage files
    '''
    def __init__(self, path):
        self.path = path
        self.name = path.replace('\\','/').split('/')[-1]
        self.year = [int(s) for s in self.name.split('_') if s.isdigit()][0]
        self.freq = '15min'
        self.per_hour = 4
        self.get_df()
        self.get_gage_mean()
        self.reset_thresh()
    
    def get_df(self):
        dates = ['year','month','day','hour','minute']
        names = dates[:]
        __foo = [names.append(i) for i in list(string.uppercase[:24])]
        
        dateparse = lambda x: pd.datetime.strptime(x, '%Y %m %d %H %M')
        self.df = pd.read_csv(self.path, delim_whitespace=True, header=None, names=names,
                              na_values = '-99',
                              parse_dates={'date_time': dates},
                              date_parser=dateparse, index_col=[0])
        self.df.months = self.df.groupby(self.df.index.month)
        self.months = self.df.months.groups.keys()
    
    def get_gage_mean(self):
        self.gage_mean = self.df.mean(axis=1)
        self.gage_mean.months = self.gage_mean.groupby(self.gage_mean.index.month)
    
    def reset_thresh(self):
        self.thresh = min([i for i in self.df['A'] if i > 0])
        
    def get_wet(self):
        self.wet = self.gage_mean >= self.thresh
        self.wet.months = self.wet.groupby(self.wet.index.month)
        self.wet.hours = self.wet.groupby(self.wet.index.hour)
        
    def get_prob_wet(self, interval='seasonal'):
        fig = plt.figure(figsize=(16,6))
        ax = fig.add_subplot(111)
        
        if interval is 'seasonal':
            self.wet.months.mean().plot(kind='bar',legend=None, ax=ax)
            ax.set_xlabel("Months of "+ str(self.year))

        elif interval is 'diurnal':
            self.wet.hours.mean().plot(kind='bar',legend=None, ax=ax)
            ax.set_xlabel("Hours of "+ str(self.year))
        
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
                ax.set_xlabel("Hours of %i-%i"% (interval, self.year))
            else:
                self.wet.hours.months.plot(kind='bar', ax=ax)
                ax.set_xlabel("Hours of %i"% self.year)

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
            nrows = len(df_list)/6
            ncols = len(df_list)/nrows
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
        plt.gca().invert_xaxis()

        if not hasattr(self, 'distribution'):
            self.distribution = {}
        if interval not in self.distribution.keys():
            self.distribution.update({interval: {time_step:fig}})
        self.distribution[interval].update({time_step: fig})
    
    def get_quantiles(self, time_step='15min', interval='seasonal'):
        '''
        interval={'seasonal', 'diurnal', int referring to specific month}
        
        '''
        if interval is 'seasonal':
            df_list = [df[1] for df in self.gage_mean.months]
            fig = plt.figure(figsize=(16,4))
            
        if interval is 'diurnal':
            df_list = [i[1] for i in self.gage_mean.groupby(self.gage_mean.index.hour)]
            fig = plt.figure(figsize=(16,12))
            
        if interval in self.months:
            df = [df[1] for df in self.gage_mean.months][interval-self.months[0]]
            df_list = [i[1] for i in df.groupby(df.index.hour)]
            fig = plt.figure(figsize=(16,12))
            fig.suptitle('%s Rain Rate Distribution (excluding dry %s) %i-%i' %
                         (time_step, time_step, interval, self.year), fontsize=20)
            
        if interval not in self.months:
            fig.suptitle('%s Rain Rate Distribution (excluding dry %s) %i' %
                         (time_step, time_step, self.year), fontsize=20)
        self.get_boxplots(df_list, fig, time_step, interval=interval)
        self.get_distribution(time_step, interval)
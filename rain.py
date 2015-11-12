from math import cos, pi
import string
import posixpath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from common import *

class Rain:
    '''
    Base class for Rain files
    '''
    group_kwargs = dict(time_step=None, 
                    base=0, 
                    interval='seasonal',
                    gage=None, 
                    m=None, 
                    h=None)
    
    def __init__(self, df_file=None, path='.', year=None, name="Philadelphia_raingage_{YEAR}_NAN_filled.dat",
                 freq='15min', per_hour=4, ngages=24, units='mm', ll_file="RG_lon_lat.txt", save_path='.'):
        '''
        If you choose to give a year, you still need to give a path to the
        parent directory and a general name for the files. 
        '''
        self.freq = freq
        self.per_hour = per_hour
        self.ngages = ngages
        self.units = units
        self.ll_file = ll_file
        self.path = path.replace('\\','/')
        self.save_path = save_path.replace('\\','/')
        
        if not path.endswith('/'):
            self.path = self.path + '/'
        
        if df_file is not None:
            self.name = df_file
            self.df = pd.read_csv(self.path+self.name, 
                                  delim_whitespace=True, na_values='-99', 
                                  index_col=0, parse_dates=True)
            self.df.columns.name = 'RG'
            self.year = '{first_year}-{last_year}'.format(first_year=self.df.index[0].year,
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
            self.year = '{first_year}-{last_year}'.format(first_year=year[0],
                                                             last_year=year[-1])
        else:
            print "Error: incorrect init of RainGage"

        if self.units.startswith('in'):
            self.df = self.df * 25.4
            self.units = 'mm'
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
        print self.ngages
        print self.units
        print self.thresh
        print self.ll_file
    
    def get_ll(self, cols=['RG', 'lon', 'lat'], path=None, ll_file=None):
        if ll_file is None:
            ll_file = self.ll_file
        if path is None:
            path = self.path
        ll = pd.read_csv(path+ll_file, delim_whitespace=True, 
                                 header=None, names=cols)
        if 'RG' not in cols:
            ll['RG'] = ll.index + 1
        ll['RG'] = 'RG' + ll['RG'].apply(str)
        ll.index = ll['RG']
            
        ll['Y'] = ll['lat']*110.574
        ll['X'] = ll['lon']*111.320*(ll['lat']*pi/180).apply(cos)
        ll['X'] = ll['X'] - min(ll['X'])
        ll['Y'] = ll['Y'] - min(ll['Y'])
        self.ll = ll
    
    def plot_ll(self, save_path='./output/StLouis/'):
        df = self.ll[self.ll.lat!=0]
        x_dist=[]
        y_dist=[]
        for i, val in enumerate(df.X):
            a = (val-df.X[(i+1):]).reshape(-1)
            x_dist = np.concatenate((x_dist, a))
        for i, val in enumerate(df.Y):
            a = (val-df.Y[(i+1):]).reshape(-1)
            y_dist = np.concatenate((y_dist, a))

        title = 'Distances between gages (angle of {angle} from horizontal)'

        plt.figure(figsize=(10,8))
        plt.scatter(x_dist,y_dist, s=5)
        plt.ylabel('Y distance (km)')
        plt.xlabel('X distance (km)')

        # determine best fit line
        par = np.polyfit(x_dist, y_dist, 1, full=True)

        slope=par[0][0]
        intercept=par[0][1]
        xl = [min(x_dist), max(x_dist)]
        yl = [slope*xx + intercept for xx in xl]
        plt.plot(xl, yl, '-r')

        plt.title(title.format(angle=round(slope*90)))
        plt.savefig(self.save_path+'{title}.jpg'.format(title=title).format(angle=round(slope*90)))
    
    def get_df(self): 
        dates = ['year','month','day','hour','minute']
        names = dates[:]
        RG = ['RG' + str(i) for i in range(1,self.ngages+1,1)]
        [names.append(i) for i in RG]
        
        dateparse = lambda x: pd.datetime.strptime(x, '%Y %m %d %H %M')
        self.df = pd.read_csv(self.path+self.name, header=None, names=names,
                              delim_whitespace=True, na_values = '-99',
                              parse_dates={'date_time': dates},
                              date_parser=dateparse, index_col=[0])
        self.df.columns.name = 'RG'
    
    def plot_rate(self, time_step=None, base=0, interval='seasonal',
                  gage=None, m=None, h=None):
        if time_step is None:
            time_step = self.freq
        kwargs = dict(time_step=time_step, base=base, interval=interval, gage=gage, m=m, h=h)
        self.gb = choose_group(self.rate, **kwargs)
        if interval is 'seasonal' and m is not None:
            if type(m) is not list and type(m) is not tuple:
                m = [m]
            df = self.gb.mean().loc[m]
        elif interval is 'diurnal' and h is not None:
            if type(h) is not list and type(h) is not tuple:
                h = [h]
            df = self.gb.mean().loc[h]
        else:
            df = self.gb.mean()
        title = create_title('{ts} Rain Rate', self.year, **kwargs)
        df.plot(kind='bar', figsize=(16,6), title=title)
        plt.ylabel('Mean Rain Rate (mm/hr)')
        plt.savefig(self.save_path+title+'.jpg')
            
    def reset_thresh(self): 
        self.thresh = min([i for i in self.df[self.df.columns[0]] if i > 0])

    def get_wet(self):
        self.wet = self.rate >= self.thresh * self.per_hour
        
    def plot_prob_wet(self, time_step=None, interval='seasonal', base=0,
                     gage=None, m=None, h=None):
        if time_step is None:
            time_step = self.freq
        fig = plt.figure(figsize=(16,6))
        ax = fig.add_subplot(111)
        
        kwargs = dict(interval=interval, gage=gage, m=m, h=h)
        d, date_time = get_index(self.wet, 'date_time')
        
        self.wet = self.wet.resample(time_step, axis=d, how='sum',label='right', closed='right')>=1
        self.gb = choose_group(self.rate, **kwargs)
        if interval is 'seasonal' and m is not None:
            if type(m) is not list and type(m) is not tuple:
                m = [m]
            df = self.gb.mean().loc[m]
        elif interval is 'diurnal' and h is not None:
            if type(h) is not list and type(h) is not tuple:
                h = [h]
            df = self.gb.mean().loc[h]
        else:
            df = self.gb.mean()
        df.plot(kind='bar', ax=ax)

        ax.set_ylabel("Probability of wet {ts}".format(ts=time_step))
        title = create_title('Probability of wet {ts}',self.year, time_step=time_step, **kwargs)
        plt.title(title)
        plt.savefig(self.save_path+title+'.jpg')

    def plot_boxplots(self, time_step=None, base=0, interval='seasonal', gage=None, m=None, h=None):
        if time_step is None:
            time_step = self.freq
        kwargs = dict(time_step=time_step, base=base, interval=interval, gage=gage, m=m, h=h)
        self.group = choose_group(self.rate, **kwargs)
        wet_rates = {}
        for name, df in self.group:
            if interval is 'seasonal' and m is not None:
                if name not in m:
                    continue
            if interval is 'diurnal' and h is not None:
                if name not in h:
                    continue
            elif interval is 'diurnal':
                if name%2 == 1:  # if the hour is odd
                    continue
            df = df[df >= self.thresh].dropna()  # only keep wet days
            wet_rates.update({name:df.values})
        fig = plt.figure(figsize=(len(wet_rates)*4/3,4))
        plt.boxplot(wet_rates.values(), sym='', whis=[10,90], meanline=True, labels=wet_rates.keys())
        plt.yscale('linear')
        plt.ylabel('Rain Rate (mm/hr)')
        title = '{ts} Rain Rate Distribution (excluding dry {ts})'
        title = create_title(title, self.year, **kwargs)
        plt.title(title)
        plt.savefig(self.save_path+title+'.jpg')

    def plot_distribution(self, time_step=None, base=0, interval='seasonal', 
                         gage=None, m=None, h=None, look_closer=None):
        if time_step is None:
            time_step = self.freq
        kwargs = dict(time_step=time_step, base=base, interval=interval, gage=gage, m=m, h=h)
        gb = choose_group(self.df,**kwargs)
        labels = []
        foo = []
        for name, df in gb:
            if interval is 'seasonal' and m is not None:
                if name not in m:
                    continue
            if interval is 'diurnal' and h is not None:
                if name not in h:
                    continue
            else:
                if name%4 != 0:  # if the hour isn't divisible by 4
                    continue
            df = df[df >= self.thresh*self.per_hour].dropna()  # only keep wet days
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
        title = create_title(title, self.year, **kwargs)
        plt.title(title)
        plt.savefig(self.save_path+title+'.jpg')
        
        ax1 = fig.add_subplot(212)
        self.quantiles.plot(xlim=(0.9,1),ax=ax1, legend=None)
        ax1.set_ylabel('Rain Rate (mm/hr)') 
              
    def get_wettest(self, time_step=None, path='SVG_data'):
        if time_step is None:
            time_step = self.freq

        df = self.df.resample(time_step, how='sum',label='right', closed='right').dropna(how='all')
        
        if not hasattr(self,'thresh'):
            self.get_thresh()

        wet = df >= self.thresh
        df = df.drop(wet[wet.sum(axis=1) != self.ngages].index)
        wettest = df.sum(axis=1).sort_values()

        list_of_series = []
        for t in wettest.tail(5).index:
            list_of_series.append(df.loc[t])
        
        if not hasattr(self,'ll'):
            self.get_ll()
 
        self.wettest = self.ll.join(list_of_series)
        self.wettest.to_csv(path, index=False)
    
    def get_rainiest_days(self, n):
        if not hasattr(self,'ll'):
            self.get_ll()
        
        rainiest = self.rate.resample('24H', base=12, **get_resample_kwargs(self.rate))
        
        largest = rainiest.mean(axis=get_index(self.rate, 'RG')[0]).sort_values().dropna().tail(n)
        r = self.ll.join(rainiest.loc[largest.index].transpose())
        self.rainiest = r[r.lat != 0] 
    
    def get_storm(self, storm_day='2013-08-13', time_step=None, path='SVG_data'):
        if time_step is None:
            time_step = self.freq
        df = self.df[storm_day].resample(time_step, how='sum',label='right', closed='right').dropna(how='all')

        if not hasattr(self,'thresh'):
            self.get_thresh()

        wet = df >= self.thresh
        df = df.drop(wet[wet.sum(axis=1) <= 18].index) # nCr for 18C2 produces just 5 bins of 30, so this is the fewest for good results
        storm = df.transpose()

        if not hasattr(self,'ll'):
            self.get_ll()

        storm = self.ll.join(storm)
        storm = storm.drop(storm[storm.lat == 0].index)
        self.storm = storm[:]
        for col in storm:
            if col not in self.ll.columns:
                storm[col] = storm[col].replace(0, np.nan)
        storm.to_csv(path, index=False)


class RadarGage(Rain):
    
    def __init__(self, radar, gage):
        self.rate = pd.Panel({'gage': gage.rate, 'radar': radar.rate})
        self.freq = gage.freq
        self.year = gage.year
        self.thresh = gage.thresh
        self.path = gage.path
        self.ll_file = gage.ll_file
        self.ll = gage.ll
        self.gage = gage
        self.radar = radar
    
    def plot_correlation(self, time_step=None, base=0):
        if time_step is None:
            time_step = self.freq
        p = self.rate.resample(time_step, base=base, **get_resample_kwargs(self.rate))
        
        title = create_title('{ts} Radar Rain Gage Correlation', year=self.year, time_step=time_step)
        plt.figure(figsize=(8,8))
        plt.scatter(x=p.gage, y=p.radar, s=5)
        plt.ylabel('radar')
        plt.xlabel('gage')
        m = min(p.max().max())
        plt.xlim(0, m)
        plt.ylim(0, m)
        plt.title(title)
        plt.savefig(self.save_path+title+'.jpg')
        
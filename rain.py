from __init__ import *
from common import *
from event import *

class Rain:
    '''
    Base class for Rain files
    '''  
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
            print "Error: incorrect init of Rain"

        if self.units.startswith('in'):
            self.df = self.df * 25.4
            self.units = 'mm'
        self.rate = self.df*self.per_hour
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
        print self.ngages
        print self.units
        print self.thresh
        print self.ll_file
    
    def list_gages(self):
        return list(get_index(self.rate, 'RG')[1])
    
    def get_ll(self, cols=['RG', 'lon', 'lat'], path=None, ll_file=None):
        if ll_file is None:
            ll_file = self.ll_file
        if path is None:
            path = self.path
        ll = pd.read_csv(path+ll_file, sep=',', header=None, names=cols) 
        if 'RG' not in cols:
            ll['RG'] = ll.index + 1
        ll['RG'] = 'RG' + ll['RG'].apply(str)
        ll.index = ll['RG']
            
        ll['Y'] = ll['lat']*110.574
        ll['X'] = ll['lon']*111.320*(ll['lat']*pi/180).apply(cos)
        ll['X'] = ll['X'] - min(ll['X'])
        ll['Y'] = ll['Y'] - min(ll['Y'])
        self.ll = ll
    
    def plot_ll(self, title=None, save=True):
        df = self.ll[self.ll.lat > -200]
        if hasattr(self, 'df_corr'):
            df_corr = self.df_corr.loc[df.index, df.index]
        else:
            df_corr = None
        x_dist=[]
        y_dist=[]
        dist_corr = []
        for i, val in enumerate(df.X[:-1]):
            a = (val-df.X[(i+1):]).reshape(-1)
            x_dist = np.concatenate((x_dist, a))
        for i, val in enumerate(df.Y[:-1]):
            a = (val-df.Y[(i+1):]).reshape(-1)
            y_dist = np.concatenate((y_dist, a))
            if df_corr is not None:
                c = (df_corr.loc[df.index[i], df.index[i+1]:df.index[-1]]).reshape(-1)
                dist_corr = np.concatenate((dist_corr, c))
        if title is None:
            title = 'Distances between gages ({angle} from horizontal)'
        
        if df_corr is None:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
            axes = [axes]
            scat = axes[0].scatter(x_dist,y_dist, s=5)
        else:
            title = title + ' with color representing corr coef'
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,12))
            scat = axes[0].scatter(x_dist,y_dist, s=5, c=dist_corr, cmap='gist_earth_r')
        axes[0].set_ylabel('Y distance (km)')
        axes[0].set_xlabel('X distance (km)')

        # determine best fit line
        par = np.polyfit(x_dist, y_dist, 1, full=True)

        slope=par[0][0]
        intercept=par[0][1]
        xl = [min(x_dist), max(x_dist)]
        yl = [slope*xx + intercept for xx in xl]
        axes[0].plot(xl, yl, '-r')

        axes[0].set_title(title.format(angle=round(slope*90)))
        s = 'y = {slope} * x + {intercept}'.format(slope=np.round(slope,1), intercept=np.round(intercept,1))
        axes[0].annotate(s=s, xy=(.2,.8), xycoords='figure fraction')
        
        if df_corr is not None:
            plt.colorbar(scat, ax=axes[0])
        
            axes[1].set_title('Spatial correlation function of rain rate')
            axes[1].scatter((x_dist**2+y_dist**2)**.5, dist_corr, c=dist_corr, cmap='gist_earth_r')
            axes[1].set_ylabel('Correlation coefficient')
            axes[1].set_xlabel('Distance between gages (km)')

        if save:
            plt.savefig(self.save_path+'{title}.jpg'.format(title=title).format(angle=round(slope*90)))
    
    def get_df(self): 
        dates = ['year','month','day','hour','minute']
        names = dates[:]
        RG = ['RG' + str(i) for i in range(1,self.ngages+1,1)]
        [names.append(i) for i in RG]
        
        def dateparse(Y, m, d, H, M):
            d = pd.datetime(int(Y), int(m), int(d), int(H), int(M))
            return d
        self.df = pd.read_csv(self.path+self.name, header=None, names=names,
                              sep = ',', na_values = '-99',
                              parse_dates={'date_time': dates},
                              date_parser=dateparse, index_col=[0])
        self.df.columns.name = 'RG'
    
    def get_df_corr(self, df=None):
        if df is None:
            rate = self.rate
        else:
            rate = df
        df = pd.DataFrame(index=rate.columns, columns=rate.columns)
        for col in rate.columns:
            for other_col in rate.columns:
                cor = rate[col].corr(rate[other_col])
                df.set_value(col, other_col, cor)
        self.df_corr = df

    def plot_rate(self, time_step=None, base=0, interval=None,
                  gage=None, m=None, h=None, df=None, title=None,
                  save=True, bar=True, color=None, map=False, **map_kwargs):
        kwargs = dict(time_step=time_step, base=base, interval=interval, gage=gage, m=m, h=h)
        if df:
            rate = df
        else:
            rate = self.rate
        self.gb = choose_group(rate, **kwargs)
        self.df = gb_to_df(self.gb, **kwargs)
        if title is None:
            title = create_title('Mean Rain Rate', self.year, **kwargs)
        if bar:
            self.df.plot(kind='bar', figsize=(16, 6), color=color, title=title)
            plt.ylabel('Mean Rain Rate (mm/hr)')
            if save:
                plt.savefig(self.save_path+title+'.jpg')
        if map:
            try:
                if sharec:
                    title = title.replace('at listed gages', 'on same scale')
                else:
                    title = title.replace('at listed gages', 'on differing scales')
            except:
                pass
            Event(self.ll.join(self.df)).map_rain(self.save_path, 'Map of '+title, save=save, **map_kwargs)
            
    def reset_rate(self, time_step=None):
        if time_step is None:
            time_step = self.freq
        else:
            self.freq = time_step
        self.rate = self.rate.resample(time_step, **get_resample_kwargs(self.rate))
        self.reset_thresh()
        if 'min' in self.freq:
            try:
                self.per_hour = 60/int(self.freq.strip('min'))
            except:
                print('choose a new Rain.per_hour value')
        else:
            print('choose a new Rain.per_hour value')

    def reset_thresh(self): 
        self.thresh = min([i for i in self.rate[self.rate.columns[0]] if i > 0])-.001

    def get_wet_bool(self, df):
        if df is None:
            df = self.rate
        def func(x):
            if x >= self.thresh:
                return True
            elif x < self.thresh:
                return False
            else:
                return x       
        self.wet = df.apply(lambda x: x.apply(lambda x: func(x)))
    
    def get_wet(self, df):
        self.wet = self.rate >= self.thresh
        self.wet[self.rate.isnull()] = np.NaN
   
    def plot_prob_wet(self, time_step=None, interval=None, base=0,
                      gage=None, m=None, h=None, title=None,
                      save=True, bar=True, color=None, map=False, **map_kwargs):
            
        kwargs = dict(time_step=time_step, base=base, interval=interval, gage=gage, m=m, h=h)
        self.gb = choose_group(self.rate, wet=True, **kwargs)
        self.df = gb_to_prob_wet(self.gb, self.thresh, **kwargs)
        
        if time_step is None:
            time_step = self.freq
            kwargs.update(dict(time_step = self.freq))
        
        if title is None:
            title = create_title('Probability of wet {ts}', self.year, **kwargs)
            title = title + ' (threshold={t}mm)'.format(t=self.thresh/self.per_hour)

        if bar:
            self.df.plot(kind='bar', figsize=(16,6), color=color, title=title)
            plt.ylabel("Probability of wet {ts}".format(ts=time_step))
            if save:
                plt.savefig(self.save_path+title+'.jpg')
        if map:
            try:
                if sharec:
                    title = title.replace('at listed gages', 'on same scale')
                else:
                    title = title.replace('at listed gages', 'on differing scales')
            except:
                pass
            Event(self.ll.join(self.df)).map_rain(self.save_path, 'Map of '+title, save=save, **map_kwargs)
        
    def plot_boxplots(self, time_step=None, base=0, interval=None, 
                      gage=None, m=None, h=None, save=True, sort_by_type=False):
        kwargs = dict(time_step=time_step, base=base, interval=interval, gage=gage, m=m, h=h)
        
        self.group = choose_group(self.rate, **kwargs)
        wet_rates = []
        for name, df in self.group:
            if len(df.columns) > 1:
                df = df[df>=self.thresh]
                try:
                    df = df.add_suffix(' '+'{0:0=2d}'.format(name))
                except:
                    pass
                wet_rates.append((df.columns[0], df[df.columns[0]].dropna().values))
                wet_rates.append((df.columns[1], df[df.columns[1]].dropna().values))
            else:
                df = df[df >= self.thresh].dropna(how='all')  # only keep wet days
                wet_rates.append((name, df.values))
        if sort_by_type:
            wet_rates.sort()
        while len(wet_rates) > 12:
            wet_rates = [wet_rates[k] for k in range(0,len(wet_rates),2)]
        if time_step is None:
            kwargs.update(dict(time_step=self.freq))
        self.wet_rates = wet_rates
        fig = plt.figure(figsize=(max(len(wet_rates)*4/3, 4),4))
        plt.boxplot([w[1] for w in wet_rates], sym='', whis=[10,90], meanline=True, labels=[w[0] for w in wet_rates])
        plt.yscale('linear')
        plt.ylabel('Rain Rate (mm/hr)')
        title = '{ts} Rain Rate Distribution (excluding dry {ts})'
        title = create_title(title, self.year, **kwargs)
        plt.title(title)
        if save:
            plt.savefig(self.save_path+title+'.jpg')

    def plot_distribution(self, time_step=None, base=0, interval=None, 
                         gage=None, m=None, h=None, save=True):

        kwargs = dict(time_step=time_step, base=base, interval=interval, gage=gage, m=m, h=h)
        self.gb = choose_group(self.rate,**kwargs)
        labels = []
        foo = []
        for name, df in self.gb:
            df = df[df >= self.thresh] # only keep wet days
            foo.append(df.quantile(np.arange(0,1.001,.001)))
            labels.append(name)
        quan = pd.DataFrame(foo).transpose()
        quan.columns = labels
        self.quantiles = quan
        
        if time_step is None:
            kwargs.update(dict(time_step=self.freq))
        
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(211)
        
        self.quantiles.plot(logy=True,ax=ax)
        ax.set_ylabel('Rain Rate (mm/hr)') 
        title = '{ts} Rain Rate Distribution (excluding dry {ts})'
        title = create_title(title, self.year, **kwargs)
        plt.title(title)
        if save:
            plt.savefig(self.save_path+title+'.jpg')
        
        ax1 = fig.add_subplot(212)
        self.quantiles.plot(xlim=(0.9,1),ax=ax1, legend=None)
        ax1.set_ylabel('Rain Rate (mm/hr)') 
    
    def get_rainiest(self, n, time_step='24H', base=12, unweighted=False):
        largest = None
        if not hasattr(self,'ll'):
            self.get_ll()
        
        if unweighted:
            if not hasattr(self, 'daily'):
                self.daily = unweighted_daily_mean(self.rate)
            largest = self.daily.dropna().sort_values().tail(n)
        
        rainiest = self.rate.resample(time_step, base=base, **get_resample_kwargs(self.rate))
        
        if type(rainiest) == pd.Panel:
            largest = rainiest.mean(axis=get_index(self.rate, 'RG')[0]).dropna(how='any').mean(axis=1).sort_values().tail(n)
            ra = rainiest.loc[:,largest.index].transpose(0,2,1)
            r = self.ll.join(ra.gage.add_prefix('Gage ')).join(ra.radar.add_prefix('Radar '))
            bar = len(r.columns)
            foo = (bar-5)/2+5
            q = zip(range(5,foo), range(foo,bar))
            l = range(5)
            [l.extend(item) for item in q]
            r = r[l]
        else:
            if largest is None:
                largest = rainiest.mean(axis=get_index(self.rate, 'RG')[0]).dropna().sort_values().tail(n)
            r = self.ll.join(rainiest.loc[largest.index].transpose())
               
        self.rainiest = r[r.lat > -200] 
    
    def get_storm(self, storm_day='2013-08-13', time_step=None, path='SVG_data'):
        if time_step is None:
            df = self.rate[storm_day]
            time_step = self.freq
        else:
            df = self.rate[storm_day].resample(time_step, **get_resample_kwargs(self.rate)).dropna(how='all')
        
        # nCr for 18C2 produces just 5 bins of 30, so this is the fewest for good results
        wet = df >= self.thresh
        df = df[wet.sum(axis=1) > 18] 

        if not hasattr(self,'ll'):
            self.get_ll()

        storm = self.ll.join(df.transpose())
        storm = storm[storm.lat > -200]

        for col in storm:
            if col not in self.ll.columns:
                storm[col] = storm[col].replace(0, np.nan)
        storm.to_csv(path, index=False)
        
        storm = self.ll.join(df.transpose())
        self.storm = storm[storm.lat > -200]

    def get_max_lowess(self, df=None, interval='diurnal', f=1/4., example_plot=4):
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        rfuncs = import_r_tools(filename='SVG.r')
        
        if df is None:
            df = self.rate
        df = df.dropna(how='all')
        if interval is 'diurnal':
            tt = df.index.hour + df.index.minute/60.
            amin, amax = 0, 24
        if interval is 'seasonal':
            tt = df.index.month + df.index.day/df.index.days_in_month.astype(float)
            amin, amax = 1, 13
        tod = df.set_index(tt)
        tod = tod.sort_index()
        d = {}
        for i in tod.index.unique():
            d.update({i: tod.loc[i].mean()})
        a = pd.DataFrame(d).transpose()
        b = pd.concat((a.set_index(a.index - (amax - amin)),
                       a,
                       a.set_index(a.index + (amax - amin))))
        d={}
        k=0
        for col in b.columns:
            foo = pandas2ri.ri2py(rfuncs.get_lowess(x=b.index, y=b[col], f=f))
            foo = foo.set_index(foo.x)
            foo = foo.loc[amin:amax]
            d.update({col: foo.y.idxmax()})
            k+=1
            if k == example_plot:
                fig, ax = plt.subplots()
                ax.scatter(x=b.index, y=b[col])
                ax.plot(foo.x, foo.y, c='r')
                ax.set_xlim(amin, amax)
        lm = pd.DataFrame({'lowess_max': pd.Series(d)})
        return lm

class RadarGage(Rain):
    
    def __init__(self, radar, gage):
        self.rate = pd.Panel({'gage': gage.rate, 'radar': radar.rate})
        self.freq = gage.freq
        self.per_hour = gage.per_hour
        self.year = gage.year
        self.thresh = gage.thresh
        self.path = gage.path
        self.ll_file = gage.ll_file
        self.ll = gage.ll
        if gage.per_hour != radar.per_hour:
            print('gage and radar time steps don\'t match')
        
    def get_nonan(self):
        self.rate = self.rate.loc[:, self.rate.gage.mean(axis=1).notnull()].loc[:, self.rate.radar.mean(axis=1).notnull()]
    
    def plot_correlation(self, p=None, time_step=None, base=0,  title=None, save=True):
        if p is not None:
            p = p
        elif time_step is None:
            p = self.rate
            time_step = self.freq
        else:
            p = self.rate.resample(time_step, base=base, **get_resample_kwargs(self.rate))
        self.df = p
        p = p.to_frame().dropna(how='any')

        plt.figure(figsize=(8,8))
        plt.scatter(x=p.gage, y=p.radar, s=5)
        plt.ylabel('radar')
        plt.xlabel('gage')
        m = min(p.max())
        plt.xlim(0, m)
        plt.ylim(0, m)

        par = np.polyfit(p.gage, p.radar, 1, full=True)
        correlation = np.corrcoef(p.gage, p.radar)[0,1]
        slope=par[0][0]
        intercept=par[0][1]
        xl = [min(p.gage), max(p.gage)]
        yl = [slope*xx + intercept for xx in xl]
        plt.plot(xl, yl, '-r')
        
        if time_step is None:
            time_step = self.freq
        
        if title is None:
            title = 'Radar Rain Gage Correlation with R^2 = {R} for'.format(R=round(correlation**2, 2))
            title = create_title('{ts} '+title, year=self.year, time_step=time_step)
        plt.title(title)
        if save:
            plt.savefig(self.save_path+title+'.jpg')
        
"""
Common methods to be used on objects that might end up being called events. 
These objects would have spatial information as columns and 

"""
from __init__ import *
from common import *

import statsmodels.formula.api as sm


class Event:
    '''
    Base class for Event files
    '''  
    def __init__(self, df):
        '''
        Initialize intance of Event class
        
        Parameters
        ----------
        df : Dataframe object with locations as the index and values to map as the columns
        
        **kwargs
        '''
        self.df = df[df.lat > -200]
        self.data_cols = [col for col in self.df.columns if col not in ('RG','lat','lon','X','Y')]
        self.ll_cols = [col for col in self.df.columns if col not in self.data_cols]
        self.ll = self.df[self.ll_cols]

    def get_latlon(self, latlon):
        if latlon:
            x,y = self.df['lon'], self.df['lat']
        else:
            try:
                x,y = self.df['X'], self.df['Y']
            except:
                x,y = self.df['x'], self.df['y']  
        return x, y
    
    def map_rain(self, save_path='./', title='rain_map', sharec=False, save=True, cmap='gist_earth_r', 
                 top_to_bottom=False, hide_title=False, latlon=True):
        """
        Map rainfall at each gage location 

        Parameters
        ----------
        df : Dataframe object with locations as the index and values to map as the columns

        **kwargs

        Returns
        -------
        fig : map of rain intensity at each location
        """
        df = self.df
        x, y = self.get_latlon(latlon)
        cols = self.data_cols
        if len(cols) == 1:
            ncols = 1
            nrows = 1
        else:
            ncols = 2
            nrows = int(np.ceil(len(cols)/float(ncols)))
        if top_to_bottom:
            nrows, ncols = ncols, nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*8, 5*nrows), sharex='row', sharey='row')
        if sharec:
            try:
                vmin, vmax = sharec
            except:
                vmax = min(100, df[cols].max().max())
                vmin = max(0, df[cols].min().min())
        if not hide_title:    
            fig.suptitle(title, fontsize=18)
            fig.subplots_adjust(top=.85, hspace=.3, wspace=0.1)

        try:
            axes = axes.reshape(-1)
        except:
            axes = [axes]
        for col, ax in zip(cols, axes):
            if sharec:
                scat = ax.scatter(x=x, y=y, c=df[col], s=100, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                scat = ax.scatter(x=x, y=y, c=df[col], s=100, cmap=cmap)
                fig.colorbar(scat, ax=ax)
            ax.set_title(col)
        if sharec:
            fig.colorbar(scat, ax=list(axes))
        if save:
            plt.savefig(save_path+title+'.png')

    def movie(self, vmin=None, vmax=None, cmap='gist_earth_r', latlon=True):
        """
        Make a movie of rainfall maps

        Parameters
        ----------
        df : Dataframe object with string location names as the index and values to map as the columns

        **kwargs

        Returns
        -------
        animation : animation of maps (breaks on matplotlib > 1.4.3)
        """    
        df = self.df
        x, y = self.get_latlon(latlon)
        cols = self.data_cols

        if not vmin:
            vmin=0
        if not vmax:
            vmax = min(100, df[cols].max().max())

        fig, ax = plt.subplots(1,1,figsize= (10,6))
        sc = ax.scatter(x=x, y=y, cmap=cmap, c=y*0, vmin=vmin, vmax=vmax, s=100)
        fig.colorbar(sc)

        def animate(i):
            ax.set_title(df[[i+len(self.ll_cols)]].columns[0])
            scat = ax.scatter(x=x, y=y, cmap=cmap, c=df[[i+len(self.ll_cols)]], vmin=0, vmax=vmax, s=100)

        return animation.FuncAnimation(fig, animate, frames=len(cols), interval=300, blit=True)

    
    def detrend(self, latlon=True, plot=False, drop_zeros=True):
        """
        Generate a dataframe containing values linearly detrended in space

        Parameters
        ----------
        df : Dataframe object with string location names as the index and values to map as the columns

        **kwargs

        Returns
        -------
        res : Dataframe of means plus residuals for each time and location
        """
        df = self.df
        x, y = self.get_latlon(latlon)
        cols = self.data_cols

        res = self.ll
        for col in cols:
            foo = pd.DataFrame(x).join((y, df[col])).dropna(how='any')
            foo.columns = ['lon', 'lat', 'col']
            result = sm.ols(formula="col ~ lon + lat", data=foo).fit()
            fit = result.params['Intercept'] + result.params['lon']*foo.lon +result.params['lat']*foo.lat
            if plot:
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
                for l, ax in zip(['lon', 'lat'], axes):
                    ax.scatter(x=foo[l], y=foo.col)
                    ax.scatter(foo[l], fit, c='r')
                    ax.set_ylabel('Rain Rate (mm/hr)')
                    ax.set_xlabel(l)
                    ax.set_title('Trend in ' + l)
            res = res.join(pd.DataFrame({col: df[col]-fit + df[col].mean()}))
        if drop_zeros:
            zeros = [col for col in cols if res[col].mean()==0]
            res = res.drop(zeros, axis=1)
        self.res = res

    def variogram(self, i=5, plot_v=True, **kwargs):
        """
        Generate a variogram from a dataframe with a single data column or a column index number

        Parameters
        ----------
        i : data column index number (defaults to 5)

        **kwargs (target_np, alpha, tol_hor, max_bnd, last_max)

        Returns
        -------
        v : Dataframe containing output from r-variogram function
        """
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        rfuncs = import_r_tools()
        
        df = self.df
        
        r_df = df[[0,1,2,3,4, i]].dropna(how='any')
        v = pandas2ri.ri2py(rfuncs.get_iSVG(r_df, 6, **kwargs))
        if plot_v:
            v.plot(x='dist', y='gamma', marker = 'o', figsize=(8,4))
        return v

    def krige(self, i=5, v=None, step=1, res=True, plot_v=False, plot_k=True, animated=False, **plot_kwargs):
        """
        Krige the dataframe with a single data column or a column index number

        Parameters
        ----------
        i : data column index number (defaults to 5)

        v : variogram to use in determining sill and range

        step : grid interval to krige on (in km)

        **kwargs

        Returns
        -------
        k : Dataframe containing output from r-krige function
        """
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        rfuncs = import_r_tools()
        if res:
            if not hasattr(self, 'res'):
                self.detrend()
            df = self.res
        else:
            df = self.df
        
        r_df = df[[0,1,2,3,4, i]].dropna(how='any')
        if not v:
            v = pandas2ri.ri2py(rfuncs.get_variogram(r_df))

        model = 'Sph'
        psill = r_df.var()[4]
        for j in range(len(v)):
            if v.gamma[j] > psill:
                rng = v.dist[j]
                break
        k = pandas2ri.ri2py(rfuncs.get_krige(r_df, psill, model, rng, step=step))
        k['lat'] = k.y/110.574
        k['lon'] = k.x/(111.320*(k['lat']*pi/180).apply(cos))
        if plot_k and animated:
            return self.plot_krige(i, k, rng, step=step, res=res, animated=animated, **plot_kwargs)
        elif plot_k and not animated:
            self.plot_krige(i, k, rng, step=step, res=res, animated=animated, **plot_kwargs)
            return k
        return k

    def plot_krige(self, i, k, rng, step=1, res=True, animated=False, 
                   ax=None, cmap='gist_earth_r', vmin=None, vmax=None, latlon=False):
        if res:
            if not hasattr(self, 'res'):
                self.detrend()
            df = self.res
        else:
            df = self.df
        if not ax:
            fig, ax = plt.subplots(figsize=(10,6))
        if not vmin:
            vmin = max(0, df[[i]].min().values[0])
        if not vmax:
            vmax = df[[i]].max().values[0]
        if latlon:
            ax.scatter(k.lon, k.lat, c=k['var1.pred'], cmap=cmap, marker='s', edgecolors='none', s=step*300, vmin=vmin, vmax=vmax)
            scat = ax.scatter(df.lon, df.lat, c=df[[i]], cmap=cmap, edgecolors='1', vmin=vmin, vmax=vmax)
            ax.set_xlim(min(df.lon), max(df.lon))
            ax.set_ylim(min(df.lat), max(df.lat))
        else:
            ax.scatter(k.x, k.y, c=k['var1.pred'], cmap=cmap, marker='s', edgecolors='none', s=step*300, vmin=vmin, vmax=vmax)
            scat = ax.scatter(df.X, df.Y, c=df[[i]], cmap=cmap, edgecolors='1', vmin=vmin, vmax=vmax)
            ax.set_xlim(min(df.X), max(df.X))
            ax.set_ylim(min(df.Y), max(df.Y))
        ax.set_title('{t} (range={dts}km)'.format(t=df.columns[i], dts=round(rng)))
        if animated:
            return scat
        plt.colorbar(scat)

    def get_SVG(self, **kwargs):
        """
        Generate a panel of variograms from a dataframe

        Parameters
        ----------
        **kwargs (target_np, alpha, tol_hor, max_bnd, last_max)

        Returns
        -------
        SVGs : panel of variograms with the date_times as the items
        """
        df = self.df
        cols = self.data_cols
        
        d = {}
        for col in cols:
            d.update({col: variogram(df, df.axes[1].get_loc(col), **kwargs)})
        SVGs = pd.Panel(d)
        self.SVGs

    def combine_SVGs(self):
        """
        Wrapper for get_SVG to create a consolidated dataframe of dist vs gamma

        Returns
        -------
        combined_SVG : dataframe of dist vs gamma
        """
        SVGs = self.SVGs
        k = 0
        for i in SVGs.items:
            s = SVGs[i,:,'gamma']
            s.set_axis(0, SVGs[i,:,'dist'])
            s.name = i
            k += 1
            if k == 1:
                a = pd.DataFrame(s)
            else:
                a = a.join(s)
        combined_SVG = a
        self.combined_SVG

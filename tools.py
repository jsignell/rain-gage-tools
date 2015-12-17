"""
Common methods to be used on objects that might end up being called events. 
These objects would have spatial information as columns and 

"""
from __init__ import *
from common import *

import statsmodels.formula.api as sm

def map_rain(df, save_path='./', title='rain_map', sharec=False, save=True, cmap='gist_earth_r', 
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
    if latlon:
        x,y = df['lon'], df['lat']
    else:
        try:
            x,y = df['X'], df['Y']
        except:
            x,y = df['x'], df['y']
    df = df[x > -200]
    
    cols = [col for col in df.columns if col not in ('RG','lat','lon','X','Y')]
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
        plt.savefig(save_path+title+'.jpg')

def movie(df, vmin=None, vmax=None, cmap='gist_earth_r', latlon=True):
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
    if latlon:
        x,y = df['lon'], df['lat']
    else:
        try:
            x,y = df['X'], df['Y']
        except:
            x,y = df['x'], df['y']
    
    df = df[x > -200]
    cols = [col for col in df.columns if col not in ('RG','lat','lon','X','Y')]
    ll = [col for col in df.columns if col not in cols]        
    
    if not vmin:
        vmin=0
    if not vmax:
        vmax = min(100, df[cols].max().max())
    
    fig, ax = plt.subplots(1,1,figsize= (10,6))
    sc = ax.scatter(x=x, y=y, cmap=cmap, c=y*0, vmin=vmin, vmax=vmax, s=100)
    fig.colorbar(sc)

    def animate(i):
        ax.set_title(df[[i+len(ll)]].columns[0])
        scat = ax.scatter(x=x, y=y, cmap=cmap, c=df[[i+len(ll)]], vmin=0, vmax=vmax, s=100)

    return animation.FuncAnimation(fig, animate, frames=len(cols), interval=300, blit=True)

def detrend(df, latlon=True, plot=False, drop_zeros=True):
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
    if latlon:
        x,y = df['lon'], df['lat']
    else:
        try:
            x,y = df['X'], df['Y']
        except:
            x,y = df['x'], df['y']
    
    df = df[x > -200]
    cols = [col for col in df.columns if col not in ('RG','lat','lon','X','Y')]
    ll = [col for col in df.columns if col not in cols]
    res = df[ll]
    
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
    return res

def import_r_tools(filename='SVG.r'):
    from rpy2.robjects import pandas2ri, r, globalenv
    from rpy2.robjects.packages import STAP
    pandas2ri.activate()
    with open(filename, 'r') as f:
        string = f.read()
    rfuncs = STAP(string, "rfuncs")
    return rfuncs

def variogram(df, i, plot_v=True, **kwargs):
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    rfuncs = import_r_tools()
    
    r_df = df[[0,1,2,3,4, i]].dropna(how='any')
    v = pandas2ri.ri2py(rfuncs.get_iSVG(r_df, 6, **kwargs))
    if plot_v:
        v.plot(x='dist', y='gamma', marker = 'o', figsize=(8,4))
    return v
     
def krige(df, i, v=None, step=1, plot_v=False, plot_k=True, animated=False, **plot_kwargs):
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    rfuncs = import_r_tools()
    
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
    if plot_k and animated:
        return plot_krige(df, i, k, rng, step=step, animated=animated, **plot_kwargs)
    elif plot_k and not animated:
        plot_krige(df, i, k, rng, step=step, animated=animated, **plot_kwargs)
        return k
    return k
                  
def plot_krige(df, i, k, rng, step=1, animated=False, ax=None, cmap='gist_earth_r', vmin=None, vmax=None):
    if not ax:
        fig, ax = plt.subplots()
    if not vmin:
        vmin = max(0, df[[i]].min().values[0])
    if not vmax:
        vmax = df[[i]].max().values[0]
    ax.scatter(k.x, k.y, c=k['var1.pred'], cmap=cmap, marker='s', edgecolors='none', s=step*300, vmin=vmin, vmax=vmax)
    scat = ax.scatter(df.X, df.Y, c=df[[i]], cmap=cmap, edgecolors='1', vmin=vmin, vmax=vmax)
    ax.set_xlim(min(df.X), max(df.X))
    ax.set_ylim(min(df.Y), max(df.Y))
    ax.set_title('{t} (range={dts}km)'.format(t=df.columns[i], dts=round(rng)))
    if animated:
        return scat
    plt.colorbar(scat)

def get_SVG(df, **kwargs):
    cols = [col for col in df.columns if col not in ('RG','lat','lon','X','Y')]    
    d = {}
    for col in cols:
        d.update({col: variogram(df, df.axes[1].get_loc(col), **kwargs)})
    SVG_storm = pd.Panel(d)
    return SVG_storm

def combine_SVGs(SVG_storm):
    k = 0
    for i in SVG_storm.items:
        s = SVG_storm[i,:,'gamma']
        s.set_axis(0, SVG_storm[i,:,'dist'])
        s.name = i
        k += 1
        if k == 1:
            a = pd.DataFrame(s)
        else:
            a = a.join(s)
    combined_SVG = a
    return combined_SVG
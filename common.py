"""
Common methods to be used by core objects

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def choose_group(df, interval, gage=None, m=None, h=None):
    """
    Choose group fitting the given criteria
    
    Parameters
    ----------
    df : Dataframe object with times down the index and rain gages as the columns
    
    interval : {'seasonal', 'diurnal'}
    
    gage : list of gages as strings ['RG1', 'RG2'...]
    
    m : int exclude all months but this one
    
    h : int exclude all hours but this one

    Returns
    -------
    group : map of rain intensity at each location
    """
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

        
def map_rain(df):
    """
    Map rainfall at each gage location 
    
    Parameters
    ----------
    df : Dataframe object with locations as the index and values to map as the columns

    Returns
    -------
    fig : map of rain intensity at each location
    """
    cols = [col for col in df.columns if col not in ('RG','lat','lon','X','Y')]
    
    nrows = int(np.floor(len(cols)**.5))
    ncols = int(np.ceil(len(cols)/nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(min(ncols*5, 16), nrows*4), sharey=True)
    fig.subplots_adjust(hspace=.3, wspace=0.1)

    for col, ax in zip(cols, axes.reshape(1, len(cols))[0]):
        df.plot(kind='scatter', x='lon', y='lat', c=col, s=100, cmap='gist_earth_r', ax=ax)
        ax.set_title(col)
        
    return fig
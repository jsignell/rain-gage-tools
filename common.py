"""
Common methods to be used by core objects

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def get_index(df, index='date_time'):
    for i, full in enumerate(df.axes):
        if full.name == index:
            return (i, full)
    
def get_resample_kwargs(df):
    resample_kwargs = dict(how='mean', closed='right', label='right')
    i = get_index(df, index='date_time')[0]
    resample_kwargs.update({'axis': i})
    return resample_kwargs

def choose_group(df, time_step=None, base=0, interval=None, gage=None, m=None, h=None):
    """
    Choose group fitting the given criteria
    
    Parameters
    ----------
    df : Dataframe object with times down the index and rain gages as the columns, or 
         a Panel object of two dataframes. 
    
    time_step : time string to use as method for resample: '15min', '1H', ...
    
    base : integer offest used in resample
    
    interval : {'seasonal', 'diurnal'}
    
    gage : list of gages as strings ['RG1', 'RG2'...]
    
    m : int or list of ints exclude all months but this/these
    
    h : int or list of ints exclude all hours but this/these

    Returns
    -------
    group : groupby object
    """    
    if time_step is not None:
        df = df.resample(time_step, base=base, **get_resample_kwargs(df))
    date_time = get_index(df, 'date_time')[1]
    a = get_index(df, index='RG')[0]
    
    # Choose along gage axis
    if gage is None:
        df = df.mean(axis=a)
    elif type(gage) is not list and type(gage) is not tuple:
        df = df[[gage]]
    else:
        df = df[gage]
        
    # Group along time axis
    if (interval is 'seasonal' and h is None) or (interval is 'diurnal' and m is not None):
        gb = df.groupby(date_time.month)
    elif (interval is 'diurnal' and m is None) or (interval is 'seasonal' and h is not None):
        gb = df.groupby(date_time.hour)

    if m is None and h is None:
        return gb
    elif interval is 'diurnal' and m is not None:
        if type(m) is list or type(m) is tuple:
            df = pd.concat([gb.get_group(n) for n in m])
        else:
            df = gb.get_group(m)
        date_time = get_index(df, 'date_time')[1]
        gb = df.groupby(date_time.hour)
    
    elif interval is 'seasonal' and h is not None:
        if type(h) is list or type(h) is tuple:
            df = pd.concat([gb.get_group(n) for n in h])
        else:
            df = gb.get_group(h)
        date_time = get_index(df, 'date_time')[1]
        gb = df.groupby(date_time.month)
    
    return gb

        
def map_rain(df, save_path='.', title='rain_map'):
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10/(ncols)*nrows), sharey=True)
    fig.suptitle(title, fontsize=18)
    fig.subplots_adjust(top=.85, hspace=.3, wspace=0.1)

    for col, ax in zip(cols, axes.reshape(1, len(cols))[0]):
        df.plot(kind='scatter', x='lon', y='lat', c=col, s=100, cmap='gist_earth_r', ax=ax)
        ax.set_title(col)
   
    plt.savefig(save_path+title+'.jpg')

def create_title(title, year=None, time_step=None, base=0, interval=None,
                 gage=None, m=None, h=None):
    """
    Create an appropriate title from the rain related parameters and the starter title
    
    Parameters
    ----------
    title : str optionally containing '{ts}' where the time_step should go
    
    **kwargs

    Returns
    -------
    title : str containing a legible title which can also be used as a filename
    """
    if gage is not None:
        title = '{g}: '.format(g=', '.join(gage))+title
    if m is not None:
        title = title + ' for Month {m} of'.format(m=m)
    elif h is not None:
        title = title + ' for Hour {h} of'.format(h=h) 
    elif interval is 'seasonal':
        title = title + ' for Months of'
    elif interval is 'diurnal':
        title = title + ' for Hours of'
    if time_step is not None:
        ts = time_step.replace('min', ' minute').replace('T', ' minute').replace('H', ' hour').replace('D', ' day')
        title = title.format(ts=ts)
    if year is not None:
        title = title +' '+ year
    return title
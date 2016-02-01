"""
Common methods to be used by core objects

"""
from math import cos, pi
import string
import posixpath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from JSAnimation import IPython_display
except:
    pass
from matplotlib import animation
plt.style.use('ggplot')

def import_r_tools(filename='SVG.r'):
    import os
    from rpy2.robjects import pandas2ri, r, globalenv
    from rpy2.robjects.packages import STAP
    pandas2ri.activate()
    path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(path,filename), 'r') as f:
        string = f.read()
    rfuncs = STAP(string, "rfuncs")
    return rfuncs

def get_index(df, index='date_time'):
    """
    Find the axis of panel or dataframe with given name
    
    Parameters
    ----------
    df : Dataframe object with times down the index and rain gages as the columns, or 
         a Panel object of two such dataframes. 
    
    index : str name of axis of interest. For Rain and RadarGage objects: 'date_time' or 'RG'

    Returns
    -------
    i, full : index of desired axis, axis itself. 
    """    
    for i, full in enumerate(df.axes):
        if full.name == index:
            return (i, full)
    
def get_resample_kwargs(df):
    """
    Get a dict of kwargs including axis to use in resampling Rain and RadarGage objects
    
    Parameters
    ----------
    df : Dataframe object with times down the index and rain gages as the columns, or 
         a Panel object of two such dataframes. 
   
    Returns
    -------
    resample_kwargs : dict of kwargs including axis to use in resampling Rain and RadarGage objects
    """    
    resample_kwargs = dict(how='mean', closed='right', label='right')
    i = get_index(df, index='date_time')[0]
    resample_kwargs.update({'axis': i})
    return resample_kwargs

def get_prob_wet(df, a, thresh):
    wet = None
    d = {}
    if a is None:
        w = df[df >= thresh].count()
        nonnull = df.count()
        wet =  w/float(nonnull)
        return wet
    for i in df.axes[a]:
        w = df[i][df[i]>= thresh].count()
        nonnull = df[i].count()
        try:
            wet = pd.concat((wet, pd.Series({i: w/float(nonnull)})))
        except:
            d.update({i: w/nonnull})
    if d != {}:
        wet = pd.DataFrame(d)
    return wet

def interval_is_None(df, gage):
    if type(gage) is tuple or type(gage) is list:
        try:
            df = df.mean(axis=get_index(df, 'date_time')[0]).loc[gage]
        except:
            df = df.mean(axis=get_index(df, 'date_time')[0]).loc[gage, :]
    else:
        try:
            df = df.to_frame().mean()
        except:
            df = df.mean(axis=get_index(df, 'date_time')[0])
    return df

def mean_of_group(gb):
    """
    Compute the mean of a groupby object containing dataframes
    takes about 3 times as long as regular mean, but uses better method
    
    Parameters
    ----------
    gb : groupby object containing dataframes
    
    Returns
    -------
    s : pandas.Series object with the groupby keys as the index
    """
    if type(gb.get_group(1)) is pd.DataFrame:
        d = {}
        for name, df in gb:
            mean = np.nanmean(df.values)
            d.update({name: mean})
        s = pd.Series(d)
        return s
    
    else:
        items= gb.get_group(1).items
        d = {key: {} for key in items}
        for name, p in gb:
            for i in items:
                mean = np.nanmean(p[i].values)
                d[i].update({name: mean})
        df = pd.DataFrame(d)
        return df

def unweighted_daily_mean(real_df, base=12):
    """
    takes about 200 times as long as regular resampling but yeilds a better result if only 
    marginally so
    
    Parameters
    ----------    
    real_df : pandas.DataFrame object 
    
    base : as in resample
    
    Returns
    -------
    s : pandas.Series object containing the daily means 
    """
    s = None
    bar = None
    tomorrow = (real_df.index[0] + pd.DateOffset(1)).date()
    today = real_df.index[0].date()
    for (d, h), df in real_df.groupby((real_df.index.date,real_df.index.hour)):
        if d==tomorrow and h<base:
            bar = np.concatenate((bar,df.values.reshape(-1)))
        elif h == base:
            if bar is not None:
                val = np.nanmean(bar)
                s = pd.concat((s, pd.Series({d : val})))
            bar = df.values.reshape(-1)
            today = d
            tomorrow = (d + pd.DateOffset(1)).date()
        elif d==today and h>base:
            bar = np.concatenate((bar, df.values.reshape(-1)))
        else:
            continue
    return s

def choose_group(df, time_step=None, base=0, interval=None, gage=None, m=None, h=None, wet=False):
    """
    Choose group fitting the given criteria
    
    Parameters
    ----------
    df : Dataframe object with times down the index and rain gages as the columns, or 
         a Panel object of two dataframes. 
    
    **kwargs:
    
    time_step : time string to use as method for resample: '15min', '1H', ...
    
    base : integer offest used in resample
    
    interval : {'seasonal', 'diurnal'}
    
    gage : list of gages as strings ['RG1', 'RG2'...]
    
    m : int or list of ints exclude all months but this/these
    
    h : int or list of ints exclude all hours but this/these
    
    wet : bool to indicate whether the accumutaion should be taken when resampling

    Returns
    -------
    gb : groupby object
    """    
    if time_step is not None:
        resample_kwargs = get_resample_kwargs(df)
        if wet:
            resample_kwargs.update({'how':'sum'})
        df = df.resample(time_step, base=base, **resample_kwargs)
        
    date_time = get_index(df, 'date_time')[1]
    a, RG = get_index(df, 'RG')
    
    # Choose along gage axis
    if gage is None:
        df = df.mean(axis=a)
    else:
        try:
            df = df.loc[:,gage]
        except: 
            df = df.loc[:,:,gage]
        try:
            a, RG = get_index(df, index='RG')
        except:
            pass
   
    # Group along time axis
    if interval is 'seasonal':
        if h is not None:
            gb = df.groupby(date_time.hour)
            if type(h) is list or type(h) is tuple:
                df = pd.concat([gb.get_group(n) for n in h])
            else:
                df = gb.get_group(h)
            date_time = get_index(df, 'date_time')[1]
        gb = df.groupby(date_time.month)
        if m is not None:
            try:
                gb = [(m, gb.get_group(m))]
            except:
                gb = [(month, gb.get_group(month)) for month in m]
    
    elif interval is 'diurnal':    
        if m is not None:
            gb = df.groupby(date_time.month)
            if type(m) is list or type(m) is tuple:
                df = pd.concat([gb.get_group(n) for n in m])
            else:
                df = gb.get_group(m)
            date_time = get_index(df, 'date_time')[1]
        gb = df.groupby(date_time.hour)
        if h is not None:
            try:
                gb = [(h, gb.get_group(h))]
            except:
                gb = [(hour, gb.get_group(hour)) for hour in h]
    
    else:
        gb = [('all',df)]

    return gb

def gb_to_df(gb, time_step=None, base=0, interval=None, gage=None, m=None, h=None):
    if interval is 'seasonal' and m is not None:
        if type(m) is not list and type(m) is not tuple:
            m = [m]
        df = gb.mean().loc[m]
    elif interval is 'diurnal' and h is not None:
        if type(h) is not list and type(h) is not tuple:
            h = [h]
        df = gb.mean().loc[h]
    elif interval is 'diurnal' or interval is 'seasonal':
        df = gb.mean()
    else:
        df = interval_is_None(gb[0][1], gage)
    return df

def gb_to_prob_wet(gb, thresh, time_step=None, base=0, interval=None, gage=None, m=None, h=None):
    if interval is None:
        try:
            wet = get_prob_wet(gb[0][1], 0, thresh)
        except:
            wet = get_prob_wet(gb[0][1], 1, thresh)
        return wet
    try:
        indicator = gb.get_group(gb.keys[0])
    except:
        indicator = gb[0][1]
    if gage is not None:
        a = get_index(indicator, 'RG')[0]
    elif gage is None:
        if type(indicator) is pd.DataFrame:
            a = 1
        else:
            a = None
    d = {}
    for name, df in gb:
        d.update({name: get_prob_wet(df, a, thresh)})
    if type(d.values()[0]) is float or type(d.values()[0]) is np.float64:
        wet = pd.Series(d)
    elif type(d.values()[0]) is pd.Series:
        wet = pd.DataFrame(d)
    elif type(d.values()[0]) is pd.DataFrame:
        wet = pd.Panel(d)
    else:
        wet = d
    if interval is not None:
        try:
            wet = wet.transpose()
        except:
            pass
    return wet

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
    if type(gage) is list or type(gage) is tuple:
        title = title + ' at listed gages'
    elif gage is not None:
        title = title + ' at '+ gage
    
    if m is not None:
        title = title + ' for Month {mo} of'.format(mo=m)
    elif h is not None:
        title = title + ' for Hour {ho} of'.format(ho=h) 
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
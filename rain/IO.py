from common import *
from event import *
from rain import *
import xarray as xr

def to_netcdf(Rain_gage=None, Rain_radar=None, path='', out_file='{site}.nc', site='City', **kwargs):
    
    if Rain_gage:
        Rain_gage.rate.index.name = 'time'
        datasets = [xr.DataArray(Rain_gage.rate[i]/Rain_gage.per_hour) for i in Rain_gage.rate.columns]
        combined = xr.concat(datasets, 'station')
        ds0 = combined.to_dataset(name='rain_gage')
        ds0.rain_gage.attrs.update({'units':Rain_gage.units, 'standard_name': 'gage rain depth', 'coordinates': 'lat lon'})
        ds0.rain_gage.encoding = {'chunksizes': [5, 100000], 'zlib': True}

        f = Rain_gage.ll.loc[:,['lat', 'lon']]
        f = f.reset_index(range(len(f.index)))
        f.index.name='station'
        ds1 = xr.Dataset.from_dataframe(f.loc[:,['lat','lon']])
        ds_ = ds0.merge(ds1)
    
    if Rain_radar:
        Rain_radar.rate.index.name = 'time'
        datasets = [xr.DataArray(Rain_radar.rate[i]/Rain_radar.per_hour) for i in Rain_radar.rate.columns]
        combined = xr.concat(datasets, 'station')
        ds0 = combined.to_dataset(name='rain_radar')
        
        if not ds_:
            f = Rain_radar.ll.loc[:,['lat', 'lon']]
            f = f.reset_index(range(len(f.index)))
            f.index.name='station'
            ds1 = xr.Dataset.from_dataframe(f.loc[:,['lat','lon']])
            ds_ = ds0.merge(ds1)
            
        ds_ = ds_.merge(ds0)
        ds_.rain_radar.attrs.update({'units':Rain_radar.units, 'standard_name': 'radar rain depth', 'coordinates': 'lat lon'})
        ds_.rain_radar.encoding = {'chunksizes': [5, 100000], 'zlib': True}
   
    ds_.lat.attrs.update({'standard_name': 'latitude', 'long_name':'station latitude', 'units': 'degrees_north'})
    ds_.lon.attrs.update({'standard_name': 'longitude', 'long_name':'station longitude', 'units': 'degrees_east'})
    ds_.station.encoding = {'chunksizes': [5],'zlib': True}
    ds_.time.encoding = {'units':'minutes since 1970-01-01', 'calendar':'gregorian',
                         'chunksizes': [100000], 'zlib': True}

    ds_.attrs.update({'description': '{site} rain gage network'.format(site=site),
                      'history': 'Created {now}'.format(now=pd.datetime.now())})

    ds_.to_netcdf(path=path+out_file.format(site=site), format='netCDF4', **kwargs)
    ds_.close()
    
    
def read_netcdf(nc_file):
    
    ds = xr.open_dataset(nc_file, decode_coords=False)
    print ds
    
    gage=None
    radar=None
    
    vars = ds.data_vars.keys()
    for var in vars:
        if var.lower()[:3] not in ['lat', 'lon']:
            df = ds[var].to_pandas()
            if len(df.columns)> len(df.index):
                df = df.T
            if 'gage' in df.columns.name.lower() or df.columns.name=='station':
                df.columns.name = 'RG'
            if 'time' in df.index.name.lower():
                df.index.name = 'date_time'
            try:
                freq = (df.index[1]-df.index[0]).seconds/60
                kwargs = {'ngages': min(ds.dims.values()),
                          'units': ds[var].units,
                          'per_hour': 60/freq,
                          'freq': str(freq)+'min'}
            except:
                kwargs = None
            if 'gage' in var:
                gage = {'df':df, 'kwargs':kwargs}
            if 'radar' in var:
                radar = {'df':df, 'kwargs':kwargs}
            vars.remove(var)
    ll = ds[vars].to_dataframe()

    print('')
    print('Rain objects need specific index and column names: ')
    print('RG, date_time, lat, lon. Trying to set them...')

    if 'gage' in ll.index.name.lower() or ll.index.name=='station':
        ll.index.name = 'RG'
    ll.columns = [l.lower()[:3] for l in ll.columns]

    print ''
    if not gage and not radar:
        if df.columns.name == ll.index.name =='RG' and df.index.name=='date_time' and 'lat' in ll.columns and 'lon' in ll.columns:
            print 'Sucess!'
            if kwargs is not None:
                rg = Rain(df, **kwargs)
                rg.set_ll(ll)
                return rg
        else:
            print 'Manual editing needed'
    if gage:
        gage = Rain(gage.get('df'), show=False, **gage.get('kwargs'))
        gage.set_ll(ll)
        if not radar:
            gage.show()
            return gage
    if radar:
        radar = Rain(radar.get('df'), show=False, **radar.get('kwargs'))
        radar.set_ll(ll)
        if not gage:
            radar.show()
            return radar
    if gage and radar:
        p = RadarGage(gage, radar)
        p.set_ll(ll)
        p.show()
        return p
    print ''
    print 'Returning tuple containing data dataframe and location dataframe '
    print '(once these are cleaned, initialize Rain directly with required kwargs: '
    print 'ngages, units, per_hour, freq)'
    return (df, ll)   

from common import *
from event import *
from rain import *
import xarray as xr


def to_netcdf(Rain_gage=None, Rain_radar=None, out_file='{site}.nc', path='', site='City', station_names=None, **kwargs):
    
    def __do_ll(Rain, ds0):
        ll = ['lat', 'lon']
        if type(station_names) is list:
            ll.extend(station_names)
        f = Rain.ll.loc[:, ll]
        f = f.reset_index(range(len(f.index)))
        f.index.name='station'
        f = f.loc[:, ll]
        ds1 = xr.Dataset.from_dataframe(f)
        ds_ = ds0.merge(ds1)
        return ds_, f
    
    def __do_rate(Rain, name, standard_name):
        Rain.rate.index.name = 'time'
        datasets = [xr.DataArray(Rain.rate[i]/Rain.per_hour) for i in Rain.rate.columns]
        combined = xr.concat(datasets, 'station')
        ds0 = combined.to_dataset(name=name)
        ds0[name].attrs.update({'units': Rain.units, 'standard_name': standard_name, 'coordinates': 'lat lon'})
        ds0[name].encoding = {'chunksizes': (5, 100000), 'zlib': True}
        return ds0
    
    if Rain_gage:
        ds0 = __do_rate(Rain_gage, name='rain_gage', standard_name='gage rain depth')      
        ds_, f = __do_ll(Rain_gage, ds0)
        
    if Rain_radar:
        ds0 = __do_rate(Rain_radar, name='rain_radar', standard_name='radar rain depth') 
        if not ds_:
            ds_, f = __do_ll(Rain_radar, ds0)
        else:
            ds_ = ds_.merge(ds0)

    if type(station_names) is pd.DataFrame:
        if 'station_name' not in station_names.columns:
            station_names.index.name = 'station_name'
            station_names = station_names.reset_index(range(len(station_names.index)))
        f['station'] = f.index
        f = f.merge(station_names, how='outer', on=['lat','lon'])
        f = f.reset_index(f['station'])
        f.index.name = 'station'
        f = f.drop(['lat','lon', 'station', 'index'], axis=1)
        ds1 = xr.Dataset.from_dataframe(f)
        ds_ = ds_.merge(ds1)
        ds_.station_name.attrs.update({'long_name': 'station name', 'cf_role':'timeseries_id'})
        
    ds_.lat.attrs.update({'standard_name': 'latitude', 'long_name':'station latitude', 'units': 'degrees_north'})
    ds_.lon.attrs.update({'standard_name': 'longitude', 'long_name':'station longitude', 'units': 'degrees_east'})
    ds_.time.encoding = {'units':'minutes since 1970-01-01', 'calendar':'gregorian', 'dtype': np.double}

    ds_.attrs.update({'description': '{site} rain gage network'.format(site=site),
                      'history': 'Created {now}'.format(now=pd.datetime.now()),
                      'Conventions': "CF-1.6",
                      'featureType': 'timeSeries'})

    ds_.to_netcdf(path=path+out_file.format(site=site), format='netCDF4', engine='h5netcdf')
    ds_.close()
    
    
def read_netcdf(nc_file, path=''):
    
    ds = xr.open_dataset(path+nc_file, decode_coords=False)
    print ds
    
    gage=None
    radar=None
    
    vars = ds.data_vars.keys()
    for var in vars:
        if ds.data_vars[var].ndim == 2:
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
    if 'latitiude' in ll.columns:
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

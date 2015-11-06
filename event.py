from raingage import *
import matplotlib as mpl

class Event:
    '''
    Event class to facilitate spacial exploration of particular times
    Events are transposes of RainGage objects
    TODO: make this more about different times
    '''
    def __init__(self, gage=None, radar=None, date=None):
        gage.get_RG_lon_lat()
        self.gage = gage
        self.radar = radar
        self.date = date
        self.df = gage.RG_lon_lat.join(pd.DataFrame({'gage': gage.rate[date].mean()}))
        self.df = self.df.join(pd.DataFrame({'radar': radar.rate[date].mean()}))

    def map_event(self):
        g = self.df.gage.max()-self.df.gage.min()
        r = self.df.radar.max()-self.df.radar.min()
        div = max(g,r)

        gage_color = (self.df.gage.max()/div-(self.df.gage/div)).apply(str)
        radar_color = (self.df.radar.max()/div-(self.df.radar/div)).apply(str)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6), sharey=True)
        ax1.scatter(self.df.lon, self.df.lat, c=gage_color, s=100)
        ax1.set_title('Gage')
        ax2.scatter(self.df.lon, self.df.lat, c=radar_color, s=100)
        ax2.set_title('Radar')

        fig = plt.figure(figsize=(14, 2))
        ax2 = fig.add_axes([1, 1, .9, 0.2])
        ax3 = fig.add_axes([1, 1, .9, 0.1])

        cmap = mpl.cm.gray_r
        norm3 = mpl.colors.Normalize(vmin=self.df.radar.min(), vmax=self.df.radar.max())
        x = self.df.gage.max()-self.df.radar.max()+self.df.radar.min()
        norm2 = mpl.colors.Normalize(vmin=x, vmax=self.df.gage.max())

        cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,
                                        norm=norm3,ticklocation='bottom',
                                        orientation='horizontal', label='radar')
        cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                        norm=norm2,ticklocation='top',
                                        orientation='horizontal', label = 'gage')
        plt.show()
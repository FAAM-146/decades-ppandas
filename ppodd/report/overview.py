import datetime
import os
import subprocess
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from netCDF4 import Dataset

sys.path.append(os.path.join(os.path.expanduser('~'), 'vcs/gribgrab'))
from gribgrab import GFSDownloader


def get_data(cycle, horizon, wgrib2='wgrib2'):
    downloader = GFSDownloader(cycle, horizon=horizon, resolution=0.25)
    downloader.add_regex('.*:HGT:500 mb.*')
    downloader.add_regex('.*PRATE:surface:[0-9] hour.*')
    downloader.add_regex('.*TCDC:entire atmosphere:.*')
    downloader.add_regex('.*MSLET:mean sea level:.*')
    downloader.download(filename='gfs.grb')

    subprocess.run([wgrib2, 'gfs.grb', '-netcdf', 'gfs.nc'])


def pan_data(ncfile):
    with Dataset(ncfile, 'a') as nc:
        _vars = nc.variables

        for var in _vars:
            if len(nc[var][:].shape) != 3:
                continue

            half_lon = int(nc[var][:].shape[2] / 2)

            temp = np.empty(nc[var][:].shape)
            temp[:, :, half_lon:] = nc[var][:, :, :half_lon]
            temp[:, :, :half_lon] = nc[var][:, :, half_lon:]
            nc[var][:] = temp

        temp = np.empty(nc['longitude'][:].shape)
        temp[:half_lon] = nc['longitude'][half_lon:]
        temp[half_lon:] = nc['longitude'][:half_lon]
        temp[temp>=180] = temp[temp>=180]-360
        nc['longitude'][:] = temp


def nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def add_text(ax, meta):

    ax.text(0, .6, 'Flight: {}'.format(meta['flight_number']))
    ax.text(0, .4, 'Project: {}'.format(meta['project']))

    ax.text(0, 0, 'Depart: {}'.format(meta['takeoff']['location']))
    ax.text(0, -.2, 'Arrive: {}'.format(meta['landing']['location']))
    ax.text(0, -.4, 'Takeoff: {}'.format(meta['takeoff']['time_utc']))
    ax.text(0, -.6, 'Land: {}'.format(meta['landing']['time_utc']))

    ax.text(.5, .6, 'Lat max: {0:0.2f} °N'.format(meta['domain']['lats'][1]))
    ax.text(.5, .4, 'Lat min: {0:0.2f} °N'.format(meta['domain']['lats'][0]))
    ax.text(.5, .2, 'Lon max: {0:0.2f} °E'.format(meta['domain']['lons'][1]))
    ax.text(.5, 0, 'Lon max: {0:0.2f} °E'.format(meta['domain']['lons'][0]))
    ax.text(.5, -.4, 'Ceiling: {0:d} m'.format(meta['domain']['max_alt']))


def make_plot(flight_data, outfile='overview.pdf', meta=None):

    if meta is None:
        meta = {}

    meta['domain'] = {}
    meta['domain']['lats'] = [flight_data.lat.min(), flight_data.lat.max()]
    meta['domain']['lons'] = [flight_data.lon.min(), flight_data.lon.max()]
    meta['domain']['max_alt'] = int(flight_data.alt.max())

    LAT_EXTENT_MIN = 10
    LON_EXTENT_MIN = 10
    ASPECT_RATIO = 1.2

    lon_min = flight_data.lon.min() - 1
    lon_max = flight_data.lon.max() + 1
    lat_min = flight_data.lat.min() - 1
    lat_max = flight_data.lat.max() + 1

    if lat_max - lat_min < LAT_EXTENT_MIN:
        delta = (LAT_EXTENT_MIN - (lat_max - lat_min)) / 2
        lat_min = lat_min - delta
        lat_max = lat_max + delta

    if lon_max - lon_min < LON_EXTENT_MIN:
        delta = (LON_EXTENT_MIN - (lon_max - lon_min)) / 2
        lon_min = lon_min - delta
        lon_max = lon_max + delta


    aspect = (lon_max - lon_min) / (lat_max - lat_min)

    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    if(lon_range < .75 * lat_range):
        delta = (lat_range - lon_range) / 2
        lon_min -= delta
        lon_max += delta

    if(lat_range < .75 * lon_range):
        delta = (lon_range - lat_range) / 2
        lat_min -= delta
        lat_max += delta

    flight_time = flight_data.index[0] - datetime.timedelta(hours=2)

    cycle = flight_time.replace(minute=0, second=0, microsecond=0)

    flight_hour = flight_time.hour
    cycle_hour = flight_hour - flight_hour % 6
    cycle = cycle.replace(hour=cycle_hour)

    horizon = (flight_hour - cycle_hour) + 1

    get_data(cycle, horizon)

    if lon_max % 360 < lon_min % 360:
        pan_data('gfs.nc')
    else:
        lon_max %= 360
        lon_min %= 360

    OCEAN = cfeature.NaturalEarthFeature(
        'physical', 'ocean', '50m', edgecolor='face',
        facecolor=cfeature.COLORS['water']
    )

    fig = plt.figure(figsize=(8, 8*1.414))

    ax = fig.add_axes([.1, .5, .8, .4], projection=ccrs.PlateCarree())
    ax2 = fig.add_axes([.1, .25, .8, .2])
    hours = mdates.HourLocator()
    minutes = mdates.MinuteLocator(byminute=range(10, 60, 10))
    hours_formatter = mdates.DateFormatter('%Hz')
    ax2.xaxis.set_major_locator(hours)
    ax2.xaxis.set_major_formatter(hours_formatter)
    ax2.xaxis.set_minor_locator(minutes)

    ax3 = fig.add_axes([.1, .1, .8, .1])
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.axis('off')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(OCEAN)
    ax.add_feature(cfeature.LAND)
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    with Dataset('gfs.nc', 'r') as nc:
        lat = nc['latitude'][:]
        lon = nc['longitude'][:]

        lati_min = nearest(lat, lat_min) - 1
        loni_min = nearest(lon, lon_min) - 1
        lati_max = nearest(lat, lat_max) + 1
        loni_max = nearest(lon, lon_max) + 1

        lonr = slice(loni_min, loni_max+1)
        latr = slice(lati_min, lati_max+1)

        cloud = nc['TCDC_entireatmosphere'][-1, latr, lonr]
        rain = nc['PRATE_surface'][-1, latr, lonr]
        mslp = nc['MSLET_meansealevel'][-1, latr, lonr] / 100
        h500 = nc['HGT_500mb'][-1, latr, lonr]

        lat = lat[latr]
        lon = lon[lonr]


    for i in range(0, 100, 10):
        try:
            ax.contourf(lon, lat, cloud, transform=ccrs.PlateCarree(),
                        levels=[i,i+10], alpha=i/100, zorder=990,
                        colors=['gray'])
        except Exception:
            break

    rain_levs = np.arange(0, 0.003, 0.0003)
    for i, n in enumerate(rain_levs[:-1]):
        try:
            ax.contourf(
                lon, lat, rain, transform=ccrs.PlateCarree(),
                levels=[rain_levs[i], rain_levs[i+1]], vmin=0,
                vmax=np.max(rain_levs), alpha=min([.5, i/len(rain_levs)]),
                zorder=991
            )
        except:
            continue

    try:
        ax.contourf(
            lon, lat, rain, transform=ccrs.PlateCarree(),
            levels=[rain_levs[-1], 1], vmin=0,
            vmax=np.max(rain_levs), zorder=991, extend='above'
        )
    except:
        pass


    cs = ax.contour(lon, lat, mslp, levels=range(920, 1040, 4), colors=('k',),
                   linewidths=1, zorder=992)
    ax.clabel(cs, inline=1, fontsize=10, fmt='%d')

    cs2 = ax.contour(lon, lat, h500, levels=range(4000, 6000, 40), colors=('k',),
                     linewidths=1, linestyles='--', zorder=993)


    ax.clabel(cs2, inline=1, fontsize=10, fmt='%d')

    ax.set_aspect('auto')

    ax.set_title('Synoptic Overview (GFS {cycle} T+{tplus})'.format(
        cycle=cycle.strftime('%Y%m%d %Hz'),
        tplus=horizon
    ))

    ax.plot(flight_data.lon, flight_data.lat, transform=ccrs.PlateCarree(),
            color=[.7, 0, 0], zorder=999)

    ax2.plot(flight_data.alt, linewidth=3, color=[.7, 0, 0])
    ax2.set_title('Altitude (m)')

    add_text(ax3, meta)
    fig.savefig(outfile)

    os.remove('gfs.grb')
    os.remove('gfs.nc')

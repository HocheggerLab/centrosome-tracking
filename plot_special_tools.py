import itertools
import logging
import math
import os

import h5py
import matplotlib.axes
import matplotlib.colors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage.external.tifffile as tf
from PIL import Image
from PyQt4 import Qt, QtGui
from PyQt4.QtCore import QRect
from PyQt4.QtGui import QBrush, QColor, QFont, QPainter, QPen
from matplotlib.patches import Arc
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import axes3d
import matplotlib.colors as colors
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import parameters
from imagej_pandas import ImagejPandas

logger = logging.getLogger(__name__)

# sussex colors
SUSSEX_FLINT = colors.to_rgb('#013035')
SUSSEX_COBALT_BLUE = colors.to_rgb('#1E428A')
SUSSEX_MID_GREY = colors.to_rgb('#94A596')
SUSSEX_FUSCHIA_PINK = colors.to_rgb('#EB6BB0')
SUSSEX_CORAL_RED = colors.to_rgb('#DF465A')
SUSSEX_TURQUOISE = colors.to_rgb('#00AFAA')
SUSSEX_WARM_GREY = colors.to_rgb('#D6D2C4')
SUSSEX_SUNSHINE_YELLOW = colors.to_rgb('#FFB81C')
SUSSEX_BURNT_ORANGE = colors.to_rgb('#DC582A')
SUSSEX_SKY_BLUE = colors.to_rgb('#40B4E5')

SUSSEX_NAVY_BLUE = colors.to_rgb('#1B365D')
SUSSEX_CHINA_ROSE = colors.to_rgb('#C284A3')
SUSSEX_POWDER_BLUE = colors.to_rgb('#7DA1C4')
SUSSEX_GRAPE = colors.to_rgb('#5D3754')
SUSSEX_CORN_YELLOW = colors.to_rgb('#F2C75C')
SUSSEX_COOL_GREY = colors.to_rgb('#D0D3D4')
SUSSEX_DEEP_AQUAMARINE = colors.to_rgb('#487A7B')


# SUSSEX_NEON_BLUE=''
# SUSSEX_NEON_BRIGHT_ORANGE=''
# SUSSEX_NEON_GREEN=''
# SUSSEX_NEON_LIGHT_ORANGE=''
# SUSSEX_NEON_YELLOW=''
# SUSSEX_NEON_SALMON=''
# SUSSEX_NEON_PINK=''

class colors():
    alexa_488 = [.29, 1., 0]
    alexa_594 = [1., .61, 0]
    alexa_647 = [.83, .28, .28]
    hoechst_33342 = [0, .57, 1.]
    red = [1, 0, 0]
    green = [0, 1, 0]
    blue = [0, 0, 1]
    sussex_flint = colors.to_rgb('#013035')
    sussex_cobalt_blue = colors.to_rgb('#1e428a')
    sussex_mid_grey = colors.to_rgb('#94a596')
    sussex_fuschia_pink = colors.to_rgb('#eb6bb0')
    sussex_coral_red = colors.to_rgb('#df465a')
    sussex_turquoise = colors.to_rgb('#00afaa')
    sussex_warm_grey = colors.to_rgb('#d6d2c4')
    sussex_sunshine_yellow = colors.to_rgb('#ffb81c')
    sussex_burnt_orange = colors.to_rgb('#dc582a')
    sussex_sky_blue = colors.to_rgb('#40b4e5')

    sussex_navy_blue = colors.to_rgb('#1b365d')
    sussex_china_rose = colors.to_rgb('#c284a3')
    sussex_powder_blue = colors.to_rgb('#7da1c4')
    sussex_grape = colors.to_rgb('#5d3754')
    sussex_corn_yellow = colors.to_rgb('#f2c75c')
    sussex_cool_grey = colors.to_rgb('#d0d3d4')
    sussex_deep_aquamarine = colors.to_rgb('#487a7b')


class MyAxes3D(axes3d.Axes3D):
    def __init__(self, baseObject, sides_to_draw):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.sides_to_draw = list(sides_to_draw)
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        for t in self.w_zaxis.get_ticklines() + self.w_zaxis.get_ticklabels():
            t.set_visible(visible)
        self.w_zaxis.line.set_visible(visible)
        self.w_zaxis.pane.set_visible(visible)
        self.w_zaxis.label.set_visible(visible)

    def draw(self, renderer):
        # set visibility of some features False
        self.set_some_features_visibility(False)
        # draw the axes
        super(MyAxes3D, self).draw(renderer)
        # set visibility of some features True.
        # This could be adapted to set your features to desired visibility,
        # e.g. storing the previous values and restoring the values
        self.set_some_features_visibility(True)

        zaxis = self.zaxis
        draw_grid_old = zaxis.axes._draw_grid
        # disable draw grid
        zaxis.axes._draw_grid = False

        tmp_planes = zaxis._PLANES

        if 'l' in self.sides_to_draw:
            # draw zaxis on the left side
            zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)
        if 'r' in self.sides_to_draw:
            # draw zaxis on the right side
            zaxis._PLANES = (tmp_planes[3], tmp_planes[2],
                             tmp_planes[1], tmp_planes[0],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)

        zaxis._PLANES = tmp_planes

        # disable draw grid
        zaxis.axes._draw_grid = draw_grid_old


def set_axis_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def _compute_congression(cg):
    # compute congression signal
    cg['cgr'] = 0
    _cg = pd.DataFrame()
    for id, idf in cg.groupby(ImagejPandas.NUCLEI_INDIV_INDEX):
        time_of_c, frame_of_c, dist_of_c = ImagejPandas.get_contact_time(idf, ImagejPandas.DIST_THRESHOLD)
        if frame_of_c is not None and frame_of_c > 0:
            idf.loc[idf['Frame'] >= frame_of_c, 'cgr'] = 1
        _cg = _cg.append(idf)
    cg = _cg

    cg = cg[cg['CentrLabel'] == 'A']

    dfout = pd.DataFrame()
    for id, cdf in cg.groupby('condition'):
        total_centrosome_pairs = float(len(cdf['indiv'].unique()))
        cdf = cdf.set_index(['indiv', 'Time']).sort_index()
        cgr1_p = cdf['cgr'].unstack('indiv').fillna(method='ffill').sum(axis=1) / total_centrosome_pairs * 100.0
        cgr1_p = cgr1_p.reset_index().rename(index=str, columns={0: 'congress'})
        cgr1_p['condition'] = id
        dfout = dfout.append(cgr1_p)

    return dfout


def congression(cg, ax=None, order=None, linestyles=None):
    # plot centrosome congression as %
    # get congression signal
    cgs = _compute_congression(cg)
    palette = itertools.cycle(sns.color_palette())

    ax = ax if ax is not None else plt.gca()
    order = order if order is not None else cg['condition'].unique()

    dhandles, dlabels = list(), list()
    for id in order:
        cdf = cgs[cgs['condition'] == id]
        # PLOT centrosome congresion
        _color = matplotlib.colors.to_hex(next(palette))
        cgri = cdf.set_index('Time').sort_index()
        ls = linestyles[id] if linestyles is not None else None
        cgri.plot(y='congress', drawstyle='steps-pre', color=_color, linestyle=ls, lw=1, ax=ax)
        dlbl = '%s' % (id)
        dhandles.append(mlines.Line2D([], [], color=_color, linestyle=ls, marker=None, label=dlbl))
        dlabels.append(dlbl)

    _xticks = range(0, int(cgs['Time'].max()), 20)
    ax.set_xticks(_xticks)
    ax.set_xlabel('Time $[min]$')
    ax.set_ylabel('Congression in percentage ($d<%0.2f$ $[\mu m]$)' % ImagejPandas.DIST_THRESHOLD)
    ax.legend(dhandles, dlabels, loc='upper left')


def ribbon(df, ax, ribbon_width=0.75, n_indiv=8, indiv_cols=range(8), z_max=None):
    right_axes_class = (str(type(ax)) == "<class 'matplotlib.axes._subplots.Axes3DSubplot'>") or \
                       (str(type(ax)) == "<class 'plot_special_tools.Axes3DSubplot'>")

    if not right_axes_class:
        raise Exception('Not the right axes class for ribbon plot.')
    if df['condition'].unique().size > 1:
        raise Exception('Ribbon plot needs just one condition.')
    if len(indiv_cols) != n_indiv:
        if len(indiv_cols) != 8:
            raise Exception('Number of individuals and pick must match.')
        else:
            indiv_cols = range(n_indiv)
    # extract data
    df = df[df['CentrLabel'] == 'A']
    time_series = sorted(df['Time'].unique())
    df = df.set_index(['Time', 'indv']).sort_index()
    dmat = df['DistCentr'].unstack('indv').as_matrix()
    x = np.array(time_series)
    y = np.linspace(1, n_indiv, n_indiv)
    z = dmat[:, indiv_cols]

    numPts = x.shape[0]
    numSets = y.shape[0]
    # print x.shape, y.shape, z.shape, np.max(np.nan_to_num(z))

    # create facet color matrix
    _time_color_grad = sns.color_palette('coolwarm', len(time_series))
    _colors = np.empty((len(time_series), len(time_series)), dtype=tuple)
    for cy in range(len(time_series)):
        for cx in range(len(time_series)):
            _colors[cx, cy] = _time_color_grad[cx]

    zmax = z_max if z_max is not None else df['DistCentr'].max()

    # plot each "ribbon" as a surface plot with a certain width
    for i in np.arange(0, numSets):
        X = np.vstack((x, x)).T
        Y = np.ones((numPts, 2)) * i
        Y[:, 1] = Y[:, 0] + ribbon_width
        Z = np.vstack((z[:, i], z[:, i])).T
        surf = ax.plot_surface(X, Y, Z, vmax=zmax, rstride=1, cstride=1, facecolors=_colors,
                               edgecolors='k', alpha=0.8, linewidth=0.25)

    ax.set_facecolor('white')

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(df['condition'].unique()[0])

    ax.set_xlabel('Time $[min]$', labelpad=20)
    ax.set_ylabel('Track', labelpad=15)
    ax.set_zlabel('Distance between centrosomes $[\mu m]$', labelpad=10)

    ax.set_ylim((0, numSets + 1))

    xticks = np.arange(0, np.max(time_series), 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['%d' % t for t in xticks])

    yticks = np.arange(1, n_indiv, 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(['%d' % t for t in yticks])

    zticks = np.arange(0, zmax, 10)
    ax.set_zlim3d(0, zmax)
    ax.set_zticks(zticks)
    ax.set_zticklabels(['%d' % t for t in zticks])


def msd_indivs(df, ax, time='Time', ylim=None):
    if df.empty:
        raise Exception('Need non-empty dataframe..')
    if df['condition'].unique().size > 1:
        raise Exception('Need just one condition for using this plotting function.')

    _err_kws = {'alpha': 0.5, 'lw': 0.1}
    cond = df['condition'].unique()[0]
    df_msd = ImagejPandas.msd_particles(df)
    df_msd = _msd_tag(df_msd)

    sns.tsplot(
        data=df_msd[df_msd['condition'] == cond], lw=3,
        err_style=['unit_traces'], err_kws=_err_kws,
        time=time, value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax)
    ax.set_title(cond)
    ax.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
    ax.legend(title=None, loc='upper left')
    if time == 'Frame':
        ax.set_xlabel('Time delay $[frames]$')
        ax.set_xticks(range(0, df['Frame'].max(), 5))
        ax.set_xlim([0, df['Frame'].max()])
    else:
        ax.set_xlabel('Time delay $[min]$')
    if ylim is not None:
        ax.set_ylim(ylim)


def msd(df, ax, time='Time', ylim=None, color='k'):
    if df.empty:
        raise Exception('Need non-empty dataframe..')
    if df['condition'].unique().size > 1:
        raise Exception('Need just one condition for using this plotting function.')

    cond = df['condition'].unique()[0]
    df_msd = ImagejPandas.msd_particles(df)
    df_msd = _msd_tag(df_msd)

    sns.tsplot(data=df_msd[df_msd['msd_cat'] == cond + ' displacing more'],
               color=color, linestyle='-',
               time=time, value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax)
    sns.tsplot(data=df_msd[df_msd['msd_cat'] == cond + ' displacing less'],
               color=color, linestyle='--',
               time=time, value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax)
    ax.set_title(cond)
    ax.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
    ax.set_xticks(np.arange(0, df_msd['Time'].max(), 20.0))
    ax.legend(title=None, loc='upper left')
    if time == 'Frame':
        ax.set_xlabel('Time delay $[frames]$')
        ax.set_xticks(range(0, df['Frame'].max(), 5))
        ax.set_xlim([0, df['Frame'].max()])
    else:
        ax.set_xlabel('Time delay $[min]$')
    if ylim is not None:
        ax.set_ylim(ylim)


def distance_to_nuclei_center(df, ax, mask=None, time_contact=None, plot_interp=False):
    pal = sns.color_palette()
    nucleus_id = df['Nuclei'].min()

    dhandles, dlabels = list(), list()
    for k, [(centr_lbl), _df] in enumerate(df.groupby(['Centrosome'])):
        track = _df.set_index('Time').sort_index()
        color = pal[k % len(pal)]
        dlbl = 'N%d-C%d' % (nucleus_id, centr_lbl)
        dhandles.append(mlines.Line2D([], [], color=color, marker=None, label=dlbl))
        dlabels.append(dlbl)

        if mask is not None and not mask.empty:
            tmask = mask[mask['Centrosome'] == centr_lbl].set_index('Time').sort_index()
            orig = track.loc[tmask.index, 'Dist'][tmask['Dist']]
            if len(orig) > 0:
                orig.plot(ax=ax, label='Original', marker='o', markersize=3, linewidth=0, c=color)
            if plot_interp:
                interp = track['Dist'][~tmask['Dist']]
                if len(interp) > 0:
                    interp.plot(ax=ax, label='Interpolated', marker='<', linewidth=0, c=color)
                track['Dist'].plot(ax=ax, label=dlbl, marker=None, sharex=True, c=color)
            else:
                orig.plot(ax=ax, label=dlbl, marker=None, sharex=True, c=color)
        else:
            print('plotting distance to nuclei with no mask.')
            track['Dist'].plot(ax=ax, label=dlbl, marker=None, sharex=True, c=color)

    # plot time of contact
    if time_contact is not None:
        ax.axvline(x=time_contact, color='dimgray', linestyle='--')
        ax.axvline(x=time_contact - ImagejPandas.TIME_BEFORE_CONTACT, color='lightgray', linestyle='--')

    ax.legend(dhandles, dlabels, loc='upper right')
    ax.set_ylabel('Distance to\nnuclei center $[\mu m]$')


def distance_to_cell_center(df, ax, mask=None, time_contact=None, plot_interp=False):
    pal = sns.color_palette()
    nucleus_id = df['Nuclei'].min()

    dhandles, dlabels = list(), list()
    for k, [(centr_lbl), _df] in enumerate(df.groupby(['Centrosome'])):
        track = _df.set_index('Time').sort_index()
        color = pal[k % len(pal)]
        dlbl = 'N%d-C%d' % (nucleus_id, centr_lbl)
        dhandles.append(mlines.Line2D([], [], color=color, marker=None, label=dlbl))
        dlabels.append(dlbl)

        if mask is not None and not mask.empty:
            tmask = mask[mask['Centrosome'] == centr_lbl].set_index('Time').sort_index()
            orig = track.loc[tmask.index, 'DistCell'][tmask['DistCell']]
            if len(orig) > 0:
                orig.plot(ax=ax, label='Original', marker='o', markersize=3, linewidth=0, c=color)
            if plot_interp:
                interp = track['DistCell'][~tmask['DistCell']]
                if len(interp) > 0:
                    interp.plot(ax=ax, label='Interpolated', marker='<', linewidth=0, c=color)
                track['Dist'].plot(ax=ax, label=dlbl, marker=None, sharex=True, c=color)
            else:
                orig.plot(ax=ax, label=dlbl, marker=None, sharex=True, c=color)
        else:
            print('plotting distance to cell center with no mask.')
            track['DistCell'].plot(ax=ax, label=dlbl, marker=None, sharex=True, c=color)

    # plot time of contact
    if time_contact is not None:
        ax.axvline(x=time_contact, color='dimgray', linestyle='--')
        ax.axvline(x=time_contact - ImagejPandas.TIME_BEFORE_CONTACT, color='lightgray', linestyle='--')

    ax.legend(dhandles, dlabels, loc='upper right')
    ax.set_ylabel('Distance to\ncell center $[\mu m]$')


def distance_between_centrosomes(df, ax, mask=None, time_contact=None):
    color = sns.color_palette()[0]
    track = df.set_index('Time').sort_index()

    if mask is not None and not mask.empty:
        tmask = mask[mask['CentrLabel'] == 'A'].set_index('Time').sort_index()
        orig = track['DistCentr'][tmask['DistCentr']]
        interp = track['DistCentr'][~tmask['DistCentr']]
        if len(orig) > 0:
            orig.plot(ax=ax, label='Original', marker='o', markersize=3, linewidth=0, c=color)
        if len(interp) > 0:
            interp.plot(ax=ax, label='Interpolated', marker='<', linewidth=0, c=color)
    else:
        print('plotting distance between centrosomes with no mask.')
    track['DistCentr'].plot(ax=ax, marker=None, sharex=True, c=color)

    # plot time of contact
    if time_contact is not None:
        ax.axvline(x=time_contact, color='dimgray', linestyle='--')
        ax.axvline(x=time_contact - ImagejPandas.TIME_BEFORE_CONTACT, color='lightgray', linestyle='--')

    ax.set_ylabel('Distance between\ncentrosomes $[\mu m]$')
    ax.set_ylim([0, max(ax.get_ylim())])


def speed_to_nucleus(df, ax, mask=None, time_contact=None):
    pal = sns.color_palette()
    nucleus_id = df['Nuclei'].min()

    dhandles, dlabels = list(), list()
    for k, [(lbl_centr), _df] in enumerate(df.groupby(['Centrosome'])):
        track = _df.set_index('Time').sort_index()
        color = pal[k % len(pal)]
        dlbl = 'N%d-C%d' % (nucleus_id, lbl_centr)
        dhandles.append(mlines.Line2D([], [], color=color, marker=None, label=dlbl))
        dlabels.append(dlbl)

        if mask is not None and not mask.empty:
            tmask = mask[mask['Centrosome'] == lbl_centr].set_index('Time').sort_index()
            orig = track.loc[tmask.index, 'Speed'][tmask['Speed']]
            interp = track['Speed'][~tmask['Speed']]
            if len(orig) > 0:
                orig.plot(ax=ax, label='Original', marker='o', markersize=3, linewidth=0, c=color)
            if len(interp) > 0:
                interp.plot(ax=ax, label='Interpolated', marker='<', linewidth=0, c=color)
        else:
            print('plotting speed to nuclei with no mask.')
        track['Speed'].plot(ax=ax, marker=None, sharex=True, c=color)

    # plot time of contact
    if time_contact is not None:
        ax.axvline(x=time_contact, color='dimgray', linestyle='--')
        ax.axvline(x=time_contact - ImagejPandas.TIME_BEFORE_CONTACT, color='lightgray', linestyle='--')

    ax.legend(dhandles, dlabels, loc='upper right')
    ax.set_ylabel('Speed to\nnuclei $\\left[\\frac{\mu m}{min} \\right]$')


def speed_between_centrosomes(df, ax, mask=None, time_contact=None):
    color = sns.color_palette()[0]
    track = df.set_index('Time').sort_index()

    if mask is not None and not mask.empty:
        tmask = mask[mask['CentrLabel'] == 'A'].set_index('Time').sort_index()
        orig = track['SpeedCentr'][tmask['SpeedCentr']]
        interp = track['SpeedCentr'][~tmask['SpeedCentr']]
        if len(orig) > 0:
            orig.plot(ax=ax, label='Original', marker='o', markersize=3, linewidth=0, c=color)
        if len(interp) > 0:
            interp.plot(ax=ax, label='Interpolated', marker='<', linewidth=0, c=color)
    else:
        print('plotting speed between centrosomes with no mask.')
    track['SpeedCentr'].plot(ax=ax, marker=None, sharex=True, c=color)

    # plot time of contact
    if time_contact is not None:
        ax.axvline(x=time_contact, color='dimgray', linestyle='--')
        ax.axvline(x=time_contact - ImagejPandas.TIME_BEFORE_CONTACT, color='lightgray', linestyle='--')

    ax.set_ylabel('Speed between\ncentrosomes $\\left[\\frac{\mu m}{min} \\right]$')


def acceleration_to_nucleus(df, ax, mask=None, time_contact=None):
    pal = sns.color_palette()
    nucleus_id = df['Nuclei'].min()

    dhandles, dlabels = list(), list()
    for k, [(lbl_centr), _df] in enumerate(df.groupby(['Centrosome'])):
        track = _df.set_index('Time').sort_index()
        color = pal[k % len(pal)]
        dlbl = 'N%d-C%d' % (nucleus_id, lbl_centr)
        dhandles.append(mlines.Line2D([], [], color=color, marker=None, label=dlbl))
        dlabels.append(dlbl)

        if mask is not None and not mask.empty:
            tmask = mask[mask['Centrosome'] == lbl_centr].set_index('Time').sort_index()
            orig = track.loc[tmask.index, 'Acc'][tmask['Acc']]
            interp = track['Acc'][~tmask['Acc']]
            if len(orig) > 0:
                orig.plot(ax=ax, label='Original', marker='o', markersize=3, linewidth=0, c=color)
            if len(interp) > 0:
                interp.plot(ax=ax, label='Interpolated', marker='<', linewidth=0, c=color)
        else:
            print('plotting acceleration to nuclei with no mask.')
        track['Acc'].plot(ax=ax, marker=None, sharex=True, c=color)

    # plot time of contact
    if time_contact is not None:
        ax.axvline(x=time_contact, color='dimgray', linestyle='--')
        ax.axvline(x=time_contact - ImagejPandas.TIME_BEFORE_CONTACT, color='lightgray', linestyle='--')

    ax.legend(dhandles, dlabels, loc='upper right')
    ax.set_ylabel('Acceleration relative\nto nuclei $\\left[\\frac{\mu m}{min^2} \\right]$')


def plot_acceleration_between_centrosomes(df, ax, mask=None, time_contact=None):
    color = sns.color_palette()[0]
    track = df.set_index('Time').sort_index()

    if mask is not None and not mask.empty:
        tmask = mask[mask['CentrLabel'] == 'A'].set_index('Time').sort_index()
        orig = track['AccCentr'][tmask['AccCentr']]
        interp = track['AccCentr'][~tmask['AccCentr']]
        if len(orig) > 0:
            orig.plot(ax=ax, label='Original', marker='o', markersize=3, linewidth=0, c=color)
        if len(interp) > 0:
            interp.plot(ax=ax, label='Interpolated', marker='<', linewidth=0, c=color)
    else:
        print('plotting acceleration between centrosomes with no mask.')
    track['AccCentr'].plot(ax=ax, marker=None, sharex=True, c=color)

    # plot time of contact
    if time_contact is not None:
        ax.axvline(x=time_contact, color='dimgray', linestyle='--')
        ax.axvline(x=time_contact - ImagejPandas.TIME_BEFORE_CONTACT, color='lightgray', linestyle='--')

    ax.set_ylabel('Acceleration between\ncentrosomes $\\left[\\frac{\mu m}{min^2} \\right]$')


def load_tiff(path):
    _, img_name = os.path.split(path)
    with tf.TiffFile(path) as tif:
        if tif.is_imagej is not None:
            metadata = tif.pages[0].imagej_tags
            dt = metadata['finterval'] if 'finterval' in metadata else 1

            # asuming square pixels
            xr = tif.pages[0].tags['x_resolution'].value
            res = float(xr[0]) / float(xr[1])  # pixels per um
            if metadata['unit'] == 'centimeter':
                res = res / 1e4

            if os.path.exists(parameters.experiments_dir + 'eb3/eb3_calibration.xls'):
                cal = pd.read_excel(parameters.experiments_dir + 'eb3/eb3_calibration.xls')
                calp = cal[cal['filename'] == img_name]
                if not calp.empty:
                    calp = calp.iloc[0]
                    if calp['optivar'] == 'yes':
                        logging.info('file with optivar configuration selected!')
                        res *= 1.6

            frames = None
            if len(tif.pages) == 1:
                if ('slices' in metadata and metadata['slices'] > 1) or (
                        'frames' in metadata and metadata['frames'] > 1):
                    frames = tif.pages[0].asarray()
                else:
                    frames = [tif.pages[0].asarray()]
            elif len(tif.pages) > 1:
                # frames = np.ndarray((len(tif.pages), tif.pages[0].image_length, tif.pages[0].image_width), dtype=np.int32)
                frames = list()
                for i, page in enumerate(tif.pages):
                    frames.append(page.asarray())

    return frames, res, dt, metadata['frames'], metadata['channels']


def find_image(img_name, folder):
    for root, directories, filenames in os.walk(folder):
        for file in filenames:
            joinf = os.path.abspath(os.path.join(root, file))
            if os.path.isfile(joinf) and joinf[-4:] == '.tif' and file == img_name:
                return load_tiff(joinf)


def retrieve_image(image_arr, frame, channel=0, number_of_frames=1):
    nimgs = image_arr.shape[0]
    n_channels = int(nimgs / number_of_frames)
    ix = frame * n_channels + channel
    logger.debug("retrieving frame %d of channel %d (index=%d)" % (frame, channel, ix))
    return image_arr[ix]


def image_iterator(image_arr, channel=0, number_of_frames=1):
    nimgs = image_arr.shape[0]
    n_channels = int(nimgs / number_of_frames)
    for f in range(number_of_frames):
        ix = f * n_channels + channel
        logger.debug("retrieving frame %d of channel %d (index=%d)" % (f, channel, ix))
        if ix < nimgs: yield image_arr[ix]


def render_cell(df, ax, img=None, res=4.5, w=50, h=50):
    """
    Render an individual cell with all its measured features
    """
    from skimage import exposure

    # plot image
    if img is not None:
        img = exposure.equalize_hist(img)
        img = exposure.adjust_gamma(img, gamma=3)
        ax.imshow(img, cmap='gray', extent=(0, img.shape[0] / res, img.shape[1] / res, 0))
    # if img is not None: ax.imshow(img, cmap='gray')

    _ca = df[df['CentrLabel'] == 'A']
    _cb = df[df['CentrLabel'] == 'B']

    if not _ca['NuclBound'].empty:
        nucleus = Polygon(eval(_ca['NuclBound'].values[0][1:-1]))

        nuc_center = nucleus.centroid
        x, y = nucleus.exterior.xy
        ax.plot(x, y, color=SUSSEX_CORN_YELLOW, linewidth=1, solid_capstyle='round')
        ax.plot(nuc_center.x, nuc_center.y, color=SUSSEX_CORN_YELLOW, marker='+', linewidth=1, solid_capstyle='round',
                zorder=1)

    if not _ca['CellBound'].empty:
        cell = Polygon(eval(_ca['CellBound'].values[0])) if not _ca['CellBound'].empty > 0 else None

        cll_center = cell.centroid
        x, y = cell.exterior.xy
        ax.plot(x, y, color='w', linewidth=1, solid_capstyle='round')
        ax.plot(cll_center.x, cll_center.y, color='w', marker='o', linewidth=1, solid_capstyle='round')

    if not _ca['CentX'].empty:
        c_a = Point(_ca['CentX'], _ca['CentY'])
        ca = plt.Circle((c_a.x, c_a.y), radius=1, edgecolor=SUSSEX_NAVY_BLUE, facecolor='none', linewidth=2, zorder=10)
        ax.add_artist(ca)

        if not _cb['CentX'].empty:
            c_b = Point(_cb['CentX'], _cb['CentY'])
            cb = plt.Circle((c_b.x, c_b.y), radius=1, edgecolor=SUSSEX_CORAL_RED, facecolor='none', linewidth=2,
                            zorder=10)
            ax.plot((c_a.x, c_b.x), (c_a.y, c_b.y), color=SUSSEX_WARM_GREY, linewidth=1, zorder=10)
            ax.add_artist(cb)

    ax.set_xlim(c_a.x - w / 2, c_a.x + w / 2)
    ax.set_ylim(c_a.y - h / 2 + 10, c_a.y + h / 2 + 10)
    # ax.set_xlim(nuc_center.x - w / 2, nuc_center.x + w / 2)
    # ax.set_ylim(nuc_center.y - h / 2, nuc_center.y + h / 2)


def render_tracked_centrosomes(hdf5_fname, condition, run, nuclei):
    import sys
    import cv2
    app = QtGui.QApplication(sys.argv)
    with h5py.File(hdf5_fname, 'r') as f:
        if 'pandas_dataframe' not in f['%s/%s/measurements' % (condition, run)]:
            raise KeyError('No data for selected condition-run.')

        df = pd.read_hdf(hdf5_fname, key='%s/%s/measurements/pandas_dataframe' % (condition, run))
        nuclei_list = f['%s/%s/measurements/nuclei' % (condition, run)]
        centrosome_list = f['%s/%s/measurements/centrosomes' % (condition, run)]
        frames = len(f['%s/%s/raw' % (condition, run)])
        ch2 = f['%s/%s/raw/%03d/channel-2' % (condition, run, 0)]
        data = ch2[:]
        width, height = data.shape
        resolution = ch2.parent.attrs['resolution']

        if 'boundary' in f['%s/%s/processed' % (condition, run)]:
            k = '%s/%s/processed/boundary' % (condition, run)
            dfbound = pd.read_hdf(hdf5_fname, key=k)
            # dfbound = dfbound[(dfbound['Nuclei'] == nuclei) & (dfbound['condition'] == condition) & (dfbound['run'] == run)]
            dfbound = dfbound[dfbound['Nuclei'] == nuclei]
            # FIXME: this is a hack to not take boundary data for width/height estimation from a particular frame for the paper
            dfbest = dfbound[~ dfbound['Frame'].isin([8])]

            minx = miny = width
            maxx = maxy = 0
            for fr, dframe in dfbest.groupby('Frame'):
                bstr = dframe.iloc[0]['CellBound']
                if type(bstr) == str:
                    cell_boundary = np.array(eval(bstr)) * resolution
                    minx = np.minimum(minx, np.min(cell_boundary[:, 0]))
                    miny = np.minimum(miny, np.min(cell_boundary[:, 1]))
                    maxx = np.maximum(maxx, np.max(cell_boundary[:, 0]))
                    maxy = np.maximum(maxy, np.max(cell_boundary[:, 1]))
            cwidth, cheight = (maxx - minx, maxy - miny)

        # filter dataset for condition, run, and nuclei
        # df = df[(df['Nuclei'] == nuclei) & (df['condition'] == condition) & (df['run'] == run)]
        df = df[df['Nuclei'] == nuclei]

        for frame in range(frames):
            # get image frame
            ch2 = f['%s/%s/raw/%03d/channel-2' % (condition, run, frame)]
            data = ch2[:]
            # map the data range to 0 - 255
            data = ((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8)
            data = cv2.subtract(cv2.equalizeHist(data), 110)
            qtimage = QtGui.QImage(data.repeat(4), width, height, QtGui.QImage.Format_RGB32)
            image_pixmap = QtGui.QPixmap(qtimage)

            painter = QPainter()
            painter.begin(image_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            font = QFont('Arial', 25)
            painter.setFont(font)

            nuc = nuclei_list['N%02d' % nuclei]
            nfxy = nuc['pos'].value
            nuc_frames = nfxy.T[0]
            if frame in nuc_frames:
                fidx = nuc_frames.searchsorted(frame)
                nx = nfxy[fidx][1] * resolution
                ny = nfxy[fidx][2] * resolution
                df_fr = df[df['Frame'] == frame]
                sec = df_fr['Time'].iloc[0]
                min = np.floor(sec / 60.0)
                sec -= min * 60
                painter.drawText(10, 30, '%02d:%02d' % (min, sec))

                # render nuclei boundary
                if len(df_fr['NuclBound'].values) > 0:
                    cell_boundary = df_fr['NuclBound'].values[0]
                    if cell_boundary[1:-1] != '':
                        nucb_points = eval(cell_boundary[1:-1])
                        nucb_qpoints = [Qt.QPoint(x * resolution, y * resolution) for x, y in nucb_points]
                        nucb_poly = Qt.QPolygon(nucb_qpoints)

                        painter.setPen(QPen(QBrush(QColor(SUSSEX_CORN_YELLOW)), 2))
                        painter.setBrush(QColor('transparent'))

                        painter.drawPolygon(nucb_poly)

                        # render nuclei centroid
                        painter.setPen(QPen(QBrush(QColor('transparent')), 2))
                        painter.setBrush(QColor(SUSSEX_CORN_YELLOW))
                        painter.drawEllipse(nx - 5, ny - 5, 10, 10)
                        # painter.setPen(QPen(QBrush(QColor('white')), 2))
                        # painter.drawText(nx + 10, ny + 5, 'N%02d' % nuclei)

                # render cell boundary
                cellframe = dfbound.loc[dfbound['Frame'] == frame]
                if not cellframe.empty:
                    cell_bnd_str = cellframe.iloc[0]['CellBound']
                    if type(cell_bnd_str) == str:
                        cell_boundary = np.array(eval(cell_bnd_str)) * resolution
                        cell_centroid = cellframe.iloc[0][['CellX', 'CellY']].values * resolution
                        nucb_qpoints = [Qt.QPoint(x, y) for x, y in cell_boundary]
                        nucb_poly = Qt.QPolygon(nucb_qpoints)

                        painter.setBrush(QColor('transparent'))
                        painter.setPen(QPen(QBrush(QColor('white')), 2))
                        painter.drawPolygon(nucb_poly)

                        # render cell centroid
                        painter.setPen(QPen(QBrush(QColor('transparent')), 2))
                        painter.setBrush(QColor('white'))
                        painter.drawEllipse(cell_centroid[0] - 5, cell_centroid[1] - 5, 10, 10)
                        # painter.setPen(QPen(QBrush(QColor('white')), 2))
                        # painter.drawText(cell_centroid[0] + 5, cell_centroid[1], 'C%02d' % (nuclei))

                # draw centrosomes
                painter.setBrush(QColor('transparent'))
                centrosomes_of_nuclei_a = f['%s/%s/selection/%s/A' % (condition, run, 'N%02d' % nuclei)].keys()
                centrosomes_of_nuclei_b = f['%s/%s/selection/%s/B' % (condition, run, 'N%02d' % nuclei)].keys()

                # 1st centrosome
                painter.setPen(QPen(QBrush(QColor(SUSSEX_SKY_BLUE)), 2))
                for centr_str in centrosomes_of_nuclei_a:
                    cntr = centrosome_list[centr_str]
                    cfxy = cntr['pos'].value
                    cnt_frames = cfxy.T[0]
                    if frame in cnt_frames:
                        fidx = cnt_frames.searchsorted(frame)
                        cx = cfxy[fidx][1] * resolution
                        cy = cfxy[fidx][2] * resolution
                        painter.drawEllipse(cx - 5, cy - 5, 10, 10)
                # 2nd centrosome
                painter.setPen(QPen(QBrush(QColor(SUSSEX_CORAL_RED)), 2))
                for centr_str in centrosomes_of_nuclei_b:
                    cntr = centrosome_list[centr_str]
                    cfxy = cntr['pos'].value
                    cnt_frames = cfxy.T[0]
                    if frame in cnt_frames:
                        fidx = cnt_frames.searchsorted(frame)
                        cx = cfxy[fidx][1] * resolution
                        cy = cfxy[fidx][2] * resolution
                        painter.drawEllipse(cx - 5, cy - 5, 10, 10)

                rect = QRect(cell_centroid[0] - cwidth / 2.0, cell_centroid[1] - cheight / 2.0, cwidth, cheight)
                painter.setPen(QPen(QBrush(QColor('white')), 2))
                painter.drawText(rect.x() + 10, rect.y() + 30, '%02d:%02d' % (min, sec))

                xini, yini = rect.x() + cwidth - 70, rect.y() + cheight - 20
                # painter.drawText(xini + resolution * 10 / 8.0, yini - 5, '10um')
                painter.setPen(QPen(QBrush(QColor('white')), 4))
                painter.drawLine(xini, yini, xini + resolution * 10, yini)
                painter.end()
                cropped = image_pixmap.copy(rect)

                cropped.save(parameters.data_dir + 'out/%s_N%02d_F%03d.png' % (run, nuclei, frame))


def pil_grid(images, max_horiz=np.iinfo(int).max):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid


def canvas_to_pil(canvas):
    canvas.draw()
    s = canvas.tostring_rgb()
    w, h = canvas.get_width_height()[::-1]
    im = Image.frombytes("RGB", (w, h), s)
    return im


# functions for plotting angle in matplotlib
def get_angle_text(angle_plot):
    angle = angle_plot.get_label()[:-1]  # Excluding the degree symbol
    angle = "%0.2f" % float(angle) + u"\u00b0"  # Display angle upto 2 decimal places

    # Get the vertices of the angle arc
    vertices = angle_plot.get_verts()

    # Get the midpoint of the arc extremes
    x_width = (vertices[0][0] + vertices[-1][0]) / 2.0
    y_width = (vertices[0][5] + vertices[-1][6]) / 2.0

    # print x_width, y_width

    separation_radius = max(x_width / 2.0, y_width / 2.0)

    return [x_width + separation_radius, y_width + separation_radius, angle]


def get_angle_plot(line1, line2, offset=1, color=None, origin=[0, 0], len_x_axis=1, len_y_axis=1):
    l1xy = line1.get_xydata()

    # Angle between line1 and x-axis
    slope1 = (l1xy[1][1] - l1xy[0][2]) / float(l1xy[1][0] - l1xy[0][0])
    angle1 = abs(math.degrees(math.atan(slope1)))  # Taking only the positive angle

    l2xy = line2.get_xydata()

    # Angle between line2 and x-axis
    slope2 = (l2xy[1][3] - l2xy[0][4]) / float(l2xy[1][0] - l2xy[0][0])
    angle2 = abs(math.degrees(math.atan(slope2)))

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)

    angle = theta2 - theta1

    if color is None:
        color = line1.get_color()  # Uses the color of line 1 if color parameter is not passed.

    return Arc(origin, len_x_axis * offset, len_y_axis * offset, 0, theta1, theta2, color=color,
               label=str(angle) + u"\u00b0")

import itertools
import math
import os

import cv2
import matplotlib.axes
import matplotlib.colors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tf
from matplotlib.patches import Arc
from matplotlib.ticker import FormatStrFormatter, LinearLocator

from imagej_pandas import ImagejPandas


def anotated_boxplot(data_grouped, var, point_size=5, fontsize='small', stats_rotation='horizontal', cat='condition',
                     swarm=True, order=None, ax=None):
    sns.boxplot(data=data_grouped, y=var, x=cat, linewidth=0.5, width=0.2, fliersize=point_size, order=order, ax=ax,
                zorder=100)
    if swarm:
        _ax = sns.swarmplot(data=data_grouped, y=var, x=cat, size=point_size, order=order, ax=ax, zorder=10)
    else:
        _ax = sns.stripplot(data=data_grouped, y=var, x=cat, jitter=True, size=point_size, order=order, ax=ax,
                            zorder=10)
    for i, artist in enumerate(_ax.artists):
        artist.set_facecolor('None')

    order = order if order is not None else data_grouped[cat].unique()
    for x, c in enumerate(order):
        d = data_grouped[data_grouped[cat] == c][var]
        _max_y = _ax.axis()[3]
        count = d.count()
        mean = d.mean()
        median = d.median()
        if stats_rotation == 'vertical':
            _txt = '$\mu=%0.3f$  $\\tilde\mu=%0.3f$  $n=%d$' % (mean, median, count)
        else:
            _txt = '$\mu=%0.3f$\n$\\tilde\mu=%0.3f$\n$n=%d$' % (mean, median, count)

        _ax.text(x, _max_y * 0.8, _txt, rotation=stats_rotation, ha='center', va='bottom', fontsize=fontsize)
        plt.xticks(_ax.get_xticks(), rotation='vertical')


def congression(cg, ax=None, order=None):
    # plot centrosome congression as %
    # compute congression signal
    cg['cgr'] = 0
    _cg = pd.DataFrame()
    for id, idf in cg.groupby(ImagejPandas.NUCLEI_INDIV_INDEX):
        time_of_c, frame_of_c, dist_of_c = ImagejPandas.get_contact_time(idf, ImagejPandas.DIST_THRESHOLD)
        if frame_of_c > 0:
            idf.loc[idf['Frame'] >= frame_of_c, 'cgr'] = 1
        _cg = _cg.append(idf)
    cg = _cg

    cg = cg[cg['CentrLabel'] == 'A']
    palette = itertools.cycle(sns.color_palette())

    ax = ax if ax is not None else plt.gca()
    order = order if order is not None else cg['condition'].unique()

    dhandles, dlabels = list(), list()
    for id in order:
        cdf = cg[cg['condition'] == id]
        total_centrosome_pairs = float(len(cdf['indv'].unique()))
        cdf = cdf.set_index(['indv', 'Time']).sort_index()
        cgr1_p = cdf['cgr'].unstack('indv').fillna(method='ffill').sum(axis=1) / total_centrosome_pairs * 100.0
        cgr1_p = cgr1_p.reset_index().rename(index=str, columns={0: 'congress'})
        cgr1_p['condition'] = id

        # PLOT centrosome congresion
        _color = matplotlib.colors.to_hex(next(palette))
        cgri = cgr1_p.set_index('Time').sort_index()
        cgri.plot(y='congress', drawstyle='steps-pre', color=_color, lw=1, ax=ax)
        dlbl = '%s' % (id)
        dhandles.append(mlines.Line2D([], [], color=_color, marker=None, label=dlbl))
        dlabels.append(dlbl)

    _xticks = range(0, int(cg['Time'].max()), 20)
    ax.set_xticks(_xticks)
    ax.set_xlabel('Time $[min]$')
    ax.set_ylabel('Congression in percentage ($d<%0.2f$ $[\mu m]$)' % ImagejPandas.DIST_THRESHOLD)
    ax.legend(dhandles, dlabels, loc='upper left')


def ribbon(df, ax, ribbon_width=0.75, n_indiv=8, indiv_cols=range(8)):
    if str(type(ax)) != "<class 'matplotlib.axes._subplots.Axes3DSubplot'>":
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

    # plot each "ribbon" as a surface plot with a certain width
    for i in np.arange(0, numSets):
        X = np.vstack((x, x)).T
        Y = np.ones((numPts, 2)) * i
        Y[:, 1] = Y[:, 0] + ribbon_width
        Z = np.vstack((z[:, i], z[:, i])).T
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=_colors,
                               edgecolors='k', alpha=0.8, linewidth=0.25)

    ax.set_facecolor('white')
    # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(df['condition'].unique()[0])
    ax.set_xlabel('Time $[min]$')
    ax.set_ylabel('Track')
    ax.set_ylim((0, numSets))
    ax.set_zlabel('Distance between centrosomes $[\mu m]$')
    ax.set_zlim((0, np.max(np.nan_to_num(z))))
    xticks = np.arange(0, np.max(time_series), 50)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['%d' % t for t in xticks])
    yticks = np.arange(1, n_indiv)
    ax.set_yticks(yticks)
    ax.set_yticklabels(['%d' % t for t in yticks])
    # ax.get_figure().colorbar(surf, shrink=0.5, aspect=5)


def _msd_tag(df):
    mvtag = pd.DataFrame()
    for id, _df in df.groupby(ImagejPandas.NUCLEI_INDIV_INDEX):
        cond = id[0]
        c_a = _df[_df['CentrLabel'] == 'A']['msd_lfit_a'].unique()[0]
        c_b = _df[_df['CentrLabel'] == 'B']['msd_lfit_a'].unique()[0]
        if c_a > c_b:
            _df.loc[_df['CentrLabel'] == 'A', 'msd_cat'] = cond + ' moving more'
            _df.loc[_df['CentrLabel'] == 'B', 'msd_cat'] = cond + ' moving less'
        else:
            _df.loc[_df['CentrLabel'] == 'B', 'msd_cat'] = cond + ' moving more'
            _df.loc[_df['CentrLabel'] == 'A', 'msd_cat'] = cond + ' moving less'
        mvtag = mvtag.append(_df)
    return mvtag


def msd_indivs(df, ax, time='Time', ylim=None):
    if df.empty:
        raise Exception('Need non-empty dataframe..')
    if df['condition'].unique().size > 1:
        raise Exception('Need just one condition for using this plotting function.')

    _err_kws = {'alpha': 0.3, 'lw': 1}
    cond = df['condition'].unique()[0]
    df_msd = ImagejPandas.msd_centrosomes(df)
    df_msd = _msd_tag(df_msd)

    sns.tsplot(
        data=df_msd[df_msd['condition'] == cond], lw=3,
        err_style=['unit_traces'], err_kws=_err_kws,
        time=time, value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax)
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


def msd(df, ax, time='Time', ylim=None):
    if df.empty:
        raise Exception('Need non-empty dataframe..')
    if df['condition'].unique().size > 1:
        raise Exception('Need just one condition for using this plotting function.')

    cond = df['condition'].unique()[0]
    df_msd = ImagejPandas.msd_centrosomes(df)
    df_msd = _msd_tag(df_msd)

    sns.tsplot(data=df_msd[df_msd['msd_cat'] == cond + ' moving more'],
               color='k', linestyle='-',
               time=time, value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax)
    sns.tsplot(data=df_msd[df_msd['msd_cat'] == cond + ' moving less'],
               color='k', linestyle='--',
               time=time, value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax)
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


def distance_to_nucleus(df, ax, mask=None, time_contact=None, plot_interp=False):
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
            orig = track['Dist'][tmask['Dist']]
            interp = track['Dist'][~tmask['Dist']]
            if len(orig) > 0:
                orig.plot(ax=ax, label='Original', marker='o', markersize=3, linewidth=0, c=color)
            if plot_interp:
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
    ax.set_ylabel('Distance to\nnuclei $[\mu m]$')


def distance_between_centrosomes(df, ax, mask=None, time_contact=None):
    color = sns.color_palette()[0]
    track = df.set_index('Time').sort_index()

    if mask is not None and not mask.empty:
        tmask = mask.set_index('Time').sort_index()
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
            orig = track['Speed'][tmask['Speed']]
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
        tmask = mask.set_index('Time').sort_index()
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
            orig = track['Acc'][tmask['Acc']]
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
        tmask = mask.set_index('Time').sort_index()
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


def find_image(img_name, folder):
    for root, directories, filenames in os.walk(folder):
        for file in filenames:
            joinf = os.path.abspath(os.path.join(root, file))
            if os.path.isfile(joinf) and joinf[-4:] == '.tif' and file == img_name:
                image = cv2.imread(joinf)
                with tf.TiffFile(joinf, fastij=True) as tif:
                    if tif.is_imagej is not None:
                        dt = tif.pages[0].imagej_tags.finterval
                        res = 'n/a'
                        if tif.pages[0].resolution_unit == 'centimeter':
                            # asuming square pixels
                            xr = tif.pages[0].x_resolution
                            res = float(xr[0]) / float(xr[1])  # pixels per cm
                            res = res / 1e4  # pixels per um
                        elif tif.pages[0].imagej_tags.unit == 'micron':
                            # asuming square pixels
                            xr = tif.pages[0].x_resolution
                            res = float(xr[0]) / float(xr[1])  # pixels per um
                        return (image, res, dt)


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

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from imagej_pandas import ImagejPandas


def anotated_boxplot(data_grouped, var, size=5):
    sns.boxplot(data=data_grouped, y=var, x='condition', linewidth=0.5)
    ax = sns.swarmplot(data=data_grouped, y=var, x='condition', size=size)
    for i, artist in enumerate(ax.artists):
        artist.set_facecolor('None')

    cat = data_grouped['condition'].unique()
    for x, c in enumerate(cat):
        d = data_grouped[data_grouped['condition'] == c][var]
        _max_y = ax.axis()[3]
        count = d.count()
        mean = d.mean()
        median = d.median()
        ax.text(x, _max_y * 0.5, '$\mu=%0.3f$  $\\tilde\mu=%0.3f$  $n=%d$' % (mean, median, count),
                rotation='vertical', ha='center', va='bottom', fontsize='xx-small')

    plt.xticks(ax.get_xticks(), rotation='vertical')


def congression(cg):
    # plot centrosome congression as %
    # compute congression signal
    cg['cgr'] = 0
    for id, idf in cg.groupby('indv'):
        time_of_c, frame_of_c, dist_of_c = ImagejPandas.get_contact_time(idf, ImagejPandas.DIST_THRESHOLD)
        if frame_of_c > 0:
            cg.loc[(cg['CentrLabel'] == 'A') & (cg['indv'] == id) & (cg['Frame'] >= frame_of_c), 'cgr'] = 1

    cg = cg[cg['CentrLabel'] == 'A']

    cgr = pd.DataFrame()
    for id, cdf in cg.groupby('condition'):
        total_centrosome_pairs = float(len(cdf['indv'].unique()))
        cdf = cdf.set_index(['indv', 'Time']).sort_index()
        cgr1_p = cdf['cgr'].unstack('indv').fillna(method='ffill').sum(axis=1) / total_centrosome_pairs * 100.0
        cgr1_p = cgr1_p.reset_index().rename(index=str, columns={0: 'congress'})
        cgr1_p['condition'] = id
        cgr = cgr.append(cgr1_p)

    # PLOT centrosome congresion
    g = sns.FacetGrid(cgr, hue='condition', aspect=3)
    g.map(plt.plot, 'Time', 'congress', drawstyle='steps-pre', lw=1)
    g.set_axis_labels(x_var='Time $[min]$', y_var='Congression ($d<%0.2f$ $[\mu m]$)' % ImagejPandas.DIST_THRESHOLD)
    g.add_legend()


def distance_to_nucleus(df, ax, mask=None, time_contact=None):
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
            if len(interp) > 0:
                interp.plot(ax=ax, label='Interpolated', marker='<', linewidth=0, c=color)
        else:
            print 'plotting distance to nuclei with no mask.'
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
        print 'plotting distance between centrosomes with no mask.'
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
            print 'plotting speed to nuclei with no mask.'
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
        print 'plotting speed between centrosomes with no mask.'
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
            print 'plotting acceleration to nuclei with no mask.'
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
        print 'plotting acceleration between centrosomes with no mask.'
    track['AccCentr'].plot(ax=ax, marker=None, sharex=True, c=color)

    # plot time of contact
    if time_contact is not None:
        ax.axvline(x=time_contact, color='dimgray', linestyle='--')
        ax.axvline(x=time_contact - ImagejPandas.TIME_BEFORE_CONTACT, color='lightgray', linestyle='--')

    ax.set_ylabel('Acceleration between\ncentrosomes $\\left[\\frac{\mu m}{min^2} \\right]$')

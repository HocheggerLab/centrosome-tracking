import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns

from imagej_pandas import ImagejPandas


def anotated_boxplot(data_grouped, var):
    ax = sns.swarmplot(data=data_grouped, y=var, x='condition')
    cat = data_grouped['condition'].unique()
    for x, c in enumerate(cat):
        d = data_grouped[data_grouped['condition'] == c][var]
        _max_y = ax.axis()[3]
        count = d.count()
        mean = d.mean()
        ax.text(x, _max_y * 0.9, '$\mu=%0.3f$' % mean, ha='center')
        ax.text(x, _max_y * 0.7, '$n=%d$' % count, ha='center')


def plot_distance_to_nucleus(df, ax, filename=None, mask=None, time_contact=None, draw_interpolated=True):
    pal = sns.color_palette()
    nucleus_id = df['Nuclei'].min()

    dhandles, dlabels = list(), list()
    for k, [(lblCentr), _df] in enumerate(df.groupby(['Centrosome'])):
        track = _df.set_index('Frame').sort_index()

        color = pal[k % len(pal)]
        dlbl = 'N%d-C%d' % (nucleus_id, lblCentr)
        dhandles.append(mlines.Line2D([], [], color=color, marker=None, label=dlbl))
        dlabels.append(dlbl)

        if mask is not None and not mask.empty:
            tmask = mask[mask['Centrosome'] == lblCentr].set_index('Frame').sort_index()
            orig = track['Dist'][tmask['Dist']]
            interp = track['Dist'][~tmask['Dist']]
            if len(orig) > 0:
                orig.plot(ax=ax, label='Original', marker='o', markersize=3, linewidth=0, c=color)
                if not draw_interpolated:
                    orig.plot(ax=ax, label=dlbl, marker=None, sharex=True, c=color)
            if len(interp) > 0 and draw_interpolated:
                interp.plot(ax=ax, label='Interpolated', marker='<', linewidth=0, c=color)
            if draw_interpolated:
                track['Dist'].plot(ax=ax, label=dlbl, marker=None, sharex=True, c=color)
        else:
            print 'plotting with no mask.'
            track['Dist'].plot(ax=ax, label=dlbl, marker=None, sharex=True, c=color)

    # plot time of contact
    if time_contact is not None:
        ax.axvline(x=time_contact, color='dimgray', linestyle='--')
        ax.axvline(x=time_contact - ImagejPandas.TIME_BEFORE_CONTACT, color='lightgray',
                   linestyle='--')

    ax.legend(dhandles, dlabels)
    ax.set_ylabel('Dist to Nuclei $[\mu m]$')
    ax.set_xlabel('Time $[min]$')

    if filename is not None:
        plt.axes(ax).savefig(filename, format='svg')

from collections import OrderedDict

import matplotlib
import matplotlib.axes
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

import special_plots as sp
from imagej_pandas import ImagejPandas

print font_manager.OSXInstalledFonts()
print font_manager.OSXFontDirectories

matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('svg', fonttype='none')
sns.set(context='paper', style='whitegrid', font='Arial', font_scale=0.9)
pd.set_option('display.width', 320)
plt.style.use('bmh')

names = OrderedDict([('mother-daughter', 'Centrosomes')])
col_palette = [sns.xkcd_rgb["windows blue"]]
cond_colors = dict(zip(names.keys(), col_palette))
_fig_size_A3 = (11.7, 16.5)
_err_kws = {'alpha': 0.3, 'lw': 1}
msd_ylim = [0, 420]


def rename_conditions(df):
    for k, n in names.iteritems():
        df.loc[df['condition'] == k, 'condition'] = n
    return df


def sorted_conditions(df, original_conds):
    conditions = [names[c] for c in original_conds]
    _colors = [cond_colors[c] for c in original_conds]
    dfc = df[df['condition'].isin(conditions)]

    # sort by condition
    sorter_index = dict(zip(conditions, range(len(conditions))))
    dfc.loc[:, 'cnd_idx'] = dfc['condition'].map(sorter_index)
    dfc = dfc.set_index(['cnd_idx', 'run', 'Nuclei', 'Frame', 'Time']).sort_index().reset_index()

    return dfc, conditions, _colors


def mother_daughter_msd(df):
    _conds = ['mother-daughter']
    df, conds, colors = sorted_conditions(df, _conds)

    # df = df[df['Time'] <= 50]

    fig = matplotlib.pyplot.gcf()
    fig.clf()
    fig.set_size_inches(11.7, 11.7)
    gs = matplotlib.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])

    sp.msd(df[df['condition'] == names['mother-daughter']], ax1, ylim=msd_ylim)
    sp.msd_indivs(df[df['condition'] == names['mother-daughter']], ax3, ylim=msd_ylim)

    df_msd = ImagejPandas.msd_centrosomes(df)
    df_msd.loc[df_msd['CentrLabel'] == 'A', 'CentrLabel'] = 'Mother'
    df_msd.loc[df_msd['CentrLabel'] == 'B', 'CentrLabel'] = 'Daugther'
    # df_msd=df_msd.set_index(['CentrLabel','indv','Time']).sort_index().reset_index()

    sns.tsplot(data=df_msd[df_msd['CentrLabel'] == 'Mother'], color='k', linestyle='-',
               time='Time', value='msd', unit='indv', condition='CentrLabel', estimator=np.nanmean, ax=ax2)
    sns.tsplot(data=df_msd[df_msd['CentrLabel'] == 'Daugther'], color='k', linestyle='--',
               time='Time', value='msd', unit='indv', condition='CentrLabel', estimator=np.nanmean, ax=ax2)
    ax2.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
    ax2.set_xticks(np.arange(0, df['Time'].max(), 20.0))
    ax2.legend(title=None, loc='upper left')
    ax2.set_xlabel('Time delay $[min]$')

    sns.tsplot(data=df_msd, lw=3, err_style=['unit_traces'], err_kws=_err_kws,
               time='Time', value='msd', unit='indv', condition='CentrLabel', estimator=np.nanmean, ax=ax4)
    ax4.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
    ax4.set_xticks(np.arange(0, df['Time'].max(), 20.0))
    ax4.legend(title=None, loc='upper left')
    ax4.set_xlabel('Time delay $[min]$')

    ax1.set_ylim(msd_ylim)
    ax2.set_ylim(msd_ylim)
    ax3.set_ylim(msd_ylim)
    ax4.set_ylim(msd_ylim)

    plt.savefig('/Users/Fabio/mother-daughter.pdf', format='pdf')


if __name__ == '__main__':
    new_dist_name = 'Distance relative to nuclei center $[\mu m]$'
    new_speed_name = 'Speed relative to nuclei center $[\mu m/min]$'
    new_distcntr_name = 'Distance between centrosomes $[\mu m]$'
    new_speedcntr_name = 'Speed between centrosomes $[\mu m/min]$'

    df_m = pd.read_pickle('/Users/Fabio/merge.pandas')
    df_mc = pd.read_pickle('/Users/Fabio/merge_centered.pandas')

    # filter original dataframe to get just data between centrosomes
    dfcentr = df_mc[df_mc['CentrLabel'] == 'A']
    dfcentr['indv'] = dfcentr['condition'] + '-' + dfcentr['run'] + '-' + dfcentr['Nuclei'].map(int).map(str)
    dfcentr.drop(
        ['CentrLabel', 'Centrosome', 'NuclBound', 'CNx', 'CNy', 'CentX', 'CentY', 'NuclX', 'NuclY', 'Speed', 'Acc'],
        axis=1, inplace=True)

    df_m['indv'] = df_m['condition'] + '-' + df_m['run'] + '-' + df_m['Nuclei'].map(int).map(str) + '-' + \
                   df_m['Centrosome'].map(int).map(str)

    for id, df in df_m.groupby(['condition']):
        print 'condition %s: %d tracks' % (id, len(df['indv'].unique()) / 2.0)
    df_m = rename_conditions(df_m)
    dfcentr = rename_conditions(dfcentr)

    mother_daughter_msd(df_m)

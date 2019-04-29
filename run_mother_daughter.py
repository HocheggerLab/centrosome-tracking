from collections import OrderedDict

import matplotlib
import matplotlib.axes
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

import plot_special_tools as sp
from tools import stats as st
from imagej_pandas import ImagejPandas

print(font_manager.OSXInstalledFonts())
print(font_manager.OSXFontDirectories)

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


def mother_daughter_msd(df, dfc):
    _conds = ['mother-daughter']
    with PdfPages('mother-daughter.pdf') as pdf:
        df, conds, colors = sorted_conditions(df, _conds)

        # -----------
        # Page 1
        # -----------
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

        df_msd = ImagejPandas.msd_particles(df)
        df_msd.loc[df_msd['CentrLabel'] == 'A', 'CentrLabel'] = 'Mother'
        df_msd.loc[df_msd['CentrLabel'] == 'B', 'CentrLabel'] = 'Daugther'

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
        pdf.savefig()

        # -----------
        # Page 2
        # -----------
        dfc, conds, colors = sorted_conditions(dfc, _conds)
        fig.clf()
        fig.set_size_inches(11.7, 11.7)
        gs = matplotlib.gridspec.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])

        sns.tsplot(data=dfc[dfc['CentrLabel'] == 'A'], time='Time', value='DistCentr', unit='indv',
                   condition='condition', estimator=np.nanmean, lw=3, ax=ax1, err_style=['unit_traces'],
                   err_kws=_err_kws)
        ax1.set_xlabel('Time previous contact $[min]$')
        ax1.set_ylabel(new_distcntr_name)
        ax1.set_xlim(dfc['Time'].min(), 0)
        ax1.legend(title=None, loc='upper left')

        dfc_msd = ImagejPandas.msd_particles(dfc)
        dfc_msd.loc[dfc_msd['CentrLabel'] == 'A', 'CentrLabel'] = 'Mother'
        dfc_msd.loc[dfc_msd['CentrLabel'] == 'B', 'CentrLabel'] = 'Daugther'

        sns.tsplot(data=dfc_msd, time='Time', value='Dist', unit='indv',
                   condition='CentrLabel', estimator=np.nanmean, lw=3, ax=ax2, err_style=['unit_traces'],
                   err_kws=_err_kws)
        ax2.set_xlabel('Time previous contact $[min]$')
        ax2.set_ylabel(new_dist_name)
        ax2.set_xlim(dfc['Time'].min(), 0)
        ax2.legend(title=None, loc='upper left')

        sns.tsplot(data=dfc_msd[dfc_msd['CentrLabel'] == 'Mother'], color='k', linestyle='-',
                   time='Time', value='msd', unit='indv', condition='CentrLabel', estimator=np.nanmean, ax=ax3)
        sns.tsplot(data=dfc_msd[dfc_msd['CentrLabel'] == 'Daugther'], color='k', linestyle='--',
                   time='Time', value='msd', unit='indv', condition='CentrLabel', estimator=np.nanmean, ax=ax3)
        ax3.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
        ax3.legend(title=None, loc='upper left')
        ax3.set_xlabel('Time delay $[min]$')

        sns.tsplot(data=dfc_msd, lw=3, err_style=['unit_traces'], err_kws=_err_kws,
                   time='Time', value='msd', unit='indv', condition='CentrLabel', estimator=np.nanmean, ax=ax4)
        ax4.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
        ax4.legend(title=None, loc='upper left')
        ax4.set_xlabel('Time delay $[min]$')

        pdf.savefig()

        # -----------
        # Page 3
        # -----------
        fig.clf()
        fig.set_size_inches(11.7, 11.7)
        gs = matplotlib.gridspec.GridSpec(2, 2)
        # ax1 = plt.subplot(gs[0, 0])
        # ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])

        mua = dfc_msd[dfc_msd['Time'] < 0]
        sp.anotated_boxplot(mua, 'Speed', cat='CentrLabel', point_size=1, ax=ax3)
        # pmat = st.p_values(mua, 'Speed', 'CentrLabel')
        # ax3.text(0.5, 0.0, 'pvalue=%0.7f' % pmat[0, 1], ha='center', size='small')
        ax3.set_ylabel('Inst. track speed of centrosomes $[\mu m/min]$')

        mua = dfc_msd[dfc_msd['Time'] < 0].groupby(['condition', 'run', 'Nuclei', 'CentrLabel']).mean().reset_index()
        sp.anotated_boxplot(mua, 'Speed', cat='CentrLabel', point_size=3, ax=ax4)
        pmat = st.p_values(mua, 'Speed', 'CentrLabel')
        ax4.text(0.5, 0.0, 'pvalue=%0.7f' % pmat[0, 1], ha='center', size='small')
        ax4.set_ylabel('Avg. track speed of centrosomes $[\mu m/min]$')

        pdf.savefig()


def tracks(df):
    _conds = ['mother-daughter']
    df, conds, colors = sorted_conditions(df, _conds)
    fig = matplotlib.pyplot.gcf()
    fig.clf()
    fig.set_size_inches(_fig_size_A3)

    df.loc[df['CentrLabel'] == 'A', 'CentrLabel'] = 'Mother'
    df.loc[df['CentrLabel'] == 'B', 'CentrLabel'] = 'Daugther'

    g = sns.FacetGrid(df, legend_out=False, hue='CentrLabel', col='run', col_wrap=5)
    ma = g.map(plt.scatter, 'CentX', 'CentY')
    ma.add_legend()
    # g.fig.subplots_adjust(right=0.95)
    # g.fig.suptitle('Centrosome tracks')

    plt.savefig('/Users/Fabio/mother-daughter-tracks.pdf', format='pdf')


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

    df_m.loc[:, 'indv'] = df_m['condition'] + '-' + df_m['run'] + '-' + df_m['Nuclei'].map(int).map(str) + '-' + \
                          df_m['Centrosome'].map(int).map(str)
    df_mc.loc[:, 'indv'] = df_mc['condition'] + '-' + df_mc['run'] + '-' + df_mc['Nuclei'].map(int).map(str) + '-' + \
                           df_mc['Centrosome']

    for id, df in df_m.groupby(['condition']):
        print('condition %s: %d tracks' % (id, len(df['indv'].unique()) / 2.0))
    df_m = rename_conditions(df_m)
    dfcentr = rename_conditions(dfcentr)
    df_mc = rename_conditions(df_mc)

    mother_daughter_msd(df_m, df_mc)
    tracks(df_m)

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

import special_plots as sp
import stats as st

print font_manager.OSXInstalledFonts()
print font_manager.OSXFontDirectories

matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('svg', fonttype='none')
sns.set(context='paper', style='whitegrid', font='Arial', font_scale=0.9)
pd.set_option('display.width', 320)
plt.style.use('bmh')

names = {'1_N.C.': '-STLC',
         '1_P.C.': '+STLC',
         '1_DIC': 'DIC+STLC',
         '1_Dynei': 'DHC+STLC',
         '1_CENPF': 'CenpF',
         '1_BICD2': 'Bicaudal',
         '2_Kines1': 'Kinesin1',
         '2_CDK1_DK': 'DHC+Kinesin1',
         'hset': 'Hset',
         'kif25': 'Kif25',
         'hset+kif25': 'Hset+Kif25'
         }
_fig_size_A3 = (11.7, 16.5)
_err_kws = {'alpha': 0.3, 'lw': 1}


def rename_conditions(df):
    for k, n in names.iteritems():
        df.loc[df['condition'] == k, 'condition'] = n
    return df


def custom_round(x, base=1):
    return int(base * round(float(x) / base))


def sorted_conditions(df, original_conds):
    conditions = [names[c] for c in original_conds]
    colors = sns.color_palette(n_colors=len(conditions)).as_hex()
    dfc = df[df['condition'].isin(conditions)]

    # sort by condition
    sorter_index = dict(zip(conditions, range(len(conditions))))
    dfc.loc[:, 'cnd_idx'] = dfc['condition'].map(sorter_index)
    dfc = dfc.set_index(['cnd_idx', 'run', 'Nuclei', 'Frame', 'Time']).sort_index().reset_index()

    return dfc, conditions, colors


def fig_1(df, dfc):
    conditions = ['-STLC', '+STLC']
    dfc = dfc[dfc['condition'].isin(conditions)]
    df = df[df['condition'].isin(conditions)]

    fig = matplotlib.pyplot.gcf()
    fig.clf()
    fig.set_size_inches(_fig_size_A3)
    gs = matplotlib.gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, :])
    ax4 = plt.subplot(gs[2, 0], projection='3d')
    ax5 = plt.subplot(gs[2, 1], projection='3d')

    mua = dfc.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
    sp.anotated_boxplot(mua, 'SpeedCentr', order=conditions, point_size=2, ax=ax1)
    pmat = st.p_values(mua, 'SpeedCentr', 'condition')
    ax1.text(0.5, 0.6, 'pvalue=%0.2e' % pmat[0, 1], ha='center', size='small')
    ax1.set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')

    sns.tsplot(data=dfc, time='Time', value='DistCentr', unit='indv', condition='condition', estimator=np.nanmean, lw=3,
               ax=ax2, err_style=['unit_traces'], err_kws=_err_kws)
    ax2.set_xlabel('Time previous contact $[min]$')
    ax2.set_ylabel(new_distcntr_name)
    ax2.legend(title=None, loc='upper left')

    sp.congression(df, ax=ax3, order=conditions)

    sp.ribbon(df[df['condition'] == '-STLC'].groupby('indv').filter(lambda x: len(x) > 20), ax4)
    sp.ribbon(df[df['condition'] == '+STLC'].groupby('indv').filter(lambda x: len(x) > 20), ax5)

    # bugfix: rotate xticks for last subplot
    for tick in ax5.get_xticklabels():
        tick.set_rotation('horizontal')

    plt.savefig('/Users/Fabio/fig1.pdf', format='pdf')


def fig_2(df):
    df = df[df['condition'].isin([names['1_P.C.'], names['1_N.C.']])]
    df = df[df['Time'] <= 50]

    fig = matplotlib.pyplot.gcf()
    fig.clf()
    fig.set_size_inches(11.7, 11.7)
    gs = matplotlib.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])

    sp.msd(df[df['condition'] == names['1_N.C.']], ax1)
    sp.msd(df[df['condition'] == names['1_P.C.']], ax2)
    sp.msd_indivs(df[df['condition'] == names['1_N.C.']], ax3)
    sp.msd_indivs(df[df['condition'] == names['1_P.C.']], ax4)

    ax1.set_ylim(ax2.get_ylim())
    ax3.set_ylim(ax4.get_ylim())

    plt.savefig('/Users/Fabio/fig2.pdf', format='pdf')


def fig_3(df, dfc):
    _conds = ['1_P.C.', '1_Dynei', '1_DIC']
    df, conds, colors = sorted_conditions(df, _conds)
    dfc, conds, colors = sorted_conditions(dfc, _conds)

    fig = matplotlib.pyplot.gcf()
    fig.clf()
    fig.set_size_inches(11.7, 11.7)
    gs = matplotlib.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])

    mua = dfc.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
    sp.anotated_boxplot(mua, 'SpeedCentr', order=conds, point_size=2, ax=ax1, fontsize='medium')
    ax1.set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')

    sns.tsplot(data=dfc[dfc['condition'].isin(conds[0:2])],
               time='Time', value='DistCentr', unit='indv',
               condition='condition', estimator=np.nanmean, ax=ax2, lw=3,
               err_style=['unit_traces'], err_kws=_err_kws)
    ax2.set_xlabel('Time previous contact $[min]$')
    ax2.set_ylabel(new_distcntr_name)
    ax2.legend(title=None, loc='upper left')

    sp.congression(df, ax=ax3, order=conds)
    sp.msd(df[df['condition'] == names['1_DIC']], ax4)

    # bugfix: rotate xticks for last subplot
    for tick in ax4.get_xticklabels():
        tick.set_rotation('horizontal')

    plt.savefig('/Users/Fabio/fig3.pdf', format='pdf')


def fig_3sup(dfc):
    _conds = ['1_P.C.', '1_DIC']
    conditions = [names[c] for c in _conds]
    colors = dict(zip(_conds, sns.color_palette(n_colors=len(conditions))))
    dfc = dfc[dfc['condition'].isin(conditions)]

    # sort by condition
    sorter_index = dict(zip(conditions, range(len(conditions))))
    dfc['cnd_idx'] = dfc['condition'].map(sorter_index)
    dfc.sort_values(by='cnd_idx', inplace=True)

    fig = matplotlib.pyplot.gcf()
    fig.clf()
    fig.set_size_inches(11.7, 11.7)
    gs = matplotlib.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    sns.tsplot(data=dfc, time='Time', value='DistCentr', unit='indv',
               condition='condition', estimator=np.nanmean, ax=ax1, lw=3,
               err_style=['unit_traces'], err_kws=_err_kws)
    ax1.set_xlabel('Time previous contact $[min]$')
    ax1.set_ylabel(new_distcntr_name)
    ax1.legend(title=None, loc='upper left')

    plt.savefig('/Users/Fabio/fig3_sup.pdf', format='pdf')


def fig_4(df, dfc):
    fig = matplotlib.pyplot.gcf()
    fig.clf()
    fig.set_size_inches(11.7, 11.7)
    gs = matplotlib.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    _conds = ['1_P.C.', '1_Dynei', '1_CENPF', '1_BICD2']
    df1, conds, colors = sorted_conditions(df, _conds)

    with sns.color_palette(colors):
        sp.congression(df1, ax=ax1, order=conds)

    # ----------------
    # condition change
    # ----------------
    _conds = ['1_P.C.', '1_Dynei', '2_Kines1', '2_CDK1_DK']
    df1, conds, colors = sorted_conditions(df, _conds)

    with sns.color_palette(colors):
        sp.congression(df1, ax=ax2, order=conds)

    plt.savefig('/Users/Fabio/fig4.pdf', format='pdf')


def fig_4sup(df, dfc):
    fig = matplotlib.pyplot.gcf()
    fig.clf()
    fig.set_size_inches(11.7, 11.7)
    gs = matplotlib.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])

    _conds = ['1_P.C.', '1_Dynei', '1_BICD2']
    df1, conds, colors = sorted_conditions(df, _conds)
    dfc1, conds, colors = sorted_conditions(dfc, _conds)

    with sns.color_palette(colors):
        mua = dfc1.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        sp.anotated_boxplot(mua, 'SpeedCentr', order=conds, point_size=2, ax=ax1, fontsize='medium')
        ax1.set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')

        sp.congression(df1, ax=ax2, order=conds)

    # --------------------------
    # tracks for Dynein & CenpF
    # --------------------------
    _conds = ['1_Dynei', '1_CENPF']
    dfc, conds, colors = sorted_conditions(dfc, _conds)

    sns.tsplot(data=dfc, time='Time', value='DistCentr', unit='indv',
               condition='condition', estimator=np.nanmean, ax=ax3, lw=3,
               err_style=['unit_traces'], err_kws=_err_kws)
    ax3.set_xlabel('Time previous contact $[min]$')
    ax3.set_ylabel(new_distcntr_name)
    ax3.legend(title=None, loc='upper left')

    plt.savefig('/Users/Fabio/fig4_sup.pdf', format='pdf')


def fig_5(df, dfc):
    _conds = ['1_P.C.', 'hset', 'kif25', 'hset+kif25']
    df, conds, colors = sorted_conditions(df, _conds)
    dfc, conds, colors = sorted_conditions(dfc, _conds)

    with PdfPages('/Users/Fabio/fig5.pdf') as pdf:
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(_fig_size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])
        ax5 = plt.subplot(gs[2, 0])
        ax6 = plt.subplot(gs[2, 1])

        mua = dfc.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        sp.anotated_boxplot(mua, 'SpeedCentr', order=conds, point_size=2, ax=ax1, fontsize='medium')
        ax1.set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')

        sp.congression(df, ax=ax2, order=conds)

        # dfc.loc[:,'Time'] = dfc['Time'].map(lambda x: custom_round(x))
        dfc.loc[:, 'Time'] = dfc['Time'].round(1)
        print dfc[dfc['condition'] == names['hset']]['Time'].unique()
        print dfc[dfc['condition'] == names['hset']]['Frame'].unique()

        sns.tsplot(data=dfc[dfc['condition'].isin([conds[0], conds[1]])], ax=ax3, lw=3,
                   time='Frame', value='DistCentr', unit='indv',
                   condition='condition', estimator=np.nanmean,
                   err_style=['unit_traces'], err_kws=_err_kws)
        ax3.set_xlabel('Time previous contact $[frame]$')
        ax3.set_ylabel(new_distcntr_name)
        ax3.set_xticks(range(dfc['Frame'].min(), 0, 5))
        ax3.legend(title=None, loc='upper left')

        sns.tsplot(data=dfc[dfc['condition'].isin([conds[0], conds[2]])], ax=ax4, lw=3,
                   time='Frame', value='DistCentr', unit='indv',
                   condition='condition', estimator=np.nanmean,
                   err_style=['unit_traces'], err_kws=_err_kws)
        ax4.set_xlabel('Time previous contact $[frame]$')
        ax4.set_ylabel(new_distcntr_name)
        ax4.set_xticks(range(dfc['Frame'].min(), 0, 5))
        ax4.legend(title=None, loc='upper left')

        sns.tsplot(data=dfc[dfc['condition'].isin([conds[0], conds[3]])], ax=ax5, lw=3,
                   time='Frame', value='DistCentr', unit='indv',
                   condition='condition', estimator=np.nanmean,
                   err_style=['unit_traces'], err_kws=_err_kws)
        ax5.set_xlabel('Time previous contact $[frame]$')
        ax5.set_ylabel(new_distcntr_name)
        ax5.set_xticks(range(dfc['Frame'].min(), 0, 5))
        ax5.legend(title=None, loc='upper left')

        # bugfix: rotate xticks for last subplot
        for tick in ax6.get_xticklabels():
            tick.set_rotation('horizontal')

        pdf.savefig()
        plt.close()

        fig.clf()
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(_fig_size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])
        ax5 = plt.subplot(gs[2, 0])
        ax6 = plt.subplot(gs[2, 1])

        sp.msd(df[df['condition'] == conds[1]], ax1, time='Frame')
        sp.msd(df[df['condition'] == conds[2]], ax3, time='Frame')
        sp.msd(df[df['condition'] == conds[3]], ax5, time='Frame')
        sp.msd_indivs(df[df['condition'] == conds[1]], ax2, time='Frame')
        sp.msd_indivs(df[df['condition'] == conds[2]], ax4, time='Frame')
        sp.msd_indivs(df[df['condition'] == conds[3]], ax6, time='Frame')
        # bugfix: rotate xticks for last subplot
        for tick in ax6.get_xticklabels():
            tick.set_rotation('horizontal')
        pdf.savefig()
        plt.close()




if __name__ == '__main__':
    new_dist_name = 'Distance relative to nuclei center $[\mu m]$'
    new_speed_name = 'Speed relative to nuclei center $[\mu m/min]$'
    new_distcntr_name = 'Distance between centrosomes $[\mu m]$'
    new_speedcntr_name = 'Speed between centrosomes $[\mu m/min]$'

    df_m = pd.read_pickle('/Users/Fabio/merge.pandas')
    df_mc = pd.read_pickle('/Users/Fabio/merge_centered.pandas')

    df_m = df_m.loc[df_m['Time'] >= 0, :]
    df_m = df_m.loc[df_m['Time'] <= 100, :]
    df_mc = df_mc.loc[df_mc['Time'] <= 0, :]
    df_mc = df_mc.loc[df_mc['Time'] >= -100, :]

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

    fig_1(df_m, dfcentr)
    fig_2(df_m)
    fig_3(df_m, dfcentr)
    fig_3sup(dfcentr)
    fig_4(df_m, dfcentr)
    fig_4sup(df_m, dfcentr)
    fig_5(df_m, dfcentr)

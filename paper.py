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
import stats as st
from imagej_pandas import ImagejPandas

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
         '2_CDK1_DC': 'DHC+CenpF'
         }
_fig_size_A3 = (11.7, 16.5)
_err_kws = {'alpha': 0.3, 'lw': 1}


def rename_conditions(df):
    for k, n in names.iteritems():
        df.loc[df['condition'] == k, 'condition'] = n
    return df


def msd_tag(df):
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
    df = ImagejPandas.msd_centrosomes(df)
    mvtag = msd_tag(df)

    fig = matplotlib.pyplot.gcf()
    fig.clf()
    fig.set_size_inches(11.7, 11.7)
    gs = matplotlib.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])

    sns.tsplot(
        data=mvtag[mvtag['msd_cat'] == names['1_N.C.'] + ' moving more'],
        color='k', linestyle='-',
        time='Time', value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax1)
    sns.tsplot(
        data=mvtag[mvtag['msd_cat'] == names['1_N.C.'] + ' moving less'],
        color='k', linestyle='--',
        time='Time', value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax1)
    ax1.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
    ax1.set_xlabel('Time delay $[min]$')
    ax1.legend(title=None, loc='upper left')

    sns.tsplot(
        data=mvtag[mvtag['msd_cat'] == names['1_P.C.'] + ' moving more'],
        color='k', linestyle='-',
        time='Time', value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax2)
    sns.tsplot(
        data=mvtag[mvtag['msd_cat'] == names['1_P.C.'] + ' moving less'],
        color='k', linestyle='--',
        time='Time', value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax2)
    ax2.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
    ax2.set_xlabel('Time delay $[min]$')
    ax2.legend(title=None, loc='upper left')
    ax1.set_ylim(ax2.get_ylim())

    sns.tsplot(
        data=mvtag[mvtag['condition'] == names['1_N.C.']], lw=3,
        err_style=['unit_traces'], err_kws=_err_kws,
        time='Time', value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax3)
    ax3.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
    ax3.set_xlabel('Time delay $[min]$')
    ax3.legend(title=None, loc='upper left')

    sns.tsplot(
        data=mvtag[mvtag['condition'] == names['1_P.C.']], lw=3,
        err_style=['unit_traces'], err_kws=_err_kws,
        time='Time', value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax4)
    ax4.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
    ax4.set_xlabel('Time delay $[min]$')
    ax4.legend(title=None, loc='upper left')
    ax3.set_ylim(ax4.get_ylim())

    plt.savefig('/Users/Fabio/fig2.pdf', format='pdf')


def fig_3(df, dfc):
    _conds = ['1_P.C.', '1_Dynei', '1_DIC']
    conditions = [names[c] for c in _conds]
    colors = dict(zip(_conds, sns.color_palette(n_colors=len(conditions))))
    dfc = dfc[dfc['condition'].isin(conditions)]
    df = df[df['condition'].isin(conditions)]

    # sort by condition
    sorter_index = dict(zip(conditions, range(len(conditions))))
    dfc['cnd_idx'] = dfc['condition'].map(sorter_index)
    dfc.sort_values(by='cnd_idx', inplace=True)

    fig = matplotlib.pyplot.gcf()
    fig.clf()
    fig.set_size_inches(11.7, 11.7)
    gs = matplotlib.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])

    mua = dfc.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
    sp.anotated_boxplot(mua, 'SpeedCentr', order=conditions, point_size=2, ax=ax1, fontsize='medium')
    # pmat = st.p_values(mua, 'SpeedCentr', 'condition')
    # ax1.text(0.5, 0.6, 'pvalue=%0.2e' % pmat[0, 1], ha='center', size='small')
    ax1.set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')

    # sns.tsplot(data=dfc[dfc['condition'] == names['1_P.C.']], time='Time', value='DistCentr', unit='indv',
    #            condition='condition', estimator=np.nanmean, ax=ax2, lw=3, color=colors['1_P.C.'], err_style=None)
    sns.tsplot(data=dfc[dfc['condition'].isin([names[c] for c in ['1_P.C.', '1_Dynei']])],
               time='Time', value='DistCentr', unit='indv',
               condition='condition', estimator=np.nanmean, ax=ax2, lw=3,
               err_style=['unit_traces'], err_kws=_err_kws)
    # my_cmap = ListedColormap(sns.color_palette([colors['1_Dynei']]).as_hex())
    # dfc[dfc['condition'] == names['1_Dynei']].set_index('Time').sort_index().groupby('indv')['DistCentr'] \
    #     .plot(ax=ax2, colormap=my_cmap, alpha=0.5)
    # hndl = [mlines.Line2D([], [], color=colors['1_Dynei'], marker=None, label=names['1_Dynei'])]
    # lbls = list()
    # lbls.append(names['1_Dynei'])
    # ax2.legend(hndl, lbls, loc='upper right')
    ax2.set_xlabel('Time previous contact $[min]$')
    ax2.set_ylabel(new_distcntr_name)
    ax2.legend(title=None, loc='upper left')

    sp.congression(df, ax=ax3, order=conditions)

    df_msd = ImagejPandas.msd_centrosomes(df[(df['condition'] == names['1_DIC']) & (df['Time'] <= 100)])
    df_msd = msd_tag(df_msd)

    sns.tsplot(
        data=df_msd[df_msd['msd_cat'] == names['1_DIC'] + ' moving more'],
        color='k', linestyle='-',
        time='Time', value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax4)
    sns.tsplot(
        data=df_msd[df_msd['msd_cat'] == names['1_DIC'] + ' moving less'],
        color='k', linestyle='--',
        time='Time', value='msd', unit='indv', condition='msd_cat', estimator=np.nanmean, ax=ax4)
    ax4.set_xlabel('Time delay $[min]$')
    ax4.set_ylabel('Mean Square Displacement (MSD) $[\mu m^2]$')
    ax4.set_xticks(np.arange(0, df_msd['Time'].max(), 20.0))
    ax4.legend(title=None, loc='upper left')

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
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])

    _conds = ['1_P.C.', '1_Dynei', '1_CENPF', '2_CDK1_DC']
    conditions = [names[c] for c in _conds]
    colors = sns.color_palette(n_colors=len(conditions)).as_hex()
    print colors
    dfc1 = dfc[dfc['condition'].isin(conditions)]
    df1 = df[df['condition'].isin(conditions)]

    # sort by condition
    sorter_index = dict(zip(conditions, range(len(conditions))))
    dfc1['cnd_idx'] = dfc1['condition'].map(sorter_index)
    dfc1.sort_values(by='cnd_idx', inplace=True)

    with sns.color_palette(colors):
        mua = dfc1.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        sp.anotated_boxplot(mua, 'SpeedCentr', order=conditions, point_size=2, ax=ax1, fontsize='medium')
        ax1.set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')

        sp.congression(df1, ax=ax2, order=conditions)

    # ----------------
    # condition change
    # ----------------
    _conds = ['1_P.C.', '1_Dynei', '2_Kines1', '2_CDK1_DK']
    conditions = [names[c] for c in _conds]
    colors = colors[0:2] + [sns.xkcd_rgb['medium green'], sns.xkcd_rgb['pale red']]
    print colors
    dfc1 = dfc[dfc['condition'].isin(conditions)]
    df1 = df[df['condition'].isin(conditions)]

    # sort by condition
    sorter_index = dict(zip(conditions, range(len(conditions))))
    dfc1['cnd_idx'] = dfc1['condition'].map(sorter_index)
    dfc1.sort_values(by='cnd_idx', inplace=True)

    with sns.color_palette(colors):
        mua = dfc1.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        sp.anotated_boxplot(mua, 'SpeedCentr', order=conditions, point_size=2, ax=ax3, fontsize='medium')
        ax1.set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')

        sp.congression(df1, ax=ax4, order=conditions)

    plt.savefig('/Users/Fabio/fig4.pdf', format='pdf')

    _conds = ['1_P.C.', '1_Dynei', '1_CENPF', '2_CDK1_DC', '2_Kines1', '2_CDK1_DK']
    conditions = [names[c] for c in _conds]
    dfc1 = dfc[dfc['condition'].isin(conditions)]
    _mua = dfc1.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
    pmat = st.p_values(_mua, 'SpeedCentr', 'condition', filename='/Users/Fabio/fig4_pvalues.xlsx')


def fig_4sup(df, dfc):
    _conds = ['1_P.C.', '1_Dynei', '1_BICD2']
    conditions = [names[c] for c in _conds]
    colors = dict(zip(_conds, sns.color_palette(n_colors=len(conditions))))
    dfc1 = dfc[dfc['condition'].isin(conditions)]
    df1 = df[df['condition'].isin(conditions)]

    fig = matplotlib.pyplot.gcf()
    fig.clf()
    fig.set_size_inches(11.7, 11.7)
    gs = matplotlib.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])

    # sort by condition
    sorter_index = dict(zip(conditions, range(len(conditions))))
    dfc1['cnd_idx'] = dfc1['condition'].map(sorter_index)
    dfc1.sort_values(by='cnd_idx', inplace=True)

    with sns.color_palette([c for k, c in colors.iteritems()]):
        mua = dfc1.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        sp.anotated_boxplot(mua, 'SpeedCentr', order=conditions, point_size=2, ax=ax1, fontsize='medium')
        ax1.set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')

        sp.congression(df1, ax=ax2, order=conditions)

    # --------------------------
    # tracks for Dynein & CenpF
    # --------------------------
    _conds = ['1_Dynei', '1_CENPF']
    conditions = [names[c] for c in _conds]
    dfc = dfc[dfc['condition'].isin(conditions)]

    # sort by condition
    sorter_index = dict(zip(conditions, range(len(conditions))))
    dfc['cnd_idx'] = dfc['condition'].map(sorter_index)
    dfc.sort_values(by='cnd_idx', inplace=True)

    sns.tsplot(data=dfc, time='Time', value='DistCentr', unit='indv',
               condition='condition', estimator=np.nanmean, ax=ax3, lw=3,
               err_style=['unit_traces'], err_kws=_err_kws)
    ax3.set_xlabel('Time previous contact $[min]$')
    ax3.set_ylabel(new_distcntr_name)
    ax3.legend(title=None, loc='upper left')

    plt.savefig('/Users/Fabio/fig4_sup.pdf', format='pdf')


def discrepancy(df):
    # plot of every distance between centrosome's, centered at time of contact
    plt.figure(110)
    sns.set_palette('Set2')
    df_idx_grp = df.set_index('Time').sort_index().reset_index()
    g = sns.FacetGrid(df_idx_grp, col='condition', hue='indv', col_wrap=2, size=5)
    g.map(plt.plot, 'Time', 'DistCentr', linewidth=1, alpha=0.5)
    g.set_axis_labels(x_var='Time [min]', y_var=new_distcntr_name)
    plt.savefig('/Users/Fabio/dist_centrosomes_PC-pc.svg', format='svg')

    # average speed boxplot
    plt.figure(100)
    mua = df.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
    sp.anotated_boxplot(mua, 'SpeedCentr', point_size=3)
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.2, top=0.9, wspace=0.2, hspace=0.2)
    plt.xlabel('Time [min]')
    plt.ylabel(new_speedcntr_name)
    pmat = st.p_values(mua, 'SpeedCentr', 'condition', '/Users/Fabio/pval_PC-pc.txt')
    plt.text(0.5, 0.8, 'pvalue=%0.5f' % pmat[0, 1], ha='center')
    plt.savefig('/Users/Fabio/boxplot_avg_speed_PC-pc.svg', format='svg')


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
    dfcentr.drop(['CentrLabel', 'Centrosome', 'NuclBound', 'CNx', 'CNy', 'CentX', 'CentY', 'NuclX', 'NuclY'],
                 axis=1, inplace=True)

    df_m['indv'] = df_m['condition'] + '-' + df_m['run'] + '-' + df_m['Nuclei'].map(int).map(str) + '-' + \
                   df_m['Centrosome'].map(int).map(str)

    for id, df in df_m.groupby(['condition']):
        print 'condition %s: %d tracks' % (id, len(df['indv'].unique()) / 2.0)
    df_m = rename_conditions(df_m)
    dfcentr = rename_conditions(dfcentr)
    # discrepancy(dfcentr.loc[(dfcentr['Time'] <= 0) & (dfcentr['condition'].isin(['1_P.C.', 'pc'])), :])

    fig_1(df_m, dfcentr)
    fig_2(df_m)
    fig_3(df_m, dfcentr)
    fig_3sup(dfcentr)
    fig_4(df_m, dfcentr)
    fig_4sup(df_m, dfcentr)

from collections import OrderedDict

import matplotlib
import matplotlib.axes
import matplotlib.gridspec
import matplotlib.lines as mlines
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

names = OrderedDict([('1_N.C.', '-STLC'),
                     ('1_P.C.', '+STLC'),
                     ('1_DIC', 'DIC+STLC'),
                     ('1_Dynei', 'DHC+STLC'),
                     ('1_CENPF', 'CenpF'),
                     ('1_BICD2', 'Bicaudal'),
                     ('2_Kines1', 'Kinesin1'),
                     ('2_CDK1_DK', 'DHC+Kinesin1'),
                     ('1_No10+', 'Nocodazole 10ng'),
                     ('1_CyDT', 'Cytochalsin D'),
                     ('1_FAKI', 'FAKi'),
                     ('hset', 'Hset'),
                     ('kif25', 'Kif25'),
                     ('hset+kif25', 'Hset+Kif25')])
col_palette = ["#e74c3c", "#34495e",
               "#3498db", sns.xkcd_rgb["teal green"], "#9b59b6", "#2ecc71",
               sns.xkcd_rgb["windows blue"], sns.xkcd_rgb["medium green"],
               '#91744B', sns.xkcd_rgb["pale red"], sns.xkcd_rgb["amber"],
               '#91744B', sns.xkcd_rgb["pale red"], sns.xkcd_rgb["amber"]]
cond_colors = dict(zip(names.keys(), col_palette))
_fig_size_A3 = (11.7, 16.5)
_err_kws = {'alpha': 0.3, 'lw': 1}
msd_ylim = [0, 420]


def rename_conditions(df):
    for k, n in names.iteritems():
        df.loc[df['condition'] == k, 'condition'] = n
    return df


def custom_round(x, base=1):
    return int(base * round(float(x) / base))


def sorted_conditions(df, original_conds):
    conditions = [names[c] for c in original_conds]
    _colors = [cond_colors[c] for c in original_conds]
    dfc = df[df['condition'].isin(conditions)]

    # sort by condition
    sorter_index = dict(zip(conditions, range(len(conditions))))
    dfc.loc[:, 'cnd_idx'] = dfc['condition'].map(sorter_index)
    dfc = dfc.set_index(['cnd_idx', 'run', 'Nuclei', 'Frame', 'Time']).sort_index().reset_index()

    return dfc, conditions, _colors


def fig_1(df, dfc):
    _conds = ['1_N.C.', '1_P.C.']
    df, conds, colors = sorted_conditions(df, _conds)
    dfc, conds, colors = sorted_conditions(dfc, _conds)

    fig = matplotlib.pyplot.gcf()
    fig.clf()
    fig.set_size_inches(_fig_size_A3)
    gs = matplotlib.gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, :])
    ax4 = plt.subplot(gs[2, 0], projection='3d')
    ax5 = plt.subplot(gs[2, 1], projection='3d')

    with sns.color_palette(colors):
        mua = dfc.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        sp.anotated_boxplot(mua, 'SpeedCentr', order=conds, point_size=2, ax=ax1)
        pmat = st.p_values(mua, 'SpeedCentr', 'condition')
        ax1.text(0.5, 0.6, 'pvalue=%0.2e' % pmat[0, 1], ha='center', size='small')
        ax1.set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')

        sns.tsplot(data=dfc, time='Time', value='DistCentr', unit='indv', condition='condition', estimator=np.nanmean,
                   lw=3,
                   ax=ax2, err_style=['unit_traces'], err_kws=_err_kws)
        ax2.set_xlabel('Time previous contact $[min]$')
        ax2.set_ylabel(new_distcntr_name)
        ax2.legend(title=None, loc='upper left')

        sp.congression(df, ax=ax3, order=conds)

    sp.ribbon(df[df['condition'] == '-STLC'].groupby('indv').filter(lambda x: len(x) > 20), ax4)
    sp.ribbon(df[df['condition'] == '+STLC'].groupby('indv').filter(lambda x: len(x) > 20), ax5)

    # bugfix: rotate xticks for last subplot
    for tick in ax5.get_xticklabels():
        tick.set_rotation('horizontal')

    plt.savefig('/Users/Fabio/fig1.pdf', format='pdf')


def fig_2(df):
    _conds = ['1_N.C.', '1_P.C.']
    df, conds, colors = sorted_conditions(df, _conds)

    df = df[df['Time'] <= 50]

    fig = matplotlib.pyplot.gcf()
    fig.clf()
    fig.set_size_inches(11.7, 11.7)
    gs = matplotlib.gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])

    sp.msd(df[df['condition'] == names['1_N.C.']], ax1, ylim=msd_ylim)
    sp.msd(df[df['condition'] == names['1_P.C.']], ax2, ylim=msd_ylim)
    sp.msd_indivs(df[df['condition'] == names['1_N.C.']], ax3, ylim=msd_ylim)
    sp.msd_indivs(df[df['condition'] == names['1_P.C.']], ax4, ylim=msd_ylim)

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

    with sns.color_palette(colors):
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
    sp.msd(df[df['condition'] == names['1_DIC']], ax4, ylim=msd_ylim)

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

    dfc1 = dfc[dfc['condition'].isin([conds[0], conds[1]])]
    colrs1 = [colors[0], colors[1]]
    dfc2 = dfc[dfc['condition'].isin([conds[0], conds[2]])]
    colrs2 = [colors[0], colors[2]]
    dfc3 = dfc[dfc['condition'].isin([conds[0], conds[3]])]
    colrs3 = [colors[0], colors[3]]

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

        with sns.color_palette(colors):
            mua = dfc.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
            sp.anotated_boxplot(mua, 'SpeedCentr', order=conds, point_size=2, ax=ax6, fontsize='medium')
            ax6.set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')

            sp.congression(df, ax=ax4, order=conds)

        dfc1.set_index('Time').sort_index().groupby('indv')['DistCentr'].plot(ax=ax2, lw=1, color=colors[1], alpha=0.5)
        hndl = [mlines.Line2D([], [], color=colors[1], marker=None, label=conds[1])]
        lbls = list(conds[1])
        ax2.legend(hndl, lbls, loc='upper right')
        ax2.set_xlabel('Time previous contact $[min]$')
        ax2.set_ylabel(new_distcntr_name)

        sns.tsplot(data=dfc1, ax=ax1, lw=3, color=colrs1,
                   time='Time', value='DistCentr', unit='indv',
                   condition='condition', estimator=np.nanmean,
                   err_style=['unit_traces'], err_kws=_err_kws)

        sns.tsplot(data=dfc2, ax=ax3, lw=3, color=colrs2,
                   time='Time', value='DistCentr', unit='indv',
                   condition='condition', estimator=np.nanmean,
                   err_style=['unit_traces'], err_kws=_err_kws)

        sns.tsplot(data=dfc3, ax=ax5, lw=3, color=colrs3,
                   time='Time', value='DistCentr', unit='indv',
                   condition='condition', estimator=np.nanmean,
                   err_style=['unit_traces'], err_kws=_err_kws)

        for ax in [ax1, ax3, ax5]:
            ax.set_xlabel('Time previous contact $[min]$')
            ax.set_ylabel(new_distcntr_name)
            ax.legend(title=None, loc='upper left')

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

        df = df[df['Time'] <= 50]

        sp.msd(df[df['condition'] == conds[1]], ax1, ylim=msd_ylim)
        sp.msd(df[df['condition'] == conds[2]], ax3, ylim=msd_ylim)
        sp.msd(df[df['condition'] == conds[3]], ax5, ylim=msd_ylim)
        sp.msd_indivs(df[df['condition'] == conds[1]], ax2, ylim=msd_ylim)
        sp.msd_indivs(df[df['condition'] == conds[2]], ax4, ylim=msd_ylim)
        sp.msd_indivs(df[df['condition'] == conds[3]], ax6, ylim=msd_ylim)
        # bugfix: rotate xticks for last subplot
        for tick in ax6.get_xticklabels():
            tick.set_rotation('horizontal')
        pdf.savefig()
        plt.close()


def fig_6(df, dfc):
    _conds = ['1_P.C.', '1_CyDT', '1_FAKI', '1_No10+']
    df, conds, colors = sorted_conditions(df, _conds)
    dfc, conds, colors = sorted_conditions(dfc, _conds)

    dfc1 = dfc[dfc['condition'].isin([conds[1]])]
    colrs1 = ["#34495e"]
    dfc2 = dfc[dfc['condition'].isin([conds[2]])]
    colrs2 = ["#34495e"]
    dfc3 = dfc[dfc['condition'].isin([conds[3]])]
    colrs3 = ["#34495e"]

    with PdfPages('/Users/Fabio/fig6.pdf') as pdf:
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(_fig_size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])
        ax5 = plt.subplot(gs[2, 0])

        with sns.color_palette(colors):
            mua = dfc.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
            sp.anotated_boxplot(mua, 'SpeedCentr', order=conds, point_size=2, ax=ax2, fontsize='medium')
            ax2.set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')

            sp.congression(df, ax=ax4, order=conds)

        sns.tsplot(data=dfc1, ax=ax1, lw=3, color=colrs1,
                   time='Time', value='DistCentr', unit='indv',
                   condition='condition', estimator=np.nanmean,
                   err_style=['unit_traces'], err_kws=_err_kws)

        sns.tsplot(data=dfc2, ax=ax3, lw=3, color=colrs2,
                   time='Time', value='DistCentr', unit='indv',
                   condition='condition', estimator=np.nanmean,
                   err_style=['unit_traces'], err_kws=_err_kws)

        sns.tsplot(data=dfc3, ax=ax5, lw=3, color=colrs3,
                   time='Time', value='DistCentr', unit='indv',
                   condition='condition', estimator=np.nanmean,
                   err_style=['unit_traces'], err_kws=_err_kws)

        for ax in [ax1, ax3, ax5]:
            ax.set_xlabel('Time previous contact $[min]$')
            ax.set_ylabel(new_distcntr_name)
            ax.legend(title=None, loc='upper left')

        # bugfix: rotate xticks for last subplot
        for tick in ax5.get_xticklabels():
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

        df = df[df['Time'] <= 50]

        sp.msd(df[df['condition'] == conds[1]], ax1, ylim=msd_ylim)
        sp.msd(df[df['condition'] == conds[2]], ax3, ylim=msd_ylim)
        sp.msd(df[df['condition'] == conds[3]], ax5, ylim=msd_ylim)
        sp.msd_indivs(df[df['condition'] == conds[1]], ax2, ylim=msd_ylim)
        sp.msd_indivs(df[df['condition'] == conds[2]], ax4, ylim=msd_ylim)
        sp.msd_indivs(df[df['condition'] == conds[3]], ax6, ylim=msd_ylim)
        # bugfix: rotate xticks for last subplot
        for tick in ax6.get_xticklabels():
            tick.set_rotation('horizontal')
        pdf.savefig()
        plt.close()


def color_keys(dfc):
    fig = matplotlib.pyplot.gcf()
    fig.clf()
    coldf, conds, colors = sorted_conditions(dfc, names.keys())
    # print names.keys(), conds
    with sns.color_palette(colors):
        mua = coldf.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        sp.anotated_boxplot(mua, 'SpeedCentr', order=conds)
        fig.gca().set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')
        fig.savefig('/Users/Fabio/colors.pdf', format='pdf')


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

    # filter starting distances greater than a threshold
    indivs_filter = dfcentr.set_index(['Time', 'indv']).unstack('indv')['DistCentr'].fillna(method='bfill').iloc[0]
    indivs_filter = indivs_filter[indivs_filter > 5].index.values
    # print indivs_filter
    dfcentr = dfcentr[dfcentr['indv'].isin(indivs_filter)]

    color_keys(dfcentr)

    fig_1(df_m, dfcentr)
    fig_2(df_m)
    fig_3(df_m, dfcentr)
    fig_3sup(dfcentr)
    fig_4(df_m, dfcentr)
    fig_4sup(df_m, dfcentr)
    # fig_5(df_m, dfcentr)
    fig_6(df_m, dfcentr)

    # # d = df_m[df_m['CentrLabel'] == 'A']
    # d = dfcentr
    # dfct = d[d['condition'] == names['hset']]
    #
    # # plot of every speed between centrosome's, centered at time of contact
    # plt.figure(111, figsize=(10, 10))
    # g = sns.FacetGrid(dfct, col='condition', hue='indv', size=10)
    # g.map(plt.plot, 'Time', 'DistCentr', linewidth=1, alpha=0.5)
    # g.map(plt.scatter, 'Time', 'DistCentr', s=1)
    # g.set_xticklabels(labels=sorted(dfct['Time'].unique()))
    # plt.savefig('/Users/Fabio/1.pdf', format='pdf')

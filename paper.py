import matplotlib.axes
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import special_plots as sp
import stats as st

sns.set(font_scale=0.9, context='paper', style='whitegrid')
pd.set_option('display.width', 320)
plt.style.use('bmh')


def fig_1(df, dfc):
    dfc = dfc[dfc['condition'].isin(['1_P.C.', '1_N.C.'])]
    df = df[df['condition'].isin(['1_P.C.', '1_N.C.'])]
    df.loc[df['condition'] == '1_N.C.', 'condition'] = '-STLC'
    df.loc[df['condition'] == '1_P.C.', 'condition'] = '+STLC'
    dfc.loc[dfc['condition'] == '1_N.C.', 'condition'] = '-STLC'
    dfc.loc[dfc['condition'] == '1_P.C.', 'condition'] = '+STLC'

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(11.7, 16.5)
    gs = matplotlib.gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, :])
    # ax4 = plt.subplot(gs[2, 0])
    # ax5 = plt.subplot(gs[2, 1])
    # ax6 = plt.subplot(gs[6, 0])

    dfc = dfc.loc[dfc['Time'] <= 0, :]
    mua = dfc.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
    sp.anotated_boxplot(mua, 'SpeedCentr', order=['-STLC', '+STLC'], size=2, ax=ax1)
    pmat = st.p_values(mua, 'SpeedCentr', 'condition')
    ax1.text(0.5, 1.0, 'pvalue=%0.2e' % pmat[0, 1], ha='center', size='small')
    ax1.set_ylabel('Avg. track speed between centrosomes $\\left[\\frac{\mu m}{min} \\right]$')

    sns.tsplot(data=dfc, time='Time', value='DistCentr', unit='indv', condition='condition', estimator=np.nanmean,
               ax=ax2)
    ax2.set_xlabel('Time previous contact $[min]$')
    ax2.set_ylabel(new_distcntr_name)

    sp.congression(df, ax=ax3, order=['-STLC', '+STLC'])

    # dfcig = dfc[dfc['condition'] == '+STLC'].set_index('Time').sort_index().groupby('indv')
    # dfcig.plot(y='DistCentr', linewidth=1, alpha=0.5, legend=False, ax=ax4)
    # ax4.set_xlabel('Time previous contact $[min]$')
    # ax4.set_ylabel(new_distcntr_name)
    #
    # sns.tsplot(data=dfc, time='Time', value='DistCentr', unit='indv', condition='condition', estimator=np.nanmean,
    #            err_style='unit_traces', ax=ax5)
    # ax5.set_xlabel('Time previous contact $[min]$')
    # ax5.set_ylabel(new_distcntr_name)


    plt.savefig('/Users/Fabio/fig1.svg', format='svg')


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
    sp.anotated_boxplot(mua, 'SpeedCentr', size=3)
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.2, top=0.9, wspace=0.2, hspace=0.2)
    plt.xlabel('Time [min]')
    plt.ylabel(new_speedcntr_name)
    pmat = st.p_values(mua, 'SpeedCentr', 'condition', '/Users/Fabio/pval_PC-pc.txt')
    plt.text(0.5, 0.8, 'pvalue=%0.5f' % pmat[0, 1], ha='center')
    plt.savefig('/Users/Fabio/boxplot_avg_speed_PC-pc.svg', format='svg')


if __name__ == '__main__':
    new_dist_name = 'Distance relative\nto nuclei center $[\mu m]$'
    new_speed_name = 'Speed relative\nto nuclei center $\\left[\\frac{\mu m}{min} \\right]$'
    new_distcntr_name = 'Distance between \ncentrosomes $[\mu m]$'
    new_speedcntr_name = 'Speed between \ncentrosomes $\\left[\\frac{\mu m}{min} \\right]$'

    df_m = pd.read_pickle('/Users/Fabio/merge.pandas')
    df_mc = pd.read_pickle('/Users/Fabio/merge_centered.pandas')

    # filter original dataframe to get just data between centrosomes
    dfcentr = df_mc[df_mc['CentrLabel'] == 'A']
    dfcentr['indv'] = dfcentr['condition'] + '-' + dfcentr['run'] + '-' + dfcentr['Nuclei'].map(int).map(str)
    dfcentr.drop(['CentrLabel', 'Centrosome', 'NuclBound', 'CNx', 'CNy', 'CentX', 'CentY', 'NuclX', 'NuclY'],
                 axis=1, inplace=True)

    df_m['indv'] = df_m['condition'] + '-' + df_m['run'] + '-' + df_m['Nuclei'].map(int).map(str)

    # discrepancy(dfcentr.loc[(dfcentr['Time'] <= 0) & (dfcentr['condition'].isin(['1_P.C.', 'pc'])), :])
    fig_1(df_m, dfcentr)

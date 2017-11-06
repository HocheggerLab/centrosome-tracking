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

import report as r
import special_plots as sp
from imagej_pandas import ImagejPandas

print font_manager.OSXInstalledFonts()
print font_manager.OSXFontDirectories

plt.style.use('bmh')
# plt.style.use('ggplot')
# sns.set(context='paper', style='whitegrid', font='Helvetica Neue')
# matplotlib.rc('pdf', fonttype=42)
# matplotlib.rc('svg', fonttype='none')
print matplotlib.rcParams.keys()
matplotlib.rcParams.update({'axes.titlesize': 20})
matplotlib.rcParams.update({'axes.labelsize': 20})
matplotlib.rcParams.update({'xtick.labelsize': 20})
matplotlib.rcParams.update({'ytick.labelsize': 20})

matplotlib.rcParams.update({'xtick.color': sp.SUSSEX_COBALT_BLUE})
matplotlib.rcParams.update({'ytick.color': sp.SUSSEX_COBALT_BLUE})
matplotlib.rcParams.update({'text.color': sp.SUSSEX_COBALT_BLUE})
matplotlib.rcParams.update({'lines.color': sp.SUSSEX_COBALT_BLUE})
matplotlib.rcParams.update({'axes.labelcolor': sp.SUSSEX_COBALT_BLUE})
# matplotlib.rcParams.update({'axes.edgecolor': sns.light_palette(sp.SUSSEX_COBALT_BLUE, 6)[2]})
matplotlib.rcParams.update({'axes.edgecolor': '#FFFFFF00'})
# matplotlib.rcParams.update({'figure.edgecolor': sns.light_palette(sp.SUSSEX_COBALT_BLUE, 6)[2]})
matplotlib.rcParams.update({'grid.color': sns.light_palette(sp.SUSSEX_COBALT_BLUE, 6)[2]})
# matplotlib.rcParams.update({'grid.color': sp.SUSSEX_COBALT_BLUE})
# matplotlib.rcParams.update({'grid.alpha': 0.3})
matplotlib.rcParams.update({'lines.color': sp.SUSSEX_COBALT_BLUE})
matplotlib.rcParams.update({'legend.fontsize': 18})

pd.set_option('display.width', 320)

names = OrderedDict([('1_N.C.', '-STLC'),
                     ('1_P.C.', '+STLC'),

                     ('2_Kines1', 'Kinesin1'),
                     ('2_CDK1_DK', 'DHC+Kinesin1'),

                     ('1_DIC', 'DIC+STLC'),
                     ('1_Dynei', 'DHC+STLC'),  # DyneinH1
                     ('1_CENPF', 'CenpF'),
                     ('1_BICD2', 'Bicaudal'),

                     ('1_No10+', 'Nocodazole 10ng'),

                     ('1_CyDT', 'Cytochalsin D'),
                     ('1_FAKI', 'FAKi'),

                     ('hset', 'Hset'),
                     ('kif25', 'Kif25'),
                     ('hset+kif25', 'Hset+Kif25'),

                     ('pc', '+STLC.'),
                     ('mother-daughter', 'Centrosomes')])

col_palette = [[sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_CORAL_RED],
               sns.light_palette(sp.SUSSEX_MID_GREY, reverse=True, n_colors=2 + 1).as_hex()[0:2],
               sns.light_palette(sp.SUSSEX_BURNT_ORANGE, reverse=True, n_colors=4 + 1).as_hex()[0:4],
               [sp.SUSSEX_TURQUOISE],
               sns.dark_palette(sp.SUSSEX_FUSCHIA_PINK, reverse=True, n_colors=2 + 1).as_hex()[0:2],
               sns.light_palette(sp.SUSSEX_DEEP_AQUAMARINE, reverse=True, n_colors=3 + 1).as_hex(),
               sp.SUSSEX_SUNSHINE_YELLOW, sp.SUSSEX_BURNT_ORANGE]
col_palette = [item for sublist in col_palette for item in sublist]
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


def retreat0(_df, _mask):
    _conds = ['pc']
    df, conds, colors = sorted_conditions(_df, _conds)

    mask, condsm, colorsm = sorted_conditions(_mask, _conds)

    df = df[df['Nuclei'] == 2]
    df = df[df['run'] == 'run_100']
    mask = mask[mask['Nuclei'] == 2]
    mask = mask[mask['run'] == 'run_100']

    condition = df['condition'].unique()[0]
    run = df['run'].unique()[0]
    nuclei = df['Nuclei'].unique()[0]

    with PdfPages('/Users/Fabio/retreat2017fig-0.pdf') as pdf:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex='col',
                                       gridspec_kw={'height_ratios': [2, 1]}, figsize=(9.3, 9.3 / 3))
        plt.subplots_adjust(left=0.125, bottom=0.22, right=0.9, top=0.99, wspace=0.2, hspace=0.1)

        mask['Time'] = mask['Time'].astype('int32')

        between_df = df[df['CentrLabel'] == 'A']
        mask_c = r.centr_masks(mask)
        time_of_c, frame_of_c, dist_of_c = ImagejPandas.get_contact_time(df, ImagejPandas.DIST_THRESHOLD)

        sp.distance_to_nucleus(df, ax1, mask=mask, time_contact=time_of_c, plot_interp=True)
        # sp.speed_to_nucleus(df, ax2, mask=mask, time_contact=time_of_c)
        # sp.acceleration_to_nucleus(df, ax3, mask=mask, time_contact=time_of_c)
        sp.distance_between_centrosomes(between_df, ax2, mask=mask_c, time_contact=time_of_c)
        # sp.speed_between_centrosomes(between_df, ax5, mask=mask_c, time_contact=time_of_c)
        # sp.plot_acceleration_between_centrosomes(between_df, ax6, mask=mask_c, time_contact=time_of_c)

        # # change y axis title properties for small plots
        # for _ax in [ax2, ax3, ax4, ax5, ax6]:
        #     _ax.set_ylabel(_ax.get_ylabel(), rotation='horizontal', ha='right', fontsize=9, weight='ultralight')

        # ax2.set_xticks(ax1.get_xticks())
        # ax1.set_xticks(np.arange(0, df['Time'].max(), 10.0))
        # ax2.set_xticks(np.arange(0, df['Time'].max(), 10.0))
        # ax2.set_xticklabels(ax1.get_xticklabels())
        ax1.legend().remove()

        ax1.set_ylabel('$D_{nuclei}$ $[\mu m]$')
        ax2.set_ylabel('$D_{between}$')
        ax2.set_xlabel('Time $[min]$')

        pdf.savefig(transparent=True)


def retreat1(df, dfc):
    _conds = ['1_N.C.', '1_P.C.']
    _df, conds, colors = sorted_conditions(df, _conds)

    with PdfPages('/Users/Fabio/retreat2017fig-1.pdf') as pdf:
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(9.3, 9.3)
        gs = matplotlib.gridspec.GridSpec(2, 2, wspace=0.0)
        ax1 = plt.subplot(gs[0, 0], projection='3d')
        ax2 = plt.subplot(gs[0, 1], projection='3d')
        ax1 = fig.add_axes(sp.MyAxes3D(ax1, 'l'))
        ax2 = fig.add_axes(sp.MyAxes3D(ax2, 'r'))
        plt.subplots_adjust(left=0.05, bottom=0.22, right=0.95, top=0.95, wspace=0.0, hspace=0.0)

        zm = _df['DistCentr'].max()
        sp.ribbon(_df[_df['condition'] == '-STLC'].groupby('indv').filter(lambda x: len(x) > 20), ax1, z_max=zm)
        sp.ribbon(_df[_df['condition'] == '+STLC'].groupby('indv').filter(lambda x: len(x) > 20), ax2, z_max=zm)

        # bugfix: rotate xticks for last subplot
        # for tick in ax5.get_xticklabels():
        #     tick.set_rotation('horizontal')

        ax1.set_zlabel('$D_{between}$ $[\mu m]$')
        ax2.set_zlabel('')

        pdf.savefig(transparent=True)

    _conds = ['mother-daughter']
    _df, conds, colors = sorted_conditions(df, _conds)
    with PdfPages('/Users/Fabio/retreat2017fig-mother-daughter.pdf') as pdf:
        # -----------
        # Page 1
        # -----------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(9.3, 9.3)
        gs = matplotlib.gridspec.GridSpec(2, 2, wspace=0.0)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        # ax3 = plt.subplot(gs[1, 0])
        # ax4 = plt.subplot(gs[1, 1])
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

        # expressed as a fraction of the average axis height
        sp.msd(_df[_df['condition'] == names['mother-daughter']], ax1, ylim=msd_ylim, color=sp.SUSSEX_COBALT_BLUE)

        df_msd = ImagejPandas.msd_centrosomes(_df)
        df_msd.loc[df_msd['CentrLabel'] == 'A', 'CentrLabel'] = 'Mother'
        df_msd.loc[df_msd['CentrLabel'] == 'B', 'CentrLabel'] = 'Daugther'

        sns.tsplot(data=df_msd[df_msd['CentrLabel'] == 'Mother'], color=sp.SUSSEX_COBALT_BLUE, linestyle='-',
                   time='Time', value='msd', unit='indv', condition='CentrLabel', estimator=np.nanmean, ax=ax2)
        sns.tsplot(data=df_msd[df_msd['CentrLabel'] == 'Daugther'], color=sp.SUSSEX_COBALT_BLUE, linestyle='--',
                   time='Time', value='msd', unit='indv', condition='CentrLabel', estimator=np.nanmean, ax=ax2)

        ax1.set_ylabel('MSD $[\mu m^2]$')
        ax1.set_xticks(np.arange(0, _df['Time'].max(), 20.0))
        # ax1.set_xlabel('Time delay $[min]$')
        ax1.set_xlabel('')
        ax1.legend(title=None, loc='upper left')

        ax2.yaxis.set_label_text('')
        ax2.yaxis.set_ticklabels([])
        # ax2.set_xlabel('Time delay $[min]$')
        ax2.set_xlabel('')
        ax2.legend(title=None, loc='upper right')

        fig.text(0.5, 0.45, 'Time delay $[min]$', ha='center', fontsize=24)

        ax1.set_ylim(msd_ylim)
        ax2.set_ylim(msd_ylim)
        ax2.set_xticks(ax1.get_xticks())

        pdf.savefig(transparent=True)


def retreat2(df):
    _conds = ['1_N.C.', '1_P.C.', '2_Kines1', '2_CDK1_DK', '1_DIC', '1_Dynei', '1_CENPF', '1_BICD2', '1_No10+',
              '1_CyDT', '1_FAKI', 'hset', 'kif25', 'hset+kif25']
    markers = ['o', 'o', 's', 's', 'v', '^', '<', '>', 'p', 'h', 'X', '*', 'P', 'X']
    df, conds, colors = sorted_conditions(df, _conds)
    colortuple = dict(zip(conds, colors))

    with PdfPages('/Users/Fabio/retreat2017fig-2.pdf') as pdf:
        # -----------
        # Page 1
        # -----------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(9.3, 9.3)
        ax = plt.gca()
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

        df_ = df[df['Time'] <= 100]
        df_msd = ImagejPandas.msd_centrosomes(df_).set_index('Frame').sort_index()
        dfcg = sp._compute_congression(df).set_index('Time').sort_index()
        df_msd_final = pd.DataFrame()
        for _, dfmsd in df_msd.groupby(ImagejPandas.CENTROSOME_INDIV_INDEX):
            cnd = dfmsd.iloc[0]['condition']
            cgr = dfcg[dfcg['condition'] == cnd].iloc[-1]['congress']
            df_it = pd.DataFrame([[dfmsd.iloc[-1]['msd'], cgr, cnd]],
                                 columns=['msd', 'cgr', 'condition'])
            df_msd_final = df_msd_final.append(df_it)

        # inneficient as cgr is repeated for every sample
        df_msd_final = df_msd_final.groupby('condition').mean().reset_index()
        for c, m in zip(_conds, markers):
            cnd = names[c]
            p = df_msd_final[df_msd_final['condition'] == cnd]
            ax.scatter(p['cgr'], p['msd'], c=colortuple[cnd], s=100, label=cnd, marker=m, zorder=1000)

        ax.legend(loc='upper right')
        ax.set_ylabel('MSD $[\mu m^2]$')
        ax.set_xlabel('Congression [%]')

        # fig.patch.set_alpha(0.0)
        pdf.savefig(transparent=True)

        # with PdfPages('/Users/Fabio/retreat2017fig-22.pdf') as pdf:
        #     # -----------
        #     # Page 2
        #     # -----------
        #     fig = matplotlib.pyplot.gcf()
        #     fig.clf()
        #     fig.set_size_inches(9.3, 9.3)
        #     ax = plt.gca()
        #
        #     # df_ = df[df['Time'] <= 50]
        #     df_msd = ImagejPandas.msd_centrosomes(df_).set_index('Frame').sort_index()
        #     for c, m in zip(_conds, markers):
        #         cnd = names[c]
        #         dff = df_msd[df_msd['condition'] == cnd]
        #         sns.tsplot(data=dff, linestyle='-', time='Time', value='msd', unit='indv', condition='condition',
        #                    color=colortuple[cnd], estimator=np.nanmean, ax=ax)
        #
        #     ax.legend(loc='upper left')
        #     ax.set_ylabel('MSD $[\mu m^2]$')
        #     ax.set_xlabel('Time $[min]$')
        #
        #     # fig.patch.set_alpha(0.0)
        #     pdf.savefig(transparent=True)


def retreat4(df):
    dt_before_contact = 30
    t_per_frame = 5
    _conds = ['pc']
    df, conds, colors = sorted_conditions(df, _conds)
    colortuple = dict(zip(conds, colors))

    with PdfPages('/Users/Fabio/retreat2017fig-4.pdf') as pdf:
        # -----------
        # Page 1
        # -----------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(9.3, 9.3 / 2)
        ax = plt.gca()
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

        stats = pd.DataFrame()
        idx = ImagejPandas.MASK_INDEX
        idx.append('run')
        for i, (id, idf) in enumerate(df.groupby(ImagejPandas.NUCLEI_INDIV_INDEX)):
            if 'CentrLabel' in idf:
                idf = idf.drop('CentrLabel', axis=1)
            s = idf.set_index(idx).sort_index()
            u = s.unstack('Centrosome')
            # u = u.fillna(method='pad').stack().reset_index()
            u = u.interpolate().stack().reset_index()
            idf = ImagejPandas.vel_acc_nuclei(u)

            d_thr = ImagejPandas.DIST_THRESHOLD * 3
            time_of_c, frame_of_c, dist_of_c = ImagejPandas.get_contact_time(idf, d_thr)
            print i, id, time_of_c, frame_of_c, dist_of_c

            if time_of_c is not None:
                frame_before = frame_of_c - dt_before_contact / t_per_frame
                if frame_before < 0:
                    frame_before = 0
                dists_before_contact = list(idf[idf['Frame'] == frame_before]['Dist'])
                min_dist = max_dist = time_before = np.NaN
                if len(dists_before_contact) > 0:
                    max_dist = max(dists_before_contact)
                    min_dist = min(dists_before_contact)
                    time_before = idf[idf['Frame'] == frame_before]['Time'].unique()[0]
            else:
                frame_before = time_before = min_dist = max_dist = np.NaN

            ini_frame = idf.set_index('Frame').sort_index().index[0]
            ini_time = idf[idf['Frame'] == ini_frame]['Time'].unique()[0]
            ini_dist_min = min(idf[idf['Frame'] == ini_frame]['Dist'])

            df_rowc = pd.DataFrame({'Tag': [id],
                                    'Nuclei': idf['Nuclei'].unique()[0],
                                    'Frame': [frame_of_c],
                                    'Time': [time_of_c],
                                    'Stat': 'Contact',
                                    'Type': '(i)',
                                    'Dist': [dist_of_c]})
            df_row1 = pd.DataFrame({'Tag': [id],
                                    'Nuclei': idf['Nuclei'].unique()[0],
                                    'Frame': [frame_before],
                                    'Time': [time_before],
                                    'Stat': 'Contact',
                                    'Type': '(ii)',
                                    'Dist': [max_dist]})
            df_row2 = pd.DataFrame({'Tag': [id],
                                    'Nuclei': idf['Nuclei'].unique()[0],
                                    'Frame': [frame_before],
                                    'Time': [time_before],
                                    'Stat': 'Contact',
                                    'Type': '(iii)',
                                    'Dist': [min_dist]})
            stats = stats.append(df_row1, ignore_index=True)
            stats = stats.append(df_row2, ignore_index=True)
            stats = stats.append(df_rowc, ignore_index=True)

        sdata = stats[(stats['Stat'] == 'Contact') & (stats['Dist'].notnull())][['Dist', 'Type']]
        sdata['Dist'] = sdata.Dist.astype(np.float64)  # fixes a bug of seaborn
        sns.boxplot(data=sdata, y='Dist', x='Type', whis=np.inf, width=0.3)
        for i, artist in enumerate(ax.artists):
            artist.set_facecolor('None')
            artist.set_edgecolor(sp.SUSSEX_COBALT_BLUE)
            artist.set_zorder(5000)
        for i, artist in enumerate(ax.lines):
            artist.set_color(sp.SUSSEX_COBALT_BLUE)
            artist.set_zorder(5000)

        ax = sns.swarmplot(data=sdata, y='Dist', x='Type', zorder=100, color=sp.SUSSEX_CORAL_RED)
        ax.set_xlabel('')
        ax.set_ylabel('$D_{nuclei}$ $[\mu m]$')

        pdf.savefig(transparent=True)


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
    df_msk_disk = pd.read_pickle('/Users/Fabio/mask.pandas')

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

    mask = rename_conditions(df_msk_disk)
    df_m = rename_conditions(df_m)
    dfcentr = rename_conditions(dfcentr)

    # color_keys(dfcentr)

    retreat0(df_m, mask)
    retreat1(df_m, df_mc)
    retreat2(df_m)
    retreat4(df_m)

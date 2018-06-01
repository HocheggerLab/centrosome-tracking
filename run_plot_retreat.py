import ConfigParser
import json
import logging
from collections import OrderedDict

import matplotlib
import matplotlib.axes
import matplotlib.colors
import matplotlib.gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

import elastica as e
import parameters
import plot_special_tools as sp
import run_plot_report as r
from imagej_pandas import ImagejPandas

log = logging.getLogger(__name__)
log.info(font_manager.OSXInstalledFonts())
log.info(font_manager.OSXFontDirectories)

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
matplotlib.rcParams.update({'legend.fontsize': 18})

matplotlib.rcParams.update({'xtick.color': sp.SUSSEX_COBALT_BLUE})
matplotlib.rcParams.update({'ytick.color': sp.SUSSEX_COBALT_BLUE})
matplotlib.rcParams.update({'text.color': sp.SUSSEX_COBALT_BLUE})
matplotlib.rcParams.update({'lines.color': sp.SUSSEX_COBALT_BLUE})
matplotlib.rcParams.update({'axes.labelcolor': sp.SUSSEX_COBALT_BLUE})
matplotlib.rcParams.update({'axes.edgecolor': '#FFFFFF00'})
matplotlib.rcParams.update({'grid.color': sns.light_palette(sp.SUSSEX_COBALT_BLUE, 6)[2]})
matplotlib.rcParams.update({'lines.color': sp.SUSSEX_COBALT_BLUE})

pd.set_option('display.width', 320)

names = OrderedDict([('1_N.C.', '-STLC'),
                     ('1_P.C.', '+STLC'),

                     ('2_Kines1', 'Kinesin1'),
                     ('2_CDK1_DK', 'DHC+Kinesin1'),

                     ('1_DIC', 'DIC'),
                     ('1_Dynei', 'DHC'),  # DyneinH1
                     ('1_CENPF', 'CenpF'),
                     ('1_BICD2', 'Bicaudal'),

                     ('1_No10+', 'Nocodazole 10ng'),
                     ('1_MCAK', 'MCAK'),
                     ('1_chTOG', 'chTog'),

                     ('1_CyDT', 'Cytochalsin D'),
                     ('1_FAKI', 'FAKi'),
                     ('1_Bleb', 'Blebbistatin'),

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

    with PdfPages(parameters.data_dir + 'out/retreat2017fig-0.pdf') as pdf:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex='col',
                                       gridspec_kw={'height_ratios': [2, 1]}, figsize=(9.3, 9.3 / 3))
        plt.subplots_adjust(left=0.125, bottom=0.22, right=0.9, top=0.99, wspace=0.2, hspace=0.1)

        mask['Time'] = mask['Time'].astype('int32')

        between_df = df[df['CentrLabel'] == 'A']
        mask_c = r.centrosome_masks(mask)
        time_of_c, frame_of_c, dist_of_c = ImagejPandas.get_contact_time(df, ImagejPandas.DIST_THRESHOLD)

        with sns.color_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_TURQUOISE]):
            sp.distance_to_nuclei_center(df, ax1, mask=mask, time_contact=time_of_c, plot_interp=True)
        with sns.color_palette([sp.SUSSEX_COBALT_BLUE]):
            sp.distance_between_centrosomes(between_df, ax2, mask=mask_c, time_contact=time_of_c, )
        ax1.legend().remove()

        ax1.set_ylabel('$D_{nuclei}$ $[\mu m]$')
        ax2.set_ylabel('$D_{between}$')
        ax2.set_xlabel('Time $[min]$')

        pdf.savefig(transparent=True)


def retreat1(df, dfc):
    _conds = ['1_N.C.', '1_P.C.']
    _df, conds, colors = sorted_conditions(df, _conds)

    with PdfPages(parameters.data_dir + 'out/retreat2017fig-1.pdf') as pdf:
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

        ax1.set_zlabel('$D_{between}$ $[\mu m]$')
        ax2.set_zlabel('')

        pdf.savefig(transparent=True)

    _conds = ['mother-daughter']
    _df, conds, colors = sorted_conditions(df, _conds)
    with PdfPages(parameters.data_dir + 'out/retreat2017fig-mother-daughter.pdf') as pdf:
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

        df_msd = ImagejPandas.msd_particles(_df)
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
        ax2.set_xlabel('')
        ax2.legend(title=None, loc='upper right')

        fig.text(0.5, 0.45, 'Time delay $[min]$', ha='center', fontsize=24)

        ax1.set_ylim(msd_ylim)
        ax2.set_ylim(msd_ylim)
        ax2.set_xticks(ax1.get_xticks())

        pdf.savefig(transparent=True)


def retreat2(df):
    _conds = ['1_N.C.', '1_P.C.',
              '2_Kines1', '2_CDK1_DK', '1_DIC', '1_Dynei', '1_CENPF', '1_BICD2',
              '1_No10+', '1_MCAK', '1_chTOG',
              '1_CyDT', '1_FAKI', '1_Bleb']

    markers = ['o', 'o',
               's', 'X', 'v', '^', '<', '>',
               'p', 'P', 'X',
               'p', 'P', 'X']
    df, conds, colors = sorted_conditions(df, _conds)
    colors = [sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE]
    colors.extend([sp.SUSSEX_FLINT] * 6)
    colors.extend([sp.SUSSEX_FUSCHIA_PINK] * 3)
    colors.extend([sp.SUSSEX_TURQUOISE] * 3)
    colortuple = dict(zip(conds, colors))

    with PdfPages(parameters.data_dir + 'out/retreat2017fig-2.pdf') as pdf:
        # -----------
        # Page 1
        # -----------
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(12, 9.3)
        ax = plt.gca()
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.7, top=0.9, wspace=0.2, hspace=0.2)

        df_ = df[df['Time'] <= 100]
        df_msd = ImagejPandas.msd_particles(df_).set_index('Frame').sort_index()
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
            ax.scatter(p['cgr'], p['msd'], c=colortuple[cnd], s=600, label=cnd, marker=m, zorder=1000)

        ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
        ax.set_ylabel('MSD $[\mu m^2]$')
        ax.set_xlabel('Congression [%]')

        # fig.patch.set_alpha(0.0)
        pdf.savefig(transparent=True)


def retreat4(df):
    dt_before_contact = 30
    t_per_frame = 5
    _conds = ['pc']
    df, conds, colors = sorted_conditions(df, _conds)
    colortuple = dict(zip(conds, colors))

    with PdfPages(parameters.data_dir + 'out/retreat2017fig-4.pdf') as pdf:
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


def retreat5(df):
    with PdfPages(parameters.data_dir + 'out/elastica.pdf') as pdf:
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(5.27, 5.27)
        gs = matplotlib.gridspec.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0:2, 0:2])

        with open(parameters.data_dir + 'elastica.cfg.txt', 'r') as configfile:
            config = ConfigParser.ConfigParser()
            config.readfp(configfile)

        print 'sections found in file ', config.sections()

        section = config.sections()[1]
        yn = np.array(json.loads(config.get(section, 'measure')))
        inip = np.array(json.loads(config.get(section, 'comment')))

        num_points = 500

        # ---------------------
        # plot initial model
        # ---------------------
        L, a1, a2, E, F, gamma, x0, y0, theta = inip
        s = np.linspace(0, L, num_points)
        r = e.heavy_planar_bvp(s, F=F, E=E, gamma=gamma)
        pol = r.sol
        xo = pol(s)[0:2, :]
        ys = e.eval_heavy_planar(s, pol, a1, a2)[0:2, :]

        # deal with rotations and translations
        sinth, costh = np.sin(theta), np.cos(theta)
        M = np.array([[costh, -sinth], [sinth, costh]])
        ys = np.matmul(M, ys) + np.array([x0, y0]).reshape((2, 1))
        xo = np.matmul(M, xo) + np.array([x0, y0]).reshape((2, 1))

        _ax = ax1
        _ax.plot(xo[0], xo[1], lw=3, c=sp.SUSSEX_CORAL_RED, zorder=2)
        _ax.plot(ys[0], ys[1], lw=5, c=sp.SUSSEX_POWDER_BLUE, zorder=1)
        _ax.scatter(yn[0], yn[1], c=sp.SUSSEX_POWDER_BLUE, marker='X', zorder=3)

        # ---------------------
        # plot estimations
        # ---------------------
        othr = 50
        filter_df = df[df['objfn'] < othr]
        print 'filtered %d rows' % len(filter_df.index)
        for row in filter_df.iterrows():
            row = row[1]
            s = np.linspace(0, row['L'], num_points)
            r = e.heavy_planar_bvp(s, F=row['F'], E=row['E'], gamma=row['gamma'])
            pol = r.sol
            xo = pol(s)[0:2, :]
            xs, ys = e.eval_heavy_planar(s, pol, row['a1'], row['a2'])[0:2, :]
            ys = np.array([xs, ys])

            # deal with rotations and translations
            sinth, costh = np.sin(row['theta']), np.cos(row['theta'])
            M = np.array([[costh, -sinth], [sinth, costh]])
            ys = np.matmul(M, ys) + np.array([row['x0'], row['y0']]).reshape((2, 1))
            xo = np.matmul(M, xo) + np.array([row['x0'], row['y0']]).reshape((2, 1))

            _ax.plot(xo[0], xo[1], lw=1, c=sp.SUSSEX_COBALT_BLUE, label='%0.1f' % row['E'], alpha=0.4, zorder=4)

        # fig.suptitle('Solutions with an objective function lower than %0.1f' % othr)
        # ax1.set_title('Solutions with an objective function lower than %0.1f' % othr)

        red_patch = mpatches.Patch(color=sp.SUSSEX_COBALT_BLUE, label='Estimation')
        green_patch = mpatches.Patch(color=sp.SUSSEX_CORAL_RED, label='Ground')
        ax1.legend(handles=[red_patch, green_patch], loc='lower right')

        pdf.savefig(transparent=True)
        plt.close()


def animations(_df, _mask):
    import matplotlib.lines as lines
    import h5py
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.use('Agg')

    _hdf5file = parameters.data_dir + 'centrosomes.nexus.hdf5'
    _conds = ['pc']
    df, conds, colors = sorted_conditions(_df, _conds)
    mask, condsm, colorsm = sorted_conditions(_mask, _conds)

    df = df[df['Nuclei'] == 2]
    df = df[df['run'] == 'run_100']
    mask = mask[mask['Nuclei'] == 2]
    mask = mask[mask['run'] == 'run_100']

    # condition = df['condition'].unique()[0]
    condition = _conds[0]
    run = df['run'].unique()[0]
    nuclei = df['Nuclei'].unique()[0]

    dpi_anim = 326
    fig = plt.figure(dpi=dpi_anim)
    fig.set_size_inches(4.87 / 3, 2.31)
    plt.subplots_adjust(left=0.3)
    ax = plt.gca()

    with h5py.File(_hdf5file, 'r') as f:
        ch2 = f['%s/%s/raw/%03d/channel-2' % (condition, run, 0)]
        resolution = ch2.parent.attrs['resolution']

    def cell_movie(ax, frame):
        with h5py.File(_hdf5file, 'r') as f:
            if 'pandas_dataframe' not in f['%s/%s/measurements' % (condition, run)]:
                raise KeyError('No data for selected condition-run.')

            nuclei_list = f['%s/%s/measurements/nuclei' % (condition, run)]
            centrosome_list = f['%s/%s/measurements/centrosomes' % (condition, run)]
            sel = f['%s/%s/selection' % (condition, run)]
            df = pd.read_hdf(_hdf5file, key='%s/%s/measurements/pandas_dataframe' % (condition, run), mode='r')

            with h5py.File(_hdf5file, 'r') as f:
                ch2 = f['%s/%s/raw/%03d/channel-2' % (condition, run, frame)]
                data = ch2[:]

            ax.imshow(data, extent=[0, 512 / resolution, 512 / resolution, 0])

            # for nucID in nuclei_list:
            for nucID in ['N02']:
                nuc = nuclei_list[nucID]
                nid = int(nucID[1:])
                if nid == 0: continue
                nfxy = nuc['pos'].value
                nuc_frames = nfxy.T[0]

                fidx = nuc_frames.searchsorted(frame)
                nx = nfxy[fidx][1] / resolution
                ny = nfxy[fidx][2] / resolution

                # is_in_selected_nuclei = int(nucID[1:]) == nuclei
                # circle = mpatches.Circle((nx - 5, ny - 5), radius=1, color=sp.SUSSEX_CORN_YELLOW)
                # ax.add_patch(circle)
                # ax.text(nx + 10, ny + 5, nucID, color='white')
                # ax.text(10, 30, '%02d - (%03d,%03d)' % (frame, dwidth, dheight))

                # get nuclei boundary as a polygon
                df_nucfr = df[(df['Nuclei'] == nid) & (df['Frame'] == frame)]
                if len(df_nucfr['NuclBound'].values) > 0:
                    nuc_boundary_str = df_nucfr['NuclBound'].values[0]
                    if nuc_boundary_str[1:-1] != '':
                        nucb_points = eval(nuc_boundary_str[1:-1])
                        points = [[x, y] for x, y in nucb_points]
                        nucleipol = plt.Polygon(points, closed=True, fill=None, lw=1,
                                                edgecolor=sp.SUSSEX_CORN_YELLOW)
                        ax.add_patch(nucleipol)

            # for cntrID in centrosome_list:
            for cntrID, txt, col in zip(['C201', 'C001'], ['C1', 'C2'], [sp.SUSSEX_CORAL_RED, sp.SUSSEX_TURQUOISE]):
                cntr = centrosome_list[cntrID]
                cfxy = cntr['pos'].value
                cnt_frames = cfxy.T[0]

                if frame in cnt_frames:
                    fidx = cnt_frames.searchsorted(frame)
                    cx = cfxy[fidx][1]
                    cy = cfxy[fidx][2]

                    circle = mpatches.Circle((cx, cy), radius=1, color=col, zorder=300)
                    ax.add_patch(circle)
                    ax.text(cx + 2, cy, txt, color='white')

                    nuc = nuclei_list['N02']
                    nfxy = nuc['pos'].value
                    nuc_frames = nfxy.T[0]
                    if frame in nuc_frames:
                        fidx = nuc_frames.searchsorted(frame)
                        nx = nfxy[fidx][1]
                        ny = nfxy[fidx][2]

                        circle = mpatches.Circle((nx, ny), radius=0.4, color=sp.SUSSEX_CORN_YELLOW, zorder=300)
                        ax.add_patch(circle)
                        line = lines.Line2D([nx, cx], [ny, cy], c='white', lw=1, zorder=100)
                        ax.add_line(line)

            ax.set_xlabel('$X$ $[\mu m]$')
            ax.set_ylabel('$Y$ $[\mu m]$')
            ax.set_xlim(0, 60)
            ax.set_ylim(0, 60)
            ax.set_xticks(range(0, 60, 20))
        return ax

    def dist_movie(frame, df):
        figsiz = (4.87 / 3 * 2, 2.31)
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex='col',
                                       gridspec_kw={'height_ratios': [2, 1]}, figsize=figsiz)
        plt.subplots_adjust(left=0.15, bottom=0.22, right=0.9, top=0.99, wspace=0.2, hspace=0.1)
        # fig.set_size_inches(2.31, 4.87 / 3 * 2)

        max_time = df[df['Frame'] == 20]['Time'].values[0]
        df = df[df['Frame'] <= frame]
        mask['Time'] = mask['Time'].astype('int32')

        between_df = df[df['CentrLabel'] == 'A']
        mask_c = r.centrosome_masks(mask)
        time_of_c, frame_of_c, dist_of_c = ImagejPandas.get_contact_time(df, ImagejPandas.DIST_THRESHOLD)

        with sns.color_palette([sp.SUSSEX_CORAL_RED, sp.SUSSEX_TURQUOISE]):
            sp.distance_to_nuclei_center(df, ax1, mask=mask, time_contact=time_of_c, plot_interp=True)
            sp.distance_between_centrosomes(between_df, ax2, mask=mask_c, time_contact=time_of_c)
        ax1.legend().remove()

        # print max_time
        ax1.set_xlim(0, max_time)
        ax1.set_ylim(0, df['Dist'].max())
        ax1.set_ylabel('$D_{nuclei}$ $[\mu m]$')
        ax2.set_ylabel('$D_{between}$')
        ax2.set_xlabel('Time $[min]$')

        return

    for f in range(20):
        ax.cla()
        cell_movie(ax, f)
        plt.savefig(parameters.data_dir + 'out/mov1_f%03d.png' % f, dpi=dpi_anim)

    for f in range(20):
        ax.cla()
        dist_movie(f, df)
        plt.savefig(parameters.data_dir + 'out/mov2_f%03d.png' % f, dpi=dpi_anim, transparent=True)


def color_keys(dfc):
    fig = matplotlib.pyplot.gcf()
    fig.clf()
    coldf, conds, colors = sorted_conditions(dfc, names.keys())
    # print names.keys(), conds
    with sns.color_palette(colors):
        mua = coldf.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
        sp.anotated_boxplot(mua, 'SpeedCentr', order=conds)
        fig.gca().set_ylabel('Avg. track speed between centrosomes $[\mu m/min]$')
        fig.savefig(parameters.data_dir + 'out/colors.pdf', format='pdf')


if __name__ == '__main__':
    new_dist_name = 'Distance relative to nuclei center $[\mu m]$'
    new_speed_name = 'Speed relative to nuclei center $[\mu m/min]$'
    new_distcntr_name = 'Distance between centrosomes $[\mu m]$'
    new_speedcntr_name = 'Speed between centrosomes $[\mu m/min]$'

    df_m = pd.read_pickle(parameters.data_dir + 'merge.pandas')
    df_mc = pd.read_pickle(parameters.data_dir + 'merge_centered.pandas')
    df_msk_disk = pd.read_pickle(parameters.data_dir + 'mask.pandas')

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

    df_m.loc[:, 'indv'] = df_m['condition'] + '-' + df_m['run'] + '-' + df_m['Nuclei'].map(int).map(str) + '-' + \
                          df_m['Centrosome'].map(int).map(str)
    df_mc.loc[:, 'indv'] = df_mc['condition'] + '-' + df_mc['run'] + '-' + df_mc['Nuclei'].map(int).map(str) + '-' + \
                           df_mc['Centrosome']

    for id, dfc in df_m.groupby(['condition']):
        log.info('condition %s: %d tracks' % (id, len(dfc['indv'].unique()) / 2.0))

    mask = rename_conditions(df_msk_disk)
    df_m = rename_conditions(df_m)
    dfcentr = rename_conditions(dfcentr)

    # color_keys(dfcentr)

    retreat0(df_m, mask)
    retreat1(df_m, df_mc)
    retreat2(df_m)
    retreat4(df_m)
    retreat5(df=pd.read_csv(parameters.data_dir + 'elastica.csv'))
    animations(df_m, mask)

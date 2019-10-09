import os
import logging

import pandas as pd
import numpy as np

from tools.matplotlib_essentials import plt
import seaborn as sns

import parameters as p
import mechanics as m
import microtubules.eb3._manual_trackmate as tm
import microtubules.eb3._plots as pl
import tools.plot_tools as sp

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 50)


def import_dir(dir_base):
    log.info('processing data from folder %s' % dir_base)
    _df = pd.DataFrame()

    # Traverse through all subdirs looking for image files.
    # When a file is found, assume folder structure of (cond/experiment)
    file_n = 0
    for root, directories, files in os.walk(dir_base):
        for f in files:
            mpath = os.path.join(root, f)
            if os.path.isfile(mpath) and f[-4:] == '.xml':
                log.info('processing file %s in folder %s' % (f, root))
                xml_path = os.path.join(root, f)
                xdf = tm.read(xml_path)
                if not xdf.empty:
                    xdf.loc[:, "file_id"] = file_n
                    file_n += 1
                    dirname = os.path.basename(os.path.dirname(root))
                    xdf.loc[:, "condition"] = dirname
                    _df = _df.append(xdf, ignore_index=True, sort=False)
                else:
                    log.warning("empty dataframe!")
    return _df


def create_df(path):
    _df = import_dir(path)
    _df.rename(columns={'time_s': 'time', 'x_um': 'x', 'y_um': 'y', 'track_id': 'particle'}, inplace=True)
    _df['tag'] = ""

    # Show the names of tracs that don't start with Trackmate's default  of "Track_"
    track_idx = _df["track_name"].apply(lambda v: v[0:6] == "Track_")
    print(_df.loc[~track_idx, "track_name"].unique())
    _df = _df[track_idx]

    indiv_idx = ['condition', 'file_id', 'particle']
    _df.loc[:, 'indv'] = _df['condition'] + '|' + _df['file_id'].map(str) + '|' + _df['particle'].map(str)

    logging.info('computing speed, acceleration, length, MSD')
    _df = m.get_speed_acc(_df, group=indiv_idx)
    _df = m.get_center_df(_df, group=indiv_idx)
    _df = m.get_trk_length(_df, group=indiv_idx)
    _df = m.get_msd(_df, group=indiv_idx)
    for needs_rounding in ['time', 'time_i', 'dist', 'dist_i', 'speed', 'acc', 's', 'msd', 'x', 'y']:
        _df.loc[:, needs_rounding] = _df[needs_rounding].apply(np.round, decimals=4)
    _df["spot_id"] = _df["spot_id"].astype(int)
    _df["file_id"] = _df["file_id"].astype(int)

    return _df


if __name__ == '__main__':
    eb3_path = "/Users/Fabio/data/lab/airy-eb3"
    # df = create_df(eb3_path)
    # df.to_csv(os.path.join(p.compiled_data_dir, "eb3.csv"))
    df = pd.read_csv(os.path.join(p.compiled_data_dir, "eb3.csv"), index_col=False)

    # construct track_average track speed and track length
    idv_grp = ['condition', 'file_id']

    dfi = df.set_index('frame').sort_index()
    dfi.loc[:, 'speed'] = dfi['speed'].abs()

    logging.info('making speed stat plots')
    df_avg = dfi.groupby(idv_grp)['time', 'speed'].mean()
    df_avg.loc[:, 'time'] = dfi.groupby(idv_grp)['time'].first()
    df_avg.loc[:, 'n_points'] = dfi.groupby(idv_grp)['x'].count()
    df_avg.loc[:, 'length'] = dfi.groupby(idv_grp)['s'].agg(np.sum)
    df_avg = df_avg.reset_index()
    print(df_avg)

    sns.set_palette([sns.xkcd_rgb["grey"], sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
    fig = plt.figure()
    fig.set_size_inches((3.5, 3.5))
    ax = plt.gca()
    sp.anotated_boxplot(df_avg, variable='speed', stars=True, ax=ax)
    ax.set_ylim([0, 0.7])
    fig.savefig(p.out_dir + 'eb3_speed_boxplot.pdf')

    logging.info('making msd plots')
    msd = pl.MSD(df)
    # print(sorted(msd.timeseries['time'].unique()))

    sns.set_palette([sns.xkcd_rgb["grey"], sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
    fig = plt.figure()
    fig.set_size_inches((3.5, 3.5))
    ax = plt.gca()
    msd.track_each(ax)
    fig.savefig(p.out_dir + 'msd_idv.pdf')

    sns.set_palette([sns.xkcd_rgb["grey"], sp.SUSSEX_CORAL_RED, sp.SUSSEX_COBALT_BLUE])
    fig = plt.figure()
    fig.set_size_inches((3.5, 3.5))
    ax = plt.gca()
    msd.track_average(ax)
    fig.savefig(p.out_dir + 'msd_avg.pdf')

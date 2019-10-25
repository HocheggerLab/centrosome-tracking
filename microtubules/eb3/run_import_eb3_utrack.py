import logging
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import tifffile as tf

import tools.image as image
import parameters as p
import mechanics as m

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
pd.set_option('display.width', 320)
indiv_idx = ['condition', 'tag', 'particle']


def import_eb3_utrack_all(dir_base):
    def import_eb3_matlab(gen_filename, trk_filename, tag=None, limit=None):
        g = sio.loadmat(gen_filename, squeeze_me=True)
        x = sio.loadmat(trk_filename, squeeze_me=True)
        time_interval = g['timeInterval']
        tracks = x['tracksFinal']
        tracked_particles = tracks['tracksCoordAmpCG']
        trk_events = tracks['seqOfEvents']
        num_tracks = tracked_particles.shape[0]
        df_out = pd.DataFrame()
        logging.info('tracked_particles.shape=%s  trk_events.shape=%s' % (tracked_particles.shape, trk_events.shape))

        for ti, (_trk, _ev) in enumerate(zip(tracked_particles, trk_events)):
            if ti == limit:
                logging.debug('breaking on limit')
                break
            if _ev.shape == (2, 4):
                ini_frame = _ev[0, 0] if _ev[0, 1] == 1 else None
                end_frame = _ev[1, 0] + 1 if _ev[1, 1] == 2 else None
                if ini_frame is not None and end_frame is not None:
                    trk = np.reshape(_trk, [len(_trk) / 8, 8])
                    _df = pd.DataFrame(data=trk[:, 0:2], columns=['x', 'y'])
                    _df.loc[:, 'particle'] = ti
                    _df.loc[:, 'frame'] = np.arange(ini_frame, end_frame, 1, np.int16)
                    _df.loc[:, 'time'] = _df['frame'] * time_interval

                    df_out = df_out.append(_df)
                else:
                    raise Exception('Invalid event format.')

        if tag is not None:
            df_out['tag'] = tag

        return df_out.reset_index().drop('index', axis=1)

    logging.info('importing data from %s' % dir_base)
    df_matlab = pd.DataFrame()
    # 1st level = conditions
    citems = [d for d in os.listdir(dir_base) if d[0] != '.']
    for cit in citems:
        cpath = os.path.join(dir_base, cit)
        run_i = 1
        if os.path.isdir(cpath):
            # 2nd level = dates
            ditems = [d for d in os.listdir(cpath) if d[0] != '.']
            for dit in ditems:
                dpath = os.path.join(cpath, dit)
                if os.path.isdir(dpath):
                    # 3rd level = results
                    ritems = [d for d in os.listdir(dpath) if d[0] != '.']
                    for rit in ritems:
                        rpath = os.path.join(dpath, rit)
                        if os.path.isdir(rpath):
                            # 4rd level = matlab result file
                            mitems = [d for d in os.listdir(rpath) if d[0] != '.']
                            for mit in mitems:
                                mpath = os.path.join(rpath, mit)
                                if os.path.isfile(mpath) and mit != 'time.mat':
                                    logging.info('importing %s' % mpath)

                                    # Import
                                    try:
                                        dir_data = rpath
                                        genfname = os.path.join(dir_data, 'time.mat')
                                        trkfname = dir_data + '/TrackingPackage/tracks/Channel_1_tracking_result.mat'
                                        imgname = os.path.join(dpath, mit[:-4] + '.tif')
                                        with tf.TiffFile(imgname, fastij=True) as tif:
                                            if tif.is_imagej is not None:
                                                res = 'n/a'
                                                if tif.pages[0].resolution_unit == 'centimeter':
                                                    # asuming square pixels
                                                    xr = tif.pages[0].x_resolution
                                                    res = float(xr[0]) / float(xr[1])  # pixels per cm
                                                    res = res / 1e4  # pixels per um
                                                elif tif.pages[0].imagej_tags.unit == 'micron':
                                                    # asuming square pixels
                                                    xr = tif.pages[0].x_resolution
                                                    res = float(xr[0]) / float(xr[1])  # pixels per um
                                        df_mtlb = import_eb3_matlab(genfname, trkfname, tag='%s-%d' % (cit, run_i))
                                        df_mtlb.loc[:, 'condition'] = cit
                                        run_i += 1
                                        # Process
                                        df_mtlb['x'] /= res
                                        df_mtlb['y'] /= res
                                        df_matlab = df_matlab.append(df_mtlb)
                                    except IOError as ioe:
                                        logging.warning('could not import due to IO error: %s' % ioe)
    # compute speed and msd
    df_matlab = m.get_speed_acc(df_matlab, group=indiv_idx)
    df_matlab = m.get_msd(df_matlab, group=indiv_idx)
    return df_matlab


def import_eb3_icy_all(dir_base):
    logging.info('importing data from %s' % dir_base)
    df_matlab = pd.DataFrame()
    # Traverse through all subdirs looking for excel files. When a file is found, assume folder structure of (cond/date)
    for root, directories, files in os.walk(dir_base):
        for f in files:
            mpath = os.path.join(root, f)
            if os.path.isfile(mpath) and f[-4:] == '.xls' and f[:6] == 'Result':
                logging.info('processing %s' % mpath)
                try:  # Import
                    df_mtlb = pd.read_excel(mpath, header=None,
                                            names=['nn', 'particle', 'frame', 'x', 'y', 'z', 'virtual'])
                    df_mtlb.iloc[:, 1] = df_mtlb.iloc[:, 1].fillna(method='ffill')
                    df_mtlb = df_mtlb[df_mtlb['virtual'] == 0]
                    df_mtlb = df_mtlb.drop(['nn', 'virtual', 'z'], axis=1)

                    iname = f[:-11] + '.tif'
                    logging.debug('trying to find image %s' % iname)
                    img, res, dt, _, _ = image.find_image(iname, root)
                    df_mtlb['time'] = df_mtlb['frame'] * dt
                    df_mtlb['x'] /= res
                    df_mtlb['y'] /= res
                    df_mtlb['condition'] = os.path.basename(os.path.dirname(root))
                    df_mtlb['tag'] = f[10:-11]  # take "Result of" and extension out of the filename

                    df_matlab = df_matlab.append(df_mtlb)
                except IOError as ioe:
                    logging.warning('could not import due to IO error: %s' % ioe)

    df_matlab['frame'] = df_matlab['frame'].astype('int32')
    df_matlab[['x', 'y', 'time']] = df_matlab[['x', 'y', 'time']].astype('float64')
    # df_matlab['tag'] = pd.factorize(df_matlab['tag'], sort=True)[0] + 1
    # df_matlab['tag'] = df_matlab['tag'].astype('category')
    return df_matlab


if __name__ == '__main__':
    df = import_eb3_utrack_all(p.experiments_dir + 'eb3')
    df.to_pickle(p.experiments_dir + 'eb3.pandas')

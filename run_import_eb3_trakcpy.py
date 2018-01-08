import logging
import os
import sys

import numpy as np
import pandas as pd
import tifffile as tf
import trackpy as tp
import trackpy.predict

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
pd.set_option('display.width', 320)


def get_dataframe(image_path):
    with tf.TiffFile(image_path, fastij=True) as tif:
        if tif.is_imagej is not None:
            dt = tif.pages[0].imagej_tags.finterval
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

            frames = tif.pages[0].asarray()
            diam = np.ceil(1 * res) // 2 * 2 + 1

            f = tp.batch(frames[2:], diam, invert=True, minmass=200)
            pred = trackpy.predict.NearestVelocityPredict()
            t = pred.link_df(f, 5)

            t.drop(['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep'], axis=1, inplace=True)
            t['time'] = t['frame'] * dt
            t['x'] /= res
            t['y'] /= res
            t['frame'] = t['frame'].astype('int32')
            t[['x', 'y', 'time']] = t[['x', 'y', 'time']].astype('float64')
            return t


def process_dir(dir_base):
    logging.info('processing data from folder %s' % dir_base)
    df = pd.DataFrame()
    # Traverse through all subdirs looking for iimage files. When a file is found, assume folder structure of (cond/date)
    for root, directories, files in os.walk(dir_base):
        for f in files:
            mpath = os.path.join(root, f)
            if os.path.isfile(mpath) and f[-4:] == '.tif' and f[:6] == 'Result':
                logging.info('processing tag %s in folder %s' % (f[10:-4], root))
                try:  # process
                    tdf = get_dataframe(mpath)
                    tdf['condition'] = os.path.basename(os.path.dirname(root))
                    tdf['tag'] = f[10:-4]  # take "Result of" and extension out of the filename
                    df = df.append(tdf)
                except IOError as ioe:
                    logging.warning('could not import due to IO error: %s' % ioe)

    # df_matlab['tag'] = pd.factorize(df_matlab['tag'], sort=True)[0] + 1
    # df_matlab['tag'] = df_matlab['tag'].astype('category')
    return df


if __name__ == '__main__':
    _fig_size_A3 = (11.7, 16.5)
    _err_kws = {'alpha': 0.3, 'lw': 1}

    df = process_dir('/Users/Fabio/data/lab/eb3')
    df.to_pickle('/Users/Fabio/data/lab/eb3.pandas')

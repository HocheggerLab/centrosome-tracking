import logging
import os
import re
import sys

import coloredlogs
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import tifffile as tf
import trackpy as tp
import trackpy.diag
import trackpy.predict

coloredlogs.install(fmt='%(levelname)s:%(funcName)s - %(message)s', level=logging.INFO)
tp.diag.performance_report()
logging.basicConfig(stream=sys.stderr, format='%(levelname)s:%(funcName)s - %(message)s', level=logging.INFO)
coloredlogs.install()
trackpy.quiet()
pd.set_option('display.width', 320)


def do_trackpy(image_path):
    with tf.TiffFile(image_path, fastij=True) as tif:
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

        # construct frames array based on tif file structure:
        frames = None
        if len(tif.pages) == 1:
            frames = np.int32(tif.pages[0].asarray())
        elif len(tif.pages) > 1:
            frames = np.ndarray((len(tif.pages), tif.pages[0].image_length, tif.pages[0].image_width), dtype=np.int32)
            for i, page in enumerate(tif.pages):
                frames[i] = page.asarray()

        # subtract first frame and deal with negative results after the operation
        # frames = np.int32(frames)
        # frames -= frames[0]
        # frames = frames[1:, :, :]
        # frames = np.uint16(frames.clip(0))
        diam = np.ceil(1 * res) // 2 * 2 + 1

        logging.info(scipy.stats.describe(frames, axis=None))
        f = tp.batch(frames, diam, separation=diam, characterize=True)
        stat = f[['mass', 'size']].describe()
        f1 = f[((f['mass'] > stat['mass']['25%']) & (f['size'] < 1.5))]
        logging.info(stat)

        pred = trackpy.predict.DriftPredict()
        search_rng_px = 4
        # search_rng_px=np.ceil(1 * res)
        t = pred.link_df(f1, search_rng_px)

        return t


def process_dir(dir_base):
    logging.info('processing data from folder %s' % dir_base)
    df = pd.DataFrame()
    cal = pd.read_excel('/Users/Fabio/data/lab/eb3/eb3_calibration.xls')
    # Traverse through all subdirs looking for image files. When a file is found, assume folder structure of (cond/date)
    for root, directories, files in os.walk(dir_base):
        for f in files:
            mpath = os.path.join(root, f)
            if os.path.isfile(mpath) and f[-4:] == '.tif':
                logging.info('processing file %s in folder %s' % (f, root))
                try:  # process
                    tdf = do_trackpy(mpath)
                    tdf['condition'] = os.path.basename(os.path.dirname(root))
                    tdf['tag'] = f[:-4]
                    # tdf.drop(['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep'], axis=1, inplace=True)

                    calp = cal[cal['filename'] == f].iloc[0]
                    tdf['time'] = tdf['frame'] * calp['dt']
                    tdf['x'] /= calp['resolution']
                    tdf['y'] /= calp['resolution']

                    tdf['particle'] = tdf['particle'].astype('int32')
                    tdf['frame'] = tdf['frame'].astype('int32')
                    tdf[['x', 'y', 'time']] = tdf[['x', 'y', 'time']].astype('float64')

                    # consider 1.6X magification of optivar system
                    if calp['optivar'] == 'yes':
                        tdf['x'] /= 1.6
                        tdf['y'] /= 1.6

                    df = df.append(tdf)
                except IOError as ioe:
                    logging.warning('could not import due to IO error: %s' % ioe)

    # df_matlab['tag'] = pd.factorize(df_matlab['tag'], sort=True)[0] + 1
    # df_matlab['tag'] = df_matlab['tag'].astype('category')
    return df


def optivar_resolution_to_excel(dir_base):
    logging.info('constructing optivar reference from folder %s' % dir_base)
    df = pd.DataFrame()
    # Traverse through all subdirs looking for image files. When a file is found, assume folder structure of (cond/date)
    for root, directories, files in os.walk(dir_base):
        for f in files:
            mpath = os.path.join(root, f)
            if os.path.isfile(mpath) and f[-4:] == '.tif':
                logging.info('file %s in folder %s' % (f, root))
                # i['tag'] = f[10:-4]  # take "Result of" and extension out of the filename
                condition = os.path.basename(os.path.dirname(root))

                with tf.TiffFile(mpath, fastij=True) as tif:
                    has_image_meta = tif.is_imagej is not None
                    has_meta_in_log = np.any([i[-4:] == '.log' for i in os.listdir(root) if f[:-14] in i])
                    if has_image_meta or has_meta_in_log:
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

                        dt = 'n/a'
                        if has_image_meta:
                            dt = tif.pages[0].imagej_tags.finterval
                        elif has_meta_in_log:
                            file_log = [i for i in os.listdir(root) if f[:-14] in i and i[-4:] == '.log'][0]
                            with open(os.path.join(root, file_log), 'r') as log:
                                for line in log:
                                    search = re.search('^Average Timelapse Interval: ([0-9.]+) ms', line)
                                    if search is not None:
                                        dt = eval(search.group(1)) / 1000.0
                        date = os.path.basename(root)
                        i = pd.DataFrame(data=[[condition, date, f, dt, res, 'no']],
                                         columns=['condition', 'date', 'filename', 'dt', 'resolution', 'optivar'])

                        df = df.append(i)

    excel_file = os.path.join(dir_base, 'eb3_calibration.xls')
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    df.to_excel(writer, 'calibration', index=False)
    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets['calibration']
    float_format = workbook.add_format({'num_format': '#,##0.00', 'align': 'center'})
    center_format = workbook.add_format({'align': 'center'})

    # Set the column width and format.
    worksheet.set_column('A:A', 13)
    worksheet.set_column('B:B', 8)
    worksheet.set_column('C:C', 75)
    worksheet.set_column('D:D', 10, float_format)
    worksheet.set_column('E:E', 10, float_format)
    worksheet.set_column('F:F', 10, center_format)
    writer.save()

    return df


if __name__ == '__main__':
    _fig_size_A3 = (11.7, 16.5)
    _err_kws = {'alpha': 0.3, 'lw': 1}

    # df = optivar_resolution_to_excel('/Users/Fabio/data/lab/eb3')
    df = process_dir('/Users/Fabio/data/lab/eb3')
    df.to_pickle('/Users/Fabio/data/lab/eb3.pandas')

    # process dataframe and render images
    from microtubules.eb3 import run_plots_eb3

    logging.info('filtering using run_plots_eb3.')
    df, df_avg = run_plots_eb3.batch_filter(df)
    logging.info('rendering images.')
    run_plots_eb3.render_image_tracks(df, folder='/Users/Fabio/data/lab/eb3')

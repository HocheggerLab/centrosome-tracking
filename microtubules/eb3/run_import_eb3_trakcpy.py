import logging
import os
import re
import warnings

import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

import microtubules.eb3.detection as detection
from microtubules.eb3 import aster
import parameters as p
from microtubules.eb3.filters import Wheel
import trackpy.diag

warnings.filterwarnings("ignore", category=DeprecationWarning)
# coloredlogs.install(fmt='%(levelname)s:%(funcName)s - %(message)s', level=logging.DEBUG)
trackpy.diag.performance_report()
# logging.basicConfig(stream=sys.stderr, format='%(levelname)s:%(funcName)s - %(message)s', level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
tp.quiet()
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 50)


def movie(particles, filename="movie.mp4"):
    def make_frame_mpl(t):
        fr = int(t)

        ax.cla()

        # for ix, gr in linked_particles.groupby('particle'):
        #     ax.plot(gr['xum'], gr['yum'], lw=1, c='white', alpha=0.5)
        #     ax.scatter(gr.loc[gr['frame'] == fr, 'xum'], gr.loc[gr['frame'] == fr, 'yum'], s=1, c='red')
        particles.render_image(ax, frame=fr)
        particles.render_segmented_image(ax, frame=fr, color=[1., 0., 0.])
        particles.render_detected_features(ax, frame=fr, alpha=0.4, lines=True)
        particles.render_linked_features(ax, frame=fr)
        for xb, yb in particles.wheel.best:
            particles.wheel.plot(xb, yb, ax=ax)

        return mplfig_to_npimage(fig)  # RGB image of the figure

    logging.info("rendering movie %s" % filename)
    fig = plt.figure(20, dpi=300)
    ax = fig.gca()
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")

    animation = mpy.VideoClip(make_frame_mpl, duration=len(particles.images) - 1)
    animation.write_videofile(filename, fps=1)
    animation.close()


def do_tracking(image_path, asters=None):
    log.info("detecting points.")
    d = detection.Particles(image_path)

    # Estimate centrosome asters automatically if not provided by parameter
    wheel = Wheel(d.df, np.max(d.images, axis=0), radius=15)
    if asters is not None:
        for a in asters:
            wheel.add(a[0], a[1])
    d.wheel = wheel

    fig = plt.figure(dpi=300)
    ax = fig.gca()
    d.render_image(ax)
    d.render_detected_features(ax)
    d.render_linked_features(ax, wheel=True)
    fig.savefig('%s_objfn.png' % image_path[:-4])

    movie(d, filename='%s.mp4' % image_path[:-4])

    return d.linked


def process_dir(dir_base):
    logging.info('processing data from folder %s' % dir_base)
    df = pd.DataFrame()

    # Traverse through all subdirs looking for image files. When a file is found, assume folder structure of (cond/date)
    for root, directories, files in os.walk(dir_base):
        for f in files:
            mpath = os.path.join(root, f)
            if os.path.isfile(mpath) and f[-4:] == '.tif':
                logging.info('processing file %s in folder %s' % (f, root))
                csv_file = os.path.join(root, '%s.csv' % f[:-4])
                if os.path.exists(csv_file):
                    log.warning(
                        "found a csv file with the same name of the one i'm trying to create, reading file instead of running tracking algorithm.")
                    log.warning("file: %s" % csv_file)
                    df = df.append(pd.read_csv(csv_file))
                    continue
                try:  # process
                    log.info('processing %s' % f)
                    cfg_file = os.path.join(root, '%s.cfg' % f[:-4])
                    if os.path.exists(cfg_file):
                        asters = aster.read_aster_config(cfg_file)
                    else:
                        asters = aster.select_asters(mpath)
                        aster.write_aster_config(cfg_file, asters)

                    tdf = do_tracking(mpath, asters=asters)
                    if __debug__: exit()
                    if tdf.empty: continue

                    tdf['condition'] = os.path.basename(os.path.dirname(root))
                    tdf['tag'] = f[:-4]

                    tdf.to_csv(csv_file, index=False)
                    df = df.append(tdf)
                except IOError as ioe:
                    logging.warning('could not import due to IO error: %s' % ioe)

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
    df = process_dir(p.experiments_dir + 'eb3')
    df.to_pickle(p.experiments_dir + 'eb3.pandas')

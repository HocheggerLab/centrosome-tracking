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
import skimage.filters as filters
import skimage.morphology as morphology
import skimage.color as color
from matplotlib import cm

import microtubules.eb3.detection as detection
from tools.draggable import DraggableCircle
from microtubules.eb3 import aster
import parameters as p
import plot_special_tools as sp
from microtubules.eb3.filters import Wheel

warnings.filterwarnings("ignore", category=DeprecationWarning)
# coloredlogs.install(fmt='%(levelname)s:%(funcName)s - %(message)s', level=logging.DEBUG)
# tp.diag.performance_report()
# logging.basicConfig(stream=sys.stderr, format='%(levelname)s:%(funcName)s - %(message)s', level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
tp.quiet()
tp.linking.Linker.MAX_SUB_NET_SIZE = 50
pd.set_option('display.width', 320)


def movie(frames, linked_particles, wheel, filename="movie.mp4", pix_per_um=1):
    def make_frame_mpl(t):
        fr = int(t)

        ax.cla()

        ax.imshow(frames[fr], cmap=cm.gray, interpolation='none', extent=xtnt)
        ax.imshow(nobkg[fr], interpolation='none', extent=xtnt, alpha=0.4)
        # for ix, gr in linked_particles.groupby('particle'):
        #     ax.plot(gr['xum'], gr['yum'], lw=1, c='white', alpha=0.5)
        #     ax.scatter(gr.loc[gr['frame'] == fr, 'xum'], gr.loc[gr['frame'] == fr, 'yum'], s=1, c='red')
        lp = linked_particles
        try:
            ax.scatter(lp.loc[lp['frame'] == fr, 'x'], lp.loc[lp['frame'] == fr, 'y'], s=2, marker='+', c='green')
            # ax.scatter(lp.loc[lp['frame'] == fr, 'x1'], lp.loc[lp['frame'] == fr, 'y1'], s=1, c='magenta', zorder=10)
            # ax.scatter(lp.loc[lp['frame'] == fr, 'x2'], lp.loc[lp['frame'] == fr, 'y2'], s=1, c='blue', zorder=10)
            ax.plot([lp.loc[lp['frame'] == fr, 'x1'], lp.loc[lp['frame'] == fr, 'x2']],
                    [lp.loc[lp['frame'] == fr, 'y1'], lp.loc[lp['frame'] == fr, 'y2']], lw=0.5, alpha=0.3, c='white')
        except Exception as e:
            pass
        for xb, yb in wheel.best:
            wheel.plot(xb, yb, ax=ax)

        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
        ax.set_xlim([0, w / pix_per_um])
        ax.set_ylim([0, h / pix_per_um])

        return mplfig_to_npimage(fig)  # RGB image of the figure

    # subtract first frame and deal with negative results after the operation
    nobkg = np.int32(frames)
    nobkg -= nobkg[0]
    nobkg = nobkg[1:, :, :]
    nobkg = np.uint16(nobkg.clip(0))

    w, h = frames.shape[1], frames.shape[2]
    xtnt = [0, w / pix_per_um, h / pix_per_um, 0]

    # path, fim = os.path.split(image_path)
    # linked_particles['frame'] += 1

    thr_lvl = filters.threshold_otsu(nobkg)
    nobkg = (nobkg >= thr_lvl).astype(bool)
    morphology.remove_small_objects(nobkg, min_size=4 * pix_per_um, connectivity=1, in_place=True)

    nobkg = color.gray2rgb(nobkg)
    nobkg = nobkg * [1., 0., 0.]

    logging.info("rendering movie %s" % filename)
    fig = plt.figure(20, dpi=300)
    ax = fig.gca()
    # fig.tight_layout()

    animation = mpy.VideoClip(make_frame_mpl, duration=len(frames) - 1)
    animation.write_videofile(filename, fps=1)
    animation.close()


def do_tracking(image_path, asters=None):
    log.info("detecting points.")
    d = detection.Particles(image_path)
    df = d.df
    log.info("detection step completed.")

    # Estimate centrosome asters automatically if not provided by parameter
    wheel = Wheel(df, np.max(d.images, axis=0), radius=15)
    if asters is not None:
        for a in asters:
            wheel.add(a[0], a[1])

    fig = plt.figure(dpi=300)
    ax = fig.gca()

    d.render_time_projection(ax)
    d.render_detected_features(ax)

    linked = pd.DataFrame()
    for xb, yb in wheel.best:
        # wheel.plot(xb, yb, ax=ax)
        _fil = wheel.filter_wheel(xb, yb, ax=ax)
        if _fil.empty: continue

        search_range = 1
        pred = tp.predict.NearestVelocityPredict(initial_guess_vels=0.5)
        linked = linked.append(pred.link_df(_fil, search_range), sort=False)

    fig.savefig('%s_objfn.png' % image_path[:-4])

    if linked.empty: return linked

    #  filter spurious tracks
    frames_per_particle = linked.groupby('particle')['frame'].nunique()
    particles = frames_per_particle[frames_per_particle > 5].index
    linked = linked[linked['particle'].isin(particles)]
    logging.info('filtered %d particles by track length' % linked['particle'].nunique())

    # m = tp.imsd(linked, 1, 1)
    # mt = m.ix[15]
    # particles = mt[mt > 1].index
    # linked = linked[linked['particle'].isin(particles)]
    # logging.info('filtered %d particles msd' % linked['particle'].nunique())

    movie(d.images, linked, wheel, filename='%s.mp4' % image_path[:-4], pix_per_um=d.pix_per_um)

    return linked


def select_asters(image_path):
    def on_key(event):
        print('press', event.key)
        if event.key == 'c':
            ci = DraggableCircle(plt.Circle(xy=orig, radius=2, fc='g', picker=5))
            asters.append(ci)
            ax.add_artist(ci.circle)
            ci.connect()
            fig.canvas.draw()

    images, pix_per_um, dt = sp.load_tiff(image_path)
    w, h = images[0].shape[0], images[0].shape[1]

    fig = plt.figure()
    ax = fig.gca()
    ext = [0, w / pix_per_um, h / pix_per_um, 0]
    ax.imshow(np.max(images, axis=0), interpolation='none', extent=ext, cmap=cm.gray)
    orig = (w / 2 / pix_per_um, h / 2 / pix_per_um)

    ci = DraggableCircle(plt.Circle(xy=orig, radius=2, fc='g', picker=5))
    asters = [ci]
    ax.add_artist(ci.circle)
    ci.connect()

    cidkeyboard = fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.draw()
    plt.show()

    return [a.circle.center for a in asters]


def process_dir(dir_base):
    logging.info('processing data from folder %s' % dir_base)
    df = pd.DataFrame()
    cal = pd.read_excel(p.experiments_dir + 'eb3/eb3_calibration.xls')
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
                        asters = select_asters(mpath)
                        aster.write_aster_config(cfg_file, asters)

                    # vis_detection(mpath, filename='%s.mp4' % mpath[:-4])
                    tdf = do_tracking(mpath, asters=asters)
                    if __debug__: exit()
                    if tdf.empty: continue

                    tdf['condition'] = os.path.basename(os.path.dirname(root))
                    tdf['tag'] = f[:-4]
                    # tdf.drop(['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep'], axis=1, inplace=True)

                    calp = cal[cal['filename'] == f].iloc[0]
                    tdf['time'] = tdf['frame'] * calp['dt']

                    tdf['particle'] = tdf['particle'].astype('int32')
                    tdf['frame'] = tdf['frame'].astype('int32')
                    tdf[['x', 'y', 'time']] = tdf[['x', 'y', 'time']].astype('float64')

                    # consider 1.6X magification of optivar system
                    if calp['optivar'] == 'yes':
                        tdf['x'] /= 1.6
                        tdf['y'] /= 1.6

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

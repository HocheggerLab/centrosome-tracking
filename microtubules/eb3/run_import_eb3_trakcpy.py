import logging
import os
import re
import warnings

import pandas as pd
import scipy.stats
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
from scipy import ndimage
from shapely.geometry import LineString, Point
import cv2

from tools.draggable import DraggableCircle
from microtubules.eb3 import aster
import parameters as p
import plot_special_tools as sp
from microtubules.eb3.filters import Wheel
from tools.terminal import printProgressBar

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


def detection(images, pix_per_um=1):
    features = list()
    # subtract first frame and deal with negative results after the operation
    nobkg = np.int32(images)
    nobkg -= nobkg[0]
    nobkg = nobkg[1:, :, :]
    nobkg = np.uint16(nobkg.clip(0))

    thr_lvl = filters.threshold_otsu(nobkg)
    nobkg = (nobkg >= thr_lvl).astype(bool)
    morphology.remove_small_objects(nobkg, min_size=4 * pix_per_um, connectivity=1, in_place=True)

    w, h = images[0].shape
    blackboard = np.zeros(shape=images[0].shape, dtype=np.uint8)
    for num, im in enumerate(nobkg):
        labels = ndimage.label(im)[0]

        for (i, l) in enumerate([l for l in np.unique(labels) if l > 0]):
            # find contour of mask
            blackboard[labels == l] = 255
            cnt = cv2.findContours(blackboard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2][0]
            blackboard[labels == l] = 0
            if cnt.shape[0] < 5: continue
            # if region.eccentricity < 0.8: continue
            ellipse = cv2.fitEllipse(cnt)
            # print(ellipse)
            (x0, y0), (we, he), angle_deg = ellipse  # angle_rad goes from 0 to 180
            x0, y0, we, he = np.array([x0, y0, we, he]) / pix_per_um
            angle_rad = np.deg2rad(angle_deg - 90)  # angle_rad goes from -pi/2 to pi/2

            # if num > 2 and 50 < x0 < 100 and 50 < y0 < 100:
            #     from matplotlib.patches import Ellipse
            #
            #     fig = plt.figure(dpi=100)
            #     ax = fig.gca()
            #     ext = [0, w / pix_per_um, h / pix_per_um, 0]
            #     ax.imshow(images[num], interpolation='none', extent=ext)
            #     # blackboard[labels == l] = 255
            #     # ax.imshow(blackboard, interpolation='none', extent=ext, alpha=0.5)
            #     ax.imshow(im, interpolation='none', extent=ext, alpha=0.3)
            #
            #     ell = Ellipse(xy=(x0, y0), angle=angle_deg,
            #                   width=we, height=he,
            #                   facecolor='gray', alpha=0.5)
            #     ax.add_artist(ell)
            #
            #     x1 = x0 + np.cos(angle_rad) * 0.5 * he
            #     y1 = y0 + np.sin(angle_rad) * 0.5 * he
            #     x2 = x0 + np.sin(angle_rad) * 0.5 * we
            #     y2 = y0 - np.cos(angle_rad) * 0.5 * we
            #     ax.plot((x0, x1), (y0, y1), '-y', linewidth=1, zorder=15)  # major axis
            #     ax.plot((x0, x2), (y0, y2), '-r', linewidth=1, zorder=15)  # minor axis
            #     ax.plot(x0, y0, '.g', markersize=10, zorder=20)
            #
            #     l = he / 2
            #     l_sint, l_cost = np.sin(angle_rad) * l, np.cos(angle_rad) * l
            #     xx1, yy1 = x0 + l_cost, y0 + l_sint
            #     xx2, yy2 = x0 - l_cost, y0 - l_sint
            #     ax.plot((x0, xx1), (y0, yy1), color='magenta', linewidth=2, zorder=10)  # minor axis
            #     ax.plot((x0, xx2), (y0, yy2), color='blue', linewidth=2, zorder=10)  # major axis
            #
            #     x, y, wbr, hbr = np.array(cv2.boundingRect(cnt)) / pix_per_um
            #     bx = (x, x + wbr, x + wbr, x, x)
            #     by = (y, y, y + hbr, y + hbr, y)
            #     ax.plot(bx, by, '-b', linewidth=1)
            #
            #     ax.text(x + wbr, y + hbr, "orig: %0.2f" % angle_deg, color="white")
            #     ax.text(x + wbr, y, "tran: %0.2f" % angle_rad, color="white")
            #
            #     ax.set_aspect('equal')
            #     ax.set_xlim([x - wbr, x + 2 * wbr])
            #     ax.set_ylim([y - hbr, y + 2 * hbr])
            #
            #     plt.show()

            l = he / 2
            l_sint, l_cost = np.sin(angle_rad) * l, np.cos(angle_rad) * l
            xx1, yy1 = x0 + l_cost, y0 + l_sint
            xx2, yy2 = x0 - l_cost, y0 - l_sint

            features.append({'x': x0, 'y': y0,
                             'pt1': Point((xx1, yy1)), 'pt2': Point((xx2, yy2)),
                             'l': LineString([(xx1, yy1), (xx2, yy2)]),
                             'x1': xx1, 'y1': yy1, 'x2': xx2, 'y2': yy2,
                             'theta': angle_rad, 'frame': num})

    return pd.DataFrame(features)


def do_tracking(image_path, asters=None):
    images, pix_per_um, dt = sp.load_tiff(image_path)
    w, h = images[0].shape[0], images[0].shape[1]

    log.info("detecting points.")
    f = detection(images, pix_per_um=pix_per_um)
    # f.to_pickle('debug_f.pandas')
    # f = pd.read_pickle('debug_f.pandas')
    log.info("detection step completed.")

    # Estimate centrosome asters automatically if not provided by parameter
    wheel = Wheel(f, np.max(images, axis=0), radius=15)
    if asters is not None:
        for a in asters:
            wheel.add(a[0], a[1])
    else:
        x = np.arange(10, w / pix_per_um - 10, 5)
        y = np.arange(10, h / pix_per_um - 10, 5)
        xx, yy = np.meshgrid(x, y)
        obj = np.zeros((x.size, y.size), dtype=np.float32)
        tot = x.size * y.size - 1
        printProgressBar(0, tot, prefix='Progress:', suffix='', length=50)
        for i in range(x.size):
            for j in range(y.size):
                lines_per_cuadrant = wheel.count_lines_with_same_slope(xx[j, i], yy[j, i])
                obj[j, i] = min(lines_per_cuadrant)
                # Update Progress Bar
                printProgressBar(i * x.size + j, tot, prefix='Progress:', suffix=' Fn=%d' % obj[j, i], length=50)
        # np.savetxt('obj.csv', obj)
        # obj = np.loadtxt('obj.csv')

        p_high = np.percentile(obj.ravel(), 95)
        log.debug("selecting Fn higher than %d" % p_high)
        log.debug(scipy.stats.describe(obj.ravel()))
        for i in range(x.size):
            for j in range(y.size):
                if obj[j, i] > p_high:
                    wheel.add(xx[j, i], yy[j, i])
                # Update Progress Bar
                # printProgressBar(i * x.size + j, x.size * y.size, prefix='Progress:', suffix='Complete. Fn=%d' % obj[j, i],
                #                  length=50)
        log.debug("%d points selected" % len(wheel.best))

    fig = plt.figure(dpi=300)
    ax = fig.gca()
    ax.set_facecolor(sp.colors.sussex_cobalt_blue)
    ext = [0, w / pix_per_um, h / pix_per_um, 0]
    ax.imshow(np.max(images, axis=0), interpolation='none', extent=ext, cmap=cm.gray)

    linked = pd.DataFrame()
    for xb, yb in wheel.best:
        # wheel.plot(xb, yb, ax=ax)
        _fil = wheel.filter_wheel(xb, yb, ax=ax)
        if _fil.empty: continue

        search_range = 1
        pred = tp.predict.NearestVelocityPredict(initial_guess_vels=0.5)
        linked = linked.append(pred.link_df(_fil, search_range), sort=False)

    ax.set_aspect('equal')
    ax.set_xlim([0, w / pix_per_um])
    ax.set_ylim([0, h / pix_per_um])
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

    movie(images, linked, wheel, filename='%s.mp4' % image_path[:-4], pix_per_um=pix_per_um)

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
                    # if __debug__: exit()
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

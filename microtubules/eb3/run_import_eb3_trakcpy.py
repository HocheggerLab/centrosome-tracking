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
import skimage.draw as draw
import skimage.filters as filters
import skimage.morphology as morphology
import skimage.measure as measure
import skimage.color as color
from matplotlib import cm
from scipy import ndimage
from shapely.geometry import LineString, Point

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
pd.set_option('display.width', 320)


def movie(frames, linked_particles, objective_fn, n=16, filename="movie.mp4", pix_per_um=1):
    def make_frame_mpl(t, n=16):
        fr = int(t)

        ax.cla()
        im = render[fr]
        w, h = im.shape[0], im.shape[1]

        ax.imshow(im, cmap=cm.gray, interpolation='none', extent=[0, w / pix_per_um, h / pix_per_um, 0])
        # for ix, gr in linked_particles.groupby('particle'):
        #     ax.plot(gr['xum'], gr['yum'], lw=1, c='white', alpha=0.5)
        #     ax.scatter(gr.loc[gr['frame'] == fr, 'xum'], gr.loc[gr['frame'] == fr, 'yum'], s=1, c='red')
        lp = linked_particles
        ax.scatter(lp.loc[lp['frame'] == fr, 'x'], lp.loc[lp['frame'] == fr, 'y'], s=4, c='green')
        # ax.scatter(lp.loc[lp['frame'] == fr, 'x1'], lp.loc[lp['frame'] == fr, 'y1'], s=1, c='red')
        # ax.scatter(lp.loc[lp['frame'] == fr, 'x2'], lp.loc[lp['frame'] == fr, 'y2'], s=1, c='blue')
        ax.plot([lp.loc[lp['frame'] == fr, 'x1'], lp.loc[lp['frame'] == fr, 'x2']],
                [lp.loc[lp['frame'] == fr, 'y1'], lp.loc[lp['frame'] == fr, 'y2']], lw=0.5, c='white')

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

    # path, fim = os.path.split(image_path)
    linked_particles['frame'] += 1
    thr_lvl = filters.threshold_otsu(nobkg)
    nobkg = nobkg >= thr_lvl
    morphology.remove_small_objects(nobkg, min_size=2 * pix_per_um, in_place=True)
    nobkg = color.gray2rgb(nobkg)
    # frames = color.gray2rgb(frames)
    # render = frames[1:] * 0.2 + nobkg * sp.colors.hoechst_33342 * 0.4
    render = nobkg * sp.colors.alexa_594

    logging.info("rendering movie %s" % filename)
    fig = plt.figure(20)
    ax = fig.gca()
    fig.tight_layout()

    animation = mpy.VideoClip(make_frame_mpl, duration=len(render))
    animation.write_videofile(filename, fps=1)
    animation.close()


def vis_detection(image_path, filename="movie.mp4"):
    def make_frame_mpl(t):
        fr = int(t)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.cla()
        # im = _images[fr]
        im = frames[fr]
        # im = exposure.equalize_hist(im)

        # edges = canny(im, 2, 1, 25)
        # median=rank.median(image=im, selem=disk(0.2*pix_per_um))
        # ax1.imshow(median, cmap=cm.gray)
        ax1.imshow(_images[fr], cmap=cm.gray)

        # p2 = np.percentile(im, 15)
        # p98 = np.percentile(im, 100)
        # im = exposure.rescale_intensity(im, in_range=(p2, p98))
        ax2.imshow(im, cmap=cm.gray)

        # low = 0.1
        # high = 0.35
        # thr_lvl = filters.apply_hysteresis_threshold(im, low, high)
        thr_lvl = filters.threshold_yen(im)
        thresh = im >= thr_lvl
        # selem = morphology.disk(pix_per_um / 4)
        # thresh = morphology.opening(thresh, selem)
        morphology.remove_small_objects(thresh, min_size=1 * pix_per_um, in_place=True)
        ax3.imshow(thresh, cmap=cm.gray)

        w, h = im.shape
        labels = ndimage.label(thresh)[0]
        # image_label_overlay = color.label2rgb(labels, image=im)
        out = np.zeros((w, h, 3), dtype=np.double)

        # ax3.imshow(image_label_overlay, cmap=cm.gray)
        ax3.text(0, 0, '%d' % t)

        for k, region in enumerate(measure.regionprops(labels, coordinates='rc', cache=True)):
            # if k!=100: continue
            # if region.eccentricity < 0.9: continue
            rp, cp = draw.polygon(region.coords[:, 0], region.coords[:, 1], im.shape)
            # out[rp, cp, :] = (0,1,0)

            # rotate = transform.SimilarityTransform(rotation=np.pi / 2)
            # rotate = transform.SimilarityTransform(rotation=0)
            # rc,cc=rotate(region.centroid)[0]

            rc, cc = region.centroid
            l = region.major_axis_length / 2
            # l_sint, l_cost = np.sin(region.orientation) * l, np.cos(region.orientation) * l

            if region.orientation > 0:
                out[rp, cp, :] = (0, 1, 0)
                l_sint, l_cost = np.sin(region.orientation) * l, np.cos(region.orientation) * l
                xx1, yy1 = cc + l_sint, rc + l_cost  # don't know why, but i had to interchange sin and cos
                xx2, yy2 = cc - l_sint, rc - l_cost
            elif region.orientation == 0:
                # log.warning("orientation was zero!")
                out[rp, cp, :] = (1, 1, 1)
                xx1, yy1 = cc + l, rc
                xx2, yy2 = cc - l, rc
            else:
                out[rp, cp, :] = (1, 0, 1)
                l_sint, l_cost = np.sin(np.pi - region.orientation) * l, np.cos(np.pi - region.orientation) * l
                xx1, yy1 = cc + l_sint, rc - l_cost  # don't know why, but i had to interchange sin and cos
                xx2, yy2 = cc - l_sint, rc + l_cost
            ax3.plot(xx1, yy1, marker='o', markersize=2, c='red')
            ax3.plot(xx2, yy2, marker='o', markersize=2, c='blue')
        ax3.set_xlim([0, w])
        ax3.set_ylim([h, 0])

        ax4.imshow(out)

        return mplfig_to_npimage(fig)  # RGB image of the figure

    _images, pix_per_um, dt = sp.load_tiff(image_path)
    # subtract first frame and deal with negative results after the operation
    frames = np.int32(_images)
    frames -= frames[0]
    frames = frames[1:, :, :]
    frames = np.uint16(frames.clip(0))

    logging.info("rendering movie %s" % filename)
    fig = plt.figure(20, figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    fig.tight_layout()

    animation = mpy.VideoClip(make_frame_mpl, duration=len(frames))
    animation.write_videofile(filename, fps=1)
    animation.close()


def detection(images, pix_per_um=1):
    features = pd.DataFrame()

    # subtract first frame and deal with negative results after the operation
    nobkg = np.int32(images)
    nobkg -= nobkg[0]
    nobkg = nobkg[1:, :, :]
    nobkg = np.uint16(nobkg.clip(0))

    for num, im in enumerate(nobkg):
        thr_lvl = filters.threshold_yen(im)
        thresh = im >= thr_lvl
        morphology.remove_small_objects(thresh, min_size=1 * pix_per_um, in_place=True)

        labels = ndimage.label(thresh)[0]
        # w, h = im.shape

        for region in measure.regionprops(labels, coordinates='rc', cache=True):
            # if region.eccentricity < 0.8: continue

            rc, cc = region.centroid
            l = region.major_axis_length / 2
            if region.orientation > 0:
                l_sint, l_cost = np.sin(region.orientation) * l, np.cos(region.orientation) * l
                xx1, yy1 = cc + l_sint, rc + l_cost  # don't know why, but i had to interchange sin and cos
                xx2, yy2 = cc - l_sint, rc - l_cost
            elif region.orientation == 0:
                # log.warning("orientation was zero!")
                xx1, yy1 = cc + l, rc
                xx2, yy2 = cc - l, rc
            else:
                l_sint, l_cost = np.sin(np.pi - region.orientation) * l, np.cos(np.pi - region.orientation) * l
                xx1, yy1 = cc + l_sint, rc - l_cost  # don't know why, but i had to interchange sin and cos
                xx2, yy2 = cc - l_sint, rc + l_cost

            features = features.append([{'x': cc / pix_per_um, 'y': rc / pix_per_um,
                                         'pt1': Point((xx1 / pix_per_um, yy1 / pix_per_um)),
                                         'pt2': Point((xx2 / pix_per_um, yy2 / pix_per_um)),
                                         'l': LineString([(xx1 / pix_per_um, yy1 / pix_per_um),
                                                          (xx2 / pix_per_um, yy2 / pix_per_um)]),
                                         'x1': xx1 / pix_per_um, 'y1': yy1 / pix_per_um,
                                         'x2': xx2 / pix_per_um, 'y2': yy2 / pix_per_um,
                                         'theta': np.pi / 2 - region.orientation, 'frame': num}])

    return features.reset_index(drop=True)


def do_tracking(image_path, asters=None):
    images, pix_per_um, dt = sp.load_tiff(image_path)
    w, h = images[0].shape[0], images[0].shape[1]

    log.info("detecting points.")
    f = detection(images, pix_per_um=pix_per_um)
    # f.to_pickle('f.pandas')
    # f = pd.read_pickle('f.pandas')
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

    fig = plt.figure(dpi=150)
    ax = fig.gca()
    ax.set_facecolor(sp.colors.sussex_cobalt_blue)
    ext = [0, w / pix_per_um, h / pix_per_um, 0]
    ax.imshow(np.max(images, axis=0), interpolation='none', extent=ext, cmap=cm.gray)
    # ax.imshow(obj, interpolation='none', extent=ext, alpha=0.3)
    # for ix, row in f.iterrows():
    #     ax.plot([row['x1'], row['x2']], [row['y1'], row['y2']], lw=0.1, c='white', alpha=0.1)
    # obj_function(f, 60, 65, ax=ax)

    filtered = pd.DataFrame()
    for xb, yb in wheel.best:
        wheel.plot(xb, yb, ax=ax)
        filtered = filtered.append(wheel.filter_wheel(xb, yb))

    ax.set_aspect('equal')
    ax.set_xlim([0, w / pix_per_um])
    ax.set_ylim([0, h / pix_per_um])
    fig.savefig('%s_objfn.png' % image_path[:-4])

    if len(filtered) == 0:
        raise Exception('no data after filter.')

    search_range = 0.2
    pred = tp.predict.NearestVelocityPredict(initial_guess_vels=0.1)
    # trackpy.linking.Linker.MAX_SUB_NET_SIZE=50
    linked = pred.link_df(filtered, search_range)

    # #  filter spurious tracks
    # frames_per_particle = linked.groupby('particle')['frame'].nunique()
    # particles = frames_per_particle[frames_per_particle > 8].index
    # linked = linked[linked['particle'].isin(particles)]
    # logging.info('filtered %d particles by track length' % linked['particle'].nunique())
    #
    # m = tp.imsd(linked, 1, 1, pos_columns=['xum', 'yum'])
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

                    df.to_csv(csv_file, index=False)
                    df = df.append(tdf)
                except IOError as ioe:
                    logging.warning('could not import due to IO error: %s' % ioe)
                except Exception as e:
                    logging.error('skipped file %s' % f)
                    logging.error(e)

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
    # df = pd.read_pickle(p.experiments_dir + 'eb3.pandas')

    # process dataframe and render images
    from microtubules.eb3 import run_plots_eb3

    logging.info('filtering using run_plots_eb3.')
    df, df_avg = run_plots_eb3.batch_filter(df)
    logging.info('rendering images.')
    run_plots_eb3.render_image_tracks(df, folder=p.experiments_dir + 'eb3')

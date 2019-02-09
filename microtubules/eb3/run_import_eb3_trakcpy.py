import logging
import os
import re
import sys
import warnings

import coloredlogs
import pandas as pd
import trackpy as tp
import trackpy.diag
import trackpy.predict
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
import skimage.exposure as exposure
import skimage.color as color

import parameters as p
import plot_special_tools as sp

warnings.filterwarnings("ignore", category=DeprecationWarning)
coloredlogs.install(fmt='%(levelname)s:%(funcName)s - %(message)s', level=logging.INFO)
tp.diag.performance_report()
logging.basicConfig(stream=sys.stderr, format='%(levelname)s:%(funcName)s - %(message)s', level=logging.INFO)
coloredlogs.install()
trackpy.quiet()
pd.set_option('display.width', 320)


def movie(frames, linked_particles, filename="movie.mp4", pix_per_um=1):
    def make_frame_mpl(t):
        fr = int(t)

        ax.cla()
        im = frames[fr]
        # im = exposure.equalize_hist(im)
        # w, h = im.shape[0], im.shape[1]

        tp.annotate(linked_particles[linked_particles['frame'] == fr], im, ax=ax,
                    plot_style={"markersize": 4},
                    # imshow_style={"extent": [0, w / pix_per_um, h / pix_per_um, 0]})
                    )
        # tp.plot_displacements(linked_particles, fr, fr + 1, ax=ax, arrowprops={"color": "blue", "alpha": 0.2})
        # ax.set_xlabel("x[um]")
        # ax.set_ylabel("y[um]")

        return mplfig_to_npimage(fig)  # RGB image of the figure

    logging.info("rendering movie %s" % filename)
    fig = plt.figure(20)
    ax = fig.gca()
    fig.tight_layout()

    animation = mpy.VideoClip(make_frame_mpl, duration=len(frames))
    animation.write_videofile(filename, fps=1)
    animation.close()


def do_trackpy(image_path):
    _images, pix_per_um, dt = sp.load_tiff(image_path)

    # subtract first frame and deal with negative results after the operation
    frames = np.int32(_images)
    frames -= frames[0]
    frames = frames[1:, :, :]
    frames = np.uint16(frames.clip(0))

    diam = np.ceil(pix_per_um) // 2 * 4 + 1

    f = tp.batch(frames[1:], diameter=diam, separation=diam / 6, characterize=True)
    stat = f[['mass', 'size']].describe()
    # f = f[((f['mass'] > stat['mass']['25%']) & (f['size'] < 1.5))]
    f = f[(f['mass'] > stat['mass']['25%'])]
    # logging.info(stat)

    f['xum'] = f['x'] / pix_per_um
    f['yum'] = f['y'] / pix_per_um
    # for search_range in [1.0, 1.5, 2.0, 2.5]:
    #     linked = tp.link_df(f, search_range, pos_columns=['xum', 'yum'])
    #     hist, bins = np.histogram(np.bincount(linked.particle.astype(int)),
    #                               bins=np.arange(30), normed=True)
    #     plt.step(bins[1:], hist, label='range = {} microns'.format(search_range))
    # plt.gca().set(ylabel='relative frequency', xlabel='track length (frames)')
    # plt.legend()
    # plt.show()
    # exit()

    search_range = 1.0
    pred = tp.predict.NearestVelocityPredict(initial_guess_vels=0.4)
    linked = pred.link_df(f, search_range, pos_columns=['xum', 'yum'])
    #  filter spurious tracks
    # linked = tp.filter_stubs(linked, 4)
    frames_per_particle = linked.groupby('particle')['frame'].nunique()
    particles = frames_per_particle[frames_per_particle > 15].index
    linked = linked[linked['particle'].isin(particles)]
    logging.info('filtered %d particles by track length' % linked['particle'].nunique())

    m = tp.imsd(linked, 1, 1, pos_columns=['xum', 'yum'])
    mt = m.ix[15]
    particles = mt[mt > 2].index
    linked = linked[linked['particle'].isin(particles)]
    # plt.hist(m.ix[15],bins=100)
    # plt.savefig('%s_msdhist.png' % image_path[:-4])
    # linked = linked.set_index('particle').ix[tp.is_typical(m, 10, lower=0.1)].reset_index()
    logging.info('filtered %d particles msd' % linked['particle'].nunique())

    # path, fim = os.path.split(image_path)
    linked['frame'] += 1
    frames = exposure.equalize_hist(frames)
    _images = exposure.equalize_hist(_images)
    frames = color.gray2rgb(frames)
    _images = color.gray2rgb(_images)
    render = _images[1:] * 0.2 + frames * sp.colors.hoechst_33342 * 0.4
    movie(render, linked, filename='%s.mp4' % image_path[:-4], pix_per_um=pix_per_um)
    # exit()

    return linked


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
                try:  # process
                    tdf = do_trackpy(mpath)
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
                        tdf['xum'] /= 1.6
                        tdf['yum'] /= 1.6

                    df.to_csv(os.path.join(root, '%s.csv' % f[:-4]), index=False)
                    df = df.append(tdf)
                except IOError as ioe:
                    logging.warning('could not import due to IO error: %s' % ioe)
                except Exception as e:
                    logging.error('skipped file %s' % f)
                    logging.error(e)

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
    df = process_dir(p.experiments_dir + 'eb3')
    df.to_pickle(p.experiments_dir + 'eb3.pandas')
    # df = pd.read_pickle(p.experiments_dir + 'eb3.pandas')

    # process dataframe and render images
    from microtubules.eb3 import run_plots_eb3

    logging.info('filtering using run_plots_eb3.')
    df, df_avg = run_plots_eb3.batch_filter(df)
    logging.info('rendering images.')
    run_plots_eb3.render_image_tracks(df, folder=p.experiments_dir + 'eb3')

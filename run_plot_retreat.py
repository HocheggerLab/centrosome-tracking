import logging

import matplotlib.colors
import matplotlib.gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

import plot_special_tools as sp
import tools.data as data

log = logging.getLogger(__name__)
log.info(font_manager.OSXInstalledFonts())
log.info(font_manager.OSXFontDirectories)

plt.style.use('bmh')
# plt.style.use('ggplot')
# sns.set(context='paper', style='whitegrid', font='Helvetica Neue')
# matplotlib.rc('pdf', fonttype=42)
# matplotlib.rc('svg', fonttype='none')
print(matplotlib.rcParams.keys())
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


def images_iterator(channel=1):
    import tifffile as tf
    path = '/Users/Fabio/data/lab/pc/input/pc-114.tif'
    with tf.TiffFile(path) as tif:
        frames, channels = tif.imagej_metadata['frames'], tif.imagej_metadata['channels']
        sizeZ, sizeX, sizeY = 1, tif.pages[0].imagewidth, tif.pages[0].imagelength

        dt = None
        if tif.is_imagej is not None:
            dt = tif.imagej_metadata['finterval']
            res = 'n/a'
            if tif.imagej_metadata['unit'] == 'centimeter':
                # asuming square pixels
                xr = tif.pages[0].x_resolution
                res = float(xr[0]) / float(xr[1])  # pixels per cm
                res = res / 1e4  # pixels per um
            elif tif.imagej_metadata['unit'] == 'micron':
                # asuming square pixels
                xr = tif.pages[0].tags['XResolution'].value
                res = float(xr[0]) / float(xr[1])  # pixels per um

        for i in range(frames):
            # p = tif.pages[i * channels: (i + 1) * channels - 1]
            p = tif.pages[i * channels + channel - 1]
            im = p.asarray()
            yield i, im, res


def selected_track(data):
    from moviepy.video.io.bindings import mplfig_to_npimage

    df = data.df_m
    df_selected = df[(df['condition'] == 'pc') & (df['run'] == 'run_114') & (df['Nuclei'] == 2)]

    plt.style.use('dark_background')
    matplotlib.rcParams.update({'axes.titlesize': 10})
    matplotlib.rcParams.update({'axes.labelsize': 10})
    matplotlib.rcParams.update({'xtick.labelsize': 10})
    matplotlib.rcParams.update({'ytick.labelsize': 10})
    matplotlib.rcParams.update({'legend.fontsize': 8})

    fig = plt.figure(figsize=(15, 5), dpi=300)
    gs = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1:2])
    ax22 = ax2.twinx()
    ax1.set_aspect('equal')
    ax2.set_xlim([0, 100])
    ax22.set_ylim([0, 15])

    plt.subplots_adjust(left=0., bottom=0.15, right=0.95, top=0.95, wspace=0.0, hspace=0.0)
    sp.set_axis_size(5, 5, ax1)
    sp.set_axis_size(9, 3, ax2)
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax2.set_xlabel('time [min]')
    ax22.set_ylabel('dist [um]')
    # ax2.yaxis.set_label_position('right')
    # ax21.spines["right"].set_position(("axes", 0.4))  # red one
    # ax22.spines["right"].set_position(("axes", 0.2))  # green one

    flist = df_selected['Frame'].unique()
    it = images_iterator(channel=2)

    def make_frame_mpl(t):
        f, im, res = it.__next__()
        p = df_selected[df_selected['Frame'] <= f]
        pa = p[p['CentrLabel'] == 'A']
        pb = p[p['CentrLabel'] == 'B']

        ax1.cla()
        d = df_selected[df_selected['Frame'] == f]
        sp.render_cell(d, ax1, img=im, res=res, w=60, h=60)
        ax1.grid(False)
        ax1.set_xlabel('[um]')
        ax1.set_ylabel('[um]')

        if f < 8:
            # ax22.cla()
            ax22.plot(pa["Time"], pa["DistCentr"], c=sp.SUSSEX_WARM_GREY, marker='o')
            ax22.plot(pa["Time"], pa["Dist"], c=sp.SUSSEX_CORAL_RED, marker='o')
            ax22.plot(pb["Time"], pb["Dist"], c=sp.SUSSEX_NAVY_BLUE, marker='o')
        elif f >= 8:
            pa_ = pa[pa["Frame"] < 8]
            pb_ = pb[pb["Frame"] < 8]
            ax22.plot(pa_["Time"], pa_["DistCentr"], c=sp.SUSSEX_WARM_GREY, marker='o')
            ax22.plot(pa_["Time"], pa_["Dist"], c=sp.SUSSEX_CORAL_RED, marker='o')
            ax22.plot(pb_["Time"], pb_["Dist"], c=sp.SUSSEX_NAVY_BLUE, marker='o')

            pa_ = pa[pa["Frame"] == 8]
            pb_ = pb[pb["Frame"] == 8]
            ax22.scatter(pa_["Time"], pa_["DistCentr"], edgecolors=sp.SUSSEX_WARM_GREY, facecolors='none', lw=1)
            ax22.scatter(pa_["Time"], pa_["Dist"], edgecolors=sp.SUSSEX_CORAL_RED, facecolors='none', lw=1)
            ax22.scatter(pb_["Time"], pb_["Dist"], edgecolors=sp.SUSSEX_NAVY_BLUE, facecolors='none', lw=1)

            pa_ = pa[(pa["Frame"] > 8) & (pa["Frame"] <= f)]
            pb_ = pb[(pb["Frame"] > 8) & (pb["Frame"] <= f)]
            ax22.plot(pa_["Time"], pa_["DistCentr"], c=sp.SUSSEX_WARM_GREY, marker='o')
            ax22.plot(pa_["Time"], pa_["Dist"], c=sp.SUSSEX_CORAL_RED, marker='o')
            ax22.plot(pb_["Time"], pb_["Dist"], c=sp.SUSSEX_NAVY_BLUE, marker='o')

        fig.savefig('frame-%d.png' % f)
        return mplfig_to_npimage(fig)  # RGB image of the figure

    # animation = mpy.VideoClip(make_frame_mpl, duration=flist.size / 2 - 1)
    # animation.write_videofile("my_new_video.mp4", fps=2)
    # animation.close()
    # clip = mpy.ImageSequenceClip(['frame-%d.png' % f for f in flist], fps=2)
    # clip.write_videofile("my_new_video2.mp4", audio=False, verbose=True )
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 'x264' doesn't work
    video = cv2.VideoWriter('movie.avi', fourcc, 1, (2842, 1125))
    for f in range(14):
        make_frame_mpl(f)
        video.write(cv2.imread('frame-%d.png' % f))

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    log.info('loading data')
    data = data.Data()

    selected_track(data)

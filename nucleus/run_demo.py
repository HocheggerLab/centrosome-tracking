import os
import logging

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import seaborn as sns
import numpy as np
from pysketcher import drawing_tool

from nucleus._track import Track
from nucleus._common import _DEBUG
import nucleus.plots as plots
from plot_special_tools import canvas_to_pil
import plot_special_tools as sp

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def ensure_dir(file_path):
    file_path = os.path.abspath(file_path)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return file_path


def angle_plots(df):
    fig = Figure((width * 4 / 150, width * 4 / 150), dpi=150)
    canvas_g = FigureCanvas(fig)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    rot = plots.Rotation(df)
    # rot.angle_over_time(ax1)
    rot.angle_corrected_over_time(ax1)
    rot.angle_displacement_over_time(ax2)
    # rot.angular_speed_over_time(ax2)

    pil = canvas_to_pil(canvas_g)
    fpath = os.path.abspath(os.path.join('_grph', 'angle.png'))
    pil.save(ensure_dir(fpath))


def radius_plots(df):
    fig = plt.figure(figsize=(width * 4 / 150, width * 4 / 150), dpi=150)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    rot = plots.Rotation(df)
    rot.radius_over_time(ax1)
    rot.radius_change_over_time(ax2)

    fig.savefig(os.path.join('_grph', 'radius.png'))


def msd_plots(df):
    fig = plt.figure(figsize=(width * 4 / 150, width * 4 / 150), dpi=150)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    rot = plots.Rotation(df)
    rot.msd_over_time(ax1)
    rot.msd_change_over_time(ax2)

    fig.savefig(os.path.join('_grph', 'msd.png'))


if __name__ == '__main__':
    """
    after creating images, it is possible to compile them in a movie using:
    ffmpeg -r 1 -i frame_%02d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p out.mp4
    """
    if _DEBUG: logger.info("debug flag on")
    logger.info('loading data')
    file = '/Volumes/Kidbeat/data/Dynein etc/DYNH1 A19 X1 29-01-15/Capture 1 - Position 3.Project Maximum.tif'
    t = Track(image_file=file, nucleus_channel=2, skip_frames=5)
    print(t.nucleus_rotation)
    print(sorted(t.nucleus_rotation["frame"].unique()))
    width, height = t.images[0].shape

    # filter one set
    df = t.nucleus_rotation
    df = df[df['nucleus'] == 0]
    filtered = df[~df['particle'].isin(df[np.abs(df['th_dev']) > 1.5]['particle'].unique())]

    sns.set_palette([sp.SUSSEX_COBALT_BLUE, sp.SUSSEX_CORAL_RED])
    angle_plots(filtered)
    radius_plots(t.nucleus_rotation)
    msd_plots(t.nucleus_rotation)

    # fig = plt.figure(figsize=(width * 4 / 150, width * 4 / 150), dpi=150)
    # ax = fig.gca()
    drawing_tool.set_coordinate_system(xmin=0, xmax=width / t.pix_per_um, ymin=0, ymax=height / t.pix_per_um)
    ax = drawing_tool.ax
    sp.set_axis_size(6, 6)
    for f in t.nucleus_rotation["frame"].unique():
        ax.cla()
        t.render(ax, frame=f)
        ax.text(5, height / t.pix_per_um - 10, '%02d' % f, color='white', fontsize=20)

        plt.draw()
        drawing_tool.display()
        drawing_tool.savefig(ensure_dir(os.path.join('_mov', 'frame_%02d.png' % f)))

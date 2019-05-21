import os
import logging

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from nucleus._track import Track
from nucleus._common import _DEBUG
from nucleus.plots import Rotation
from plot_special_tools import canvas_to_pil

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def ensure_dir(file_path):
    file_path = os.path.abspath(file_path)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return file_path


if __name__ == '__main__':
    if _DEBUG: logger.info("debug flag on")
    logger.info('loading data')
    file = '/Volumes/Kidbeat/data/Dynein etc/DYNH1 A19 X1 29-01-15/Capture 1 - Position 3.Project Maximum.tif'
    t = Track(image_file=file, nucleus_channel=2)
    print(t.nucleus_rotation)
    print(sorted(t.nucleus_rotation["frame"].unique()))

    width, height = t.images[0].shape
    fig = Figure((width * 4 / 150, width * 4 / 150), dpi=150)
    canvas_g = FigureCanvas(fig)
    ax = fig.gca()
    for f in t.nucleus_rotation["frame"].unique():
        ax.cla()
        t.render(ax, frame=f)

        pil = canvas_to_pil(canvas_g)
        fpath = os.path.abspath(os.path.join('_mov', 'frame_%d.jpg' % f))
        pil.save(ensure_dir(fpath))

    fig = Figure((width * 4 / 150, width * 4 / 150), dpi=150)
    canvas_g = FigureCanvas(fig)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    r = t.nucleus_rotation
    # filter one set
    r = r[r['nucleus'] == 0]
    r = r[~r['particle'].isin(r[np.abs(r['th_dev']) > 1.5]['particle'].unique())]

    rot = Rotation(r)
    # rot.angle_over_time(ax1)
    rot.angle_corrected_over_time(ax1)
    rot.angle_displacement_over_time(ax2)
    # rot.angular_speed_over_time(ax2)

    pil = canvas_to_pil(canvas_g)
    fpath = os.path.abspath(os.path.join('_grph', 'angle.png'))
    pil.save(ensure_dir(fpath))

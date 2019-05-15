import os
import logging

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from nucleus._track import Track
from nucleus._common import _DEBUG
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
    w_um, h_um = [s / t.pix_per_um for s in t.images[0].shape]
    fig = Figure((width * 4 / 150, width * 4 / 150), dpi=150)
    canvas_g = FigureCanvas(fig)
    ax = fig.gca()

    for f in t.nucleus_rotation["frame"].unique():
        ax.cla()
        t.render(ax, frame=f)

        pil = canvas_to_pil(canvas_g)
        fpath = os.path.abspath(os.path.join('_mov', 'frame_%d.jpg' % f))
        pil.save(ensure_dir(fpath))

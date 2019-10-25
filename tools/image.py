import logging
import os
import xml.etree
import xml.etree.ElementTree

import numpy as np
import pandas as pd
from czifile import CziFile
from PIL import Image
from skimage.external import tifffile as tf

import parameters
import tools.measurements as meas

logger = logging.getLogger(__name__)


def load_tiff(path):
    _, img_name = os.path.split(path)
    with tf.TiffFile(path) as tif:
        if tif.is_imagej is not None:
            metadata = tif.pages[0].imagej_tags
            dt = metadata['finterval'] if 'finterval' in metadata else 1

            # asuming square pixels
            xr = tif.pages[0].tags['x_resolution'].value
            res = float(xr[0]) / float(xr[1])  # pixels per um
            if metadata['unit'] == 'centimeter':
                res = res / 1e4

            # This is a hack
            # Process pixel calibration from excel file if given
            if os.path.exists(parameters.out_dir + 'eb3/eb3_calibration.xls'):
                cal = pd.read_excel(parameters.out_dir + 'eb3/eb3_calibration.xls')
                calp = cal[cal['filename'] == img_name]
                if not calp.empty:
                    calp = calp.iloc[0]
                    if calp['optivar'] == 'yes':
                        logging.info('file with optivar configuration selected!')
                        res *= 1.6

            images = None
            if len(tif.pages) == 1:
                if ('slices' in metadata and metadata['slices'] > 1) or (
                        'frames' in metadata and metadata['frames'] > 1):
                    images = tif.pages[0].asarray()
                else:
                    images = [tif.pages[0].asarray()]
            elif len(tif.pages) > 1:
                images = list()
                for i, page in enumerate(tif.pages):
                    images.append(page.asarray())

            return np.asarray(images), res, dt, \
                   metadata['frames'] if 'frames' in metadata else 1, \
                   metadata['channels'] if 'channels' in metadata else 1, \
                   tif.series


def load_zeiss(path):
    _, img_name = os.path.split(path)
    with CziFile(path) as czi:
        xmltxt = czi.metadata()
        meta = xml.etree.ElementTree.fromstring(xmltxt)

        # next line is somewhat cryptic, but just extracts um/pix (calibration) of X and Y into res
        res = [float(i[0].text) for i in meta.findall('.//Scaling/Items/*') if
               i.attrib['Id'] == 'X' or i.attrib['Id'] == 'Y']
        assert res[0] == res[1], "pixels are not square"

        # get first calibration value and convert it from meters to um
        res = res[0] * 1e6

        ts_ix = [k for k, a1 in enumerate(czi.attachment_directory) if a1.filename[:10] == 'TimeStamps'][0]
        timestamps = list(czi.attachments())[ts_ix].data()
        dt = np.median(np.diff(timestamps))

        ax_dct = {n: k for k, n in enumerate(czi.axes)}
        n_frames = czi.shape[ax_dct['T']]
        n_channels = czi.shape[ax_dct['C']]
        n_X = czi.shape[ax_dct['X']]
        n_Y = czi.shape[ax_dct['Y']]

        images = list()
        for sb in czi.subblock_directory:
            images.append(sb.data_segment().data().reshape((n_X, n_Y)))

        return np.array(images), 1 / res, dt, n_frames, n_channels, None


def find_image(img_name, folder=None):
    if folder is None:
        folder = os.path.dirname(img_name)
        img_name = os.path.basename(img_name)

    for root, directories, filenames in os.walk(folder):
        for file in filenames:
            joinf = os.path.abspath(os.path.join(root, file))
            if os.path.isfile(joinf) and joinf[-4:] == '.tif' and file == img_name:
                return load_tiff(joinf)
            if os.path.isfile(joinf) and joinf[-4:] == '.czi' and file == img_name:
                return load_zeiss(joinf)


def retrieve_image(image_arr, frame, channel=0, number_of_frames=1):
    nimgs = image_arr.shape[0]
    n_channels = int(nimgs / number_of_frames)
    ix = frame * n_channels + channel
    logger.debug("retrieving frame %d of channel %d (index=%d)" % (frame, channel, ix))
    return image_arr[ix]


def image_iterator(image_arr, channel=0, number_of_frames=1):
    nimgs = image_arr.shape[0]
    n_channels = int(nimgs / number_of_frames)
    for f in range(number_of_frames):
        ix = f * n_channels + channel
        logger.debug("retrieving frame %d of channel %d (index=%d)" % (f, channel, ix))
        if ix < nimgs: yield image_arr[ix]


def mask_iterator(image_it, mask_lst):
    for fr, img in enumerate(image_it):
        for _fr, msk in mask_lst:
            if fr != _fr: continue
            msk_img = meas.generate_mask_from(msk, shape=img.shape)
            yield img * msk_img


def pil_grid(images, max_horiz=np.iinfo(int).max):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid


def canvas_to_pil(canvas):
    canvas.draw()
    s = canvas.tostring_rgb()
    w, h = canvas.get_width_height()[::-1]
    im = Image.frombytes("RGB", (w, h), s)
    return im

import logging

import numpy as np
import skimage.draw as draw
from shapely.geometry.polygon import Polygon

log = logging.getLogger(__name__)


def integral_over_surface(image, polygon):
    c, r = polygon.boundary.xy
    rr, cc = draw.polygon(r, c)

    try:
        ss = np.sum(image[rr, cc])
        return ss
    except Exception:
        log.warning('integral_over_surface measured incorrectly')
        return np.nan


def generate_mask_from(polygon: Polygon, shape=None):
    if shape is None:
        minx, miny, maxx, maxy = polygon.bounds
        image = np.zeros((maxx - minx, maxy - miny), dtype=np.bool)
    else:
        image = np.zeros(shape, dtype=np.bool)

    c, r = polygon.boundary.xy
    rr, cc = draw.polygon(r, c)
    image[rr, cc] = True
    return image

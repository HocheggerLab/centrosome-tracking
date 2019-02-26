import logging

import numpy as np
import skimage.draw as draw

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

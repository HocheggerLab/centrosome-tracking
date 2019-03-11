import logging

import pandas as pd
import numpy as np
import skimage.filters as filters
import skimage.morphology as morphology
from scipy import ndimage
from shapely.geometry import LineString, Point
import cv2
from matplotlib import cm

import plot_special_tools as sp

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Particles():
    def __init__(self, image_file):
        self.images, self.pix_per_um, self.dt = sp.load_tiff(image_file)
        self.w = self.images[0].shape[0]
        self.h = self.images[0].shape[1]
        self._df = None
        self.segmented = None

        self._remove_backgound()

    def _remove_backgound(self):
        # subtract first frame and deal with negative results after the operation
        self.nobkg = np.int32(self.images)
        self.nobkg -= self.nobkg[0]
        self.nobkg = self.nobkg[1:, :, :]
        self.nobkg = np.uint16(self.nobkg.clip(0))

    def _segmentation_step(self):
        thr_lvl = filters.threshold_otsu(self.nobkg)
        self.segmented = (self.nobkg >= thr_lvl).astype(bool)
        morphology.remove_small_objects(self.segmented, min_size=4 * self.pix_per_um, connectivity=1, in_place=True)

    def _debug_fitted_ellipse_plot(self, frame, contour):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        ellipse = cv2.fitEllipse(contour)
        (x0, y0), (we, he), angle_deg = ellipse  # angle_rad goes from 0 to 180
        x0, y0, we, he = np.array([x0, y0, we, he]) / self.pix_per_um
        angle_rad = np.deg2rad(angle_deg - 90)  # angle_rad goes from -pi/2 to pi/2

        fig = plt.figure(dpi=100)
        ax = fig.gca()
        ext = [0, self.w / self.pix_per_um, self.h / self.pix_per_um, 0]
        ax.imshow(self.images[frame], interpolation='none', extent=ext)
        ax.imshow(self.segmented[frame], interpolation='none', extent=ext, alpha=0.3)

        ell = Ellipse(xy=(x0, y0), angle=angle_deg,
                      width=we, height=he,
                      facecolor='gray', alpha=0.5)
        ax.add_artist(ell)

        x1 = x0 + np.cos(angle_rad) * 0.5 * he
        y1 = y0 + np.sin(angle_rad) * 0.5 * he
        x2 = x0 + np.sin(angle_rad) * 0.5 * we
        y2 = y0 - np.cos(angle_rad) * 0.5 * we
        ax.plot((x0, x1), (y0, y1), '-y', linewidth=1, zorder=15)  # major axis
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=1, zorder=15)  # minor axis
        ax.plot(x0, y0, '.g', markersize=10, zorder=20)

        l = he / 2
        l_sint, l_cost = np.sin(angle_rad) * l, np.cos(angle_rad) * l
        xx1, yy1 = x0 + l_cost, y0 + l_sint
        xx2, yy2 = x0 - l_cost, y0 - l_sint
        ax.plot((x0, xx1), (y0, yy1), color='magenta', linewidth=2, zorder=10)  # minor axis
        ax.plot((x0, xx2), (y0, yy2), color='blue', linewidth=2, zorder=10)  # major axis

        x, y, wbr, hbr = np.array(cv2.boundingRect(contour)) / self.pix_per_um
        bx = (x, x + wbr, x + wbr, x, x)
        by = (y, y, y + hbr, y + hbr, y)
        ax.plot(bx, by, '-b', linewidth=1)

        ax.text(x + wbr, y + hbr, "orig: %0.2f" % angle_deg, color="white")
        ax.text(x + wbr, y, "tran: %0.2f" % angle_rad, color="white")

        ax.set_aspect('equal')
        ax.set_xlim([x - wbr, x + 2 * wbr])
        ax.set_ylim([y - hbr, y + 2 * hbr])

        plt.show()

    @property
    def df(self):
        if self._df is not None: return self._df

        self._segmentation_step()

        features = list()
        blackboard = np.zeros(shape=self.images[0].shape, dtype=np.uint8)
        for num, im in enumerate(self.segmented):
            labels = ndimage.label(im)[0]

            for (i, l) in enumerate([l for l in np.unique(labels) if l > 0]):
                # find contour of mask
                blackboard[labels == l] = 255
                cnt = cv2.findContours(blackboard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2][0]
                blackboard[labels == l] = 0
                if cnt.shape[0] < 5: continue
                # if region.eccentricity < 0.8: continue
                ellipse = cv2.fitEllipse(cnt)
                (x0, y0), (we, he), angle_deg = ellipse  # angle_rad goes from 0 to 180
                x0, y0, we, he = np.array([x0, y0, we, he]) / self.pix_per_um
                angle_rad = np.deg2rad(angle_deg - 90)  # angle_rad goes from -pi/2 to pi/2

                # self._debug_fitted_ellipse_plot(num, cnt)

                l = he / 2
                l_sint, l_cost = np.sin(angle_rad) * l, np.cos(angle_rad) * l
                xx1, yy1 = x0 + l_cost, y0 + l_sint
                xx2, yy2 = x0 - l_cost, y0 - l_sint

                features.append({'x': x0, 'y': y0,
                                 'pt1': Point((xx1, yy1)), 'pt2': Point((xx2, yy2)),
                                 'l': LineString([(xx1, yy1), (xx2, yy2)]),
                                 'x1': xx1, 'y1': yy1, 'x2': xx2, 'y2': yy2,
                                 'theta': angle_rad, 'frame': num})
        self._df = pd.DataFrame(features)
        return self._df

    def render_time_projection(self, ax):
        ext = [0, self.w / self.pix_per_um, self.h / self.pix_per_um, 0]
        ax.imshow(np.max(self.images, axis=0), interpolation='none', extent=ext, cmap=cm.gray)
        ax.set_aspect('equal')
        ax.set_xlim([0, self.w / self.pix_per_um])
        ax.set_ylim([0, self.h / self.pix_per_um])

    def render_detected_features(self, ax):
        ax.scatter(self.df['x'], self.df['y'], s=1, marker='+', c='white')
        # ax.scatter(lp.loc[lp['frame'] == fr, 'x1'], lp.loc[lp['frame'] == fr, 'y1'], s=1, c='magenta', zorder=10)
        # ax.scatter(lp.loc[lp['frame'] == fr, 'x2'], lp.loc[lp['frame'] == fr, 'y2'], s=1, c='blue', zorder=10)
        # ax.plot([lp.loc[lp['frame'] == fr, 'x1'], lp.loc[lp['frame'] == fr, 'x2']],
        #         [lp.loc[lp['frame'] == fr, 'y1'], lp.loc[lp['frame'] == fr, 'y2']], lw=0.5, alpha=0.3, c='white')

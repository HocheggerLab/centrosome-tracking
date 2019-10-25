import logging
import os

import pandas as pd
import numpy as np
import skimage.filters as filters
import skimage.morphology as morphology
from scipy import ndimage
from shapely.geometry import LineString, Point
import cv2
from matplotlib import cm
import trackpy as tp
import skimage.color as skcolor

import tools.image as image
import parameters as p
import tools.plot_tools as sp

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Particles():
    def __init__(self, image_file):
        log.info("Initializing particles object")
        self.images, self.pix_per_um, self.dt, self.n_frames, self.n_channels = image.find_image(image_file)
        self.um_per_pix = 1 / self.pix_per_um
        self.im_f = os.path.basename(image_file)
        self.im_p = os.path.dirname(image_file)
        self.w = self.images[0].shape[0]
        self.h = self.images[0].shape[1]
        self._df = None
        self._linked = None
        self.wheel = None
        self.segmented = None

        self._get_cal_excel()
        self._remove_backgound()

    def _get_cal_excel(self):
        # get calibration parameters from excel file
        cal = pd.read_excel(p.experiments_dir + 'eb3/eb3_calibration.xls')

        if not (cal['filename'] == self.im_f).any(): return
        calp = cal[cal['filename'] == self.im_f].iloc[0]
        self.dt = calp['dt']

        # consider 1.6X magification of optivar system
        if calp['optivar'] == 'yes':
            # self.pix_per_um *= 1.6
            self.pix_per_um = 4.5 * 1.6

    def _remove_backgound(self):
        # subtract first frame and deal with negative results after the operation
        self.nobkg = np.int32(self.images)
        self.nobkg -= self.nobkg[0]
        self.nobkg = self.nobkg[1:, :, :]
        self.nobkg = np.uint16(self.nobkg.clip(0))

    def _segmentation_step(self):
        log.info("Segmenting images")
        thr_lvl = filters.threshold_otsu(self.nobkg)
        self.segmented = (self.nobkg >= thr_lvl).astype(bool)
        morphology.remove_small_objects(self.segmented, min_size=4 * self.pix_per_um, connectivity=1, in_place=True)
        for i, im in enumerate(self.segmented):
            self.segmented[i, :] = morphology.opening(im, selem=morphology.square(3))

    def _debug_fitted_ellipse_plot(self, frame, contour):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        ellipse = cv2.fitEllipse(contour)
        (x0, y0), (we, he), angle_deg = ellipse  # angle_rad goes from 0 to 180
        x0, y0, we, he = np.array([x0, y0, we, he]) / self.pix_per_um
        angle_rad = np.deg2rad(angle_deg - 90)  # angle_rad goes from -pi/2 to pi/2

        # fig = plt.figure(dpi=100)
        # ax = fig.gca()
        # ext = [0, self.w / self.pix_per_um, self.h / self.pix_per_um, 0]
        # mask = np.zeros(self.images[frame].shape, np.uint8)
        # cv2.drawContours(mask, [contour], 0, 255, -1)
        # ax.imshow(mask, interpolation='none', extent=ext)
        # plt.show()

        fig = plt.figure(dpi=100)
        ax = fig.gca()
        ext = [0, self.w / self.pix_per_um, self.h / self.pix_per_um, 0]
        ax.set_facecolor(sp.colors.sussex_cobalt_blue)
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

        # # compute centroid
        # M = cv2.moments(contour)
        # cx = int(M['m10'] / M['m00']) / self.pix_per_um
        # cy = int(M['m01'] / M['m00']) / self.pix_per_um
        (_x, _y), (_w, _h), _ang = cv2.minAreaRect(contour)
        # cx = (_x + _w/2 * np.cos(np.deg2rad(_ang)))/ self.pix_per_um
        # cy = (_y + _h/2 * np.sin(np.deg2rad(_ang)))/ self.pix_per_um
        cx = _x / self.pix_per_um
        cy = _y / self.pix_per_um
        ax.plot(cx, cy, '.y', markersize=50, zorder=20)

        cpts = np.array(contour.ravel()).reshape((len(contour), 2)) / self.pix_per_um
        ax.plot(cpts[:, 0], cpts[:, 1], '--y', linewidth=1, zorder=15)  # contour

        l = he / 2
        l_sint, l_cost = np.sin(angle_rad) * l, np.cos(angle_rad) * l
        xx1, yy1 = x0 + l_cost, y0 + l_sint
        xx2, yy2 = x0 - l_cost, y0 - l_sint
        ax.plot((x0, xx1), (y0, yy1), color='magenta', linewidth=2, zorder=10)  # minor axis
        ax.plot((x0, xx2), (y0, yy2), color='blue', linewidth=2, zorder=10)  # major axis

        x, y, wbr, hbr = np.array(cv2.boundingRect(contour)) / self.pix_per_um
        bx = (x, x + wbr, x + wbr, x, x)
        by = (y, y, y + hbr, y + hbr, y)
        ax.plot(bx, by, '-.b', linewidth=1)

        _txt_ppt = {'horizontalalignment': 'left', 'verticalalignment': 'center', 'transform': ax.transAxes,
                    'color': "white"}
        ax.text(x + wbr, y + hbr, "orig: %0.2f" % angle_deg, color="white")
        ax.text(x + wbr, y, "trans: %0.2f" % angle_rad, color="white")

        # area = cv2.contourArea(contour)
        # hull = cv2.convexHull(contour)
        # rect = cv2.minAreaRect(contour)
        # box = cv2.boundingRect(contour)
        # hull_area = cv2.contourArea(hull)
        #
        # rect_area = wbr * hbr * self.pix_per_um ** 2
        # solidity = float(area) / hull_area
        # extent = float(area) / rect_area
        # ax.text(0.05, 0.90, 'area: %0.2f' % area, _txt_ppt)
        # ax.text(0.05, 0.85, 'rect: %0.2f' % rect_area, _txt_ppt)
        # ax.text(0.05, 0.80, 'extent: %0.2f' % extent, _txt_ppt)
        # ax.text(0.05, 0.80, 'solidity: %0.2f' % solidity, _txt_ppt)

        ax.set_aspect('equal')
        ax.set_xlim([x - wbr, x + 2 * wbr])
        ax.set_ylim([y - hbr, y + 2 * hbr])

        plt.show()

    @property
    def df(self):
        if self._df is not None: return self._df

        self._segmentation_step()

        log.info("Extracting feature points")
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
                # self._debug_fitted_ellipse_plot(num, cnt)

                # if region.eccentricity < 0.8: continue
                ellipse = cv2.fitEllipse(cnt)
                (x0, y0), (we, he), angle_deg = ellipse  # angle_deg goes from 0 to 180
                if angle_deg != 0 and angle_deg != 180:
                    x0, y0, we, he = np.array([x0, y0, we, he]) / self.pix_per_um
                    angle_rad = np.deg2rad(angle_deg - 90)  # angle_rad goes from -pi/2 to pi/2

                    l = he / 2
                    l_sint, l_cost = np.sin(angle_rad) * l, np.cos(angle_rad) * l
                    xx1, yy1 = x0 + l_cost, y0 + l_sint
                    xx2, yy2 = x0 - l_cost, y0 - l_sint

                    features.append({'x': x0, 'y': y0,
                                     'pt1': Point((xx1, yy1)), 'pt2': Point((xx2, yy2)),
                                     'l': LineString([(xx1, yy1), (xx2, yy2)]),
                                     'x1': xx1, 'y1': yy1, 'x2': xx2, 'y2': yy2,
                                     'theta': angle_rad, 'frame': num})
                else:
                    # compute centroid
                    M = cv2.moments(cnt)
                    area = cv2.contourArea(cnt)
                    if area == 0 or M['m00'] == 0:
                        # box = cv2.boundingRect(cnt)
                        (_x, _y), _sz, _ang = cv2.minAreaRect(cnt)
                        # cx = _x + _w * np.cos(np.deg2rad(_ang))
                        # cy = _y + _h * np.sin(np.deg2rad(_ang))
                        cx = _x / self.pix_per_um
                        cy = _y / self.pix_per_um
                    else:
                        cx = int(M['m10'] / M['m00']) / self.pix_per_um
                        cy = int(M['m01'] / M['m00']) / self.pix_per_um
                    features.append({'x': cx, 'y': cy,
                                     'pt1': None, 'pt2': None,
                                     'l': 0,
                                     'x1': cx, 'y1': cy, 'x2': cx, 'y2': cy,
                                     'theta': np.nan, 'frame': num})
        self._df = pd.DataFrame(features)
        self._df['time'] = self._df['frame'] * self.dt
        log.info("Extraction step completed with %d features detected." % len(self._df))
        return self._df

    @property
    def linked(self):
        if self._linked is not None: return self._linked
        if self.wheel is None: return pd.DataFrame()

        log.info("Linking particles")
        # tp.linking.Linker.MAX_SUB_NET_SIZE = 50
        linked = pd.DataFrame()
        for xb, yb in self.wheel.best:
            _fil = self.wheel.filter_wheel(xb, yb)
            if _fil.empty: continue

            search_range = 5
            pred = tp.predict.NearestVelocityPredict(initial_guess_vels=5)
            linked = linked.append(
                pred.link_df(_fil, search_range, memory=2, link_strategy='auto', adaptive_stop=0.5, ), sort=False)

        #  filter spurious tracks
        frames_per_particle = linked.groupby('particle')['frame'].nunique()
        particles = frames_per_particle[frames_per_particle > 3].index
        linked = linked[linked['particle'].isin(particles)]
        logging.info('filtered %d particles by track length' % linked['particle'].nunique())

        # m = tp.imsd(linked, 1, 1)
        # mt = m.ix[15]
        # particles = mt[mt > 1].index
        # linked = linked[linked['particle'].isin(particles)]
        # logging.info('filtered %d particles msd' % linked['particle'].nunique())

        linked['particle'] = linked['particle'].astype('int32')
        linked['frame'] = linked['frame'].astype('int32')
        linked[['x', 'y', 'time']] = linked[['x', 'y', 'time']].astype('float64')

        self._linked = linked
        log.info("Linking step completed")
        return self._linked

    def render_image(self, ax, frame=None, alpha=1):
        ext = [0, self.w / self.pix_per_um, self.h / self.pix_per_um, 0]
        im = np.max(self.images, axis=0) if frame is None else self.images[frame]
        ax.imshow(im, interpolation='none', extent=ext, cmap=cm.gray, alpha=alpha)
        self._format_axes(ax)

    def render_segmented_image(self, ax, frame=None, color=None, alpha=0.5):
        ext = [0, self.w / self.pix_per_um, self.h / self.pix_per_um, 0]
        im = np.max(self.segmented, axis=0) if frame is None else self.segmented[frame]
        if color is None:
            ax.imshow(im, interpolation='none', extent=ext, cmap=cm.gray, alpha=alpha)
        else:
            im = skcolor.gray2rgb(im)
            im = im * color
            ax.imshow(im, interpolation='none', extent=ext, alpha=alpha)
        self._format_axes(ax)

    def _format_axes(self, ax):
        ax.set_aspect('equal')
        ax.set_xlim(0, self.w / self.pix_per_um)
        ax.set_ylim(0, self.h / self.pix_per_um)

    def render_detected_features(self, ax, frame=None, lines=False, extr_pts=False, alpha=1):
        df = self.df if frame is None else self.df[self.df['frame'] == frame]
        ax.scatter(df['x'], df['y'], s=1, marker='4', c='white', alpha=alpha)
        if lines:
            ax.plot([df['x1'].values, df['x2'].values], [df['y1'].values, df['y2'].values], lw=0.1, c='yellow')
        if extr_pts:
            ax.scatter(df['x1'].values, df['y1'].values, s=0.1, c='magenta', zorder=20)
            ax.scatter(df['x2'].values, df['y2'].values, s=0.1, c='blue', zorder=20)

    def render_linked_features(self, ax, frame=None, wheel=False):
        lnk = self.linked if frame is None else self.linked[self.linked['frame'] == frame]

        if wheel:
            for xb, yb in self.wheel.best:
                self.wheel.plot(xb, yb, ax=ax)

        ax.scatter(lnk['x'].values, lnk['y'].values, s=1, marker='*', c='green', zorder=21)
        # ax.plot([lnk['x1'].values, lnk['x2'].values], [lnk['y1'].values, lnk['y2'].values], lw=0.1, c='yellow')

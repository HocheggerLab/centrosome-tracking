import os

import pandas as pd
import numpy as np
import cv2
from shapely import affinity

from ._segment import keypoint_surf, segment
from ._common import _DEBUG, logger
import plot_special_tools as sp


def _df_to_polar(df):
    # FIXME: each (r, th) must be computed from their respective origin. Currently there are calculated from first nucleus reference frame
    nuc = df["nucleus"].centroid
    dx = df["pt0"].x - nuc.x
    dy = df["pt0"].y - nuc.y
    df["r0"] = np.sqrt(dx ** 2 + dy ** 2)
    # df["th0"] = np.arctan2(dy, dx)
    th = np.arctan(dy / dx)
    df["th0"] = th + np.pi if (dx < 0) else th + 2 * np.pi if (dx > 0 and dy < 0) else th

    dx = df["pt1"].x - nuc.x
    dy = df["pt1"].y - nuc.y
    df["r1"] = np.sqrt(dx ** 2 + dy ** 2)
    th = np.arctan(dy / dx)
    df["th1"] = th + np.pi if (dx < 0) else th + 2 * np.pi if (dx > 0 and dy < 0) else th

    df["th_"] = df["th1"] - df["th0"]
    df["th"] = np.rad2deg(df["th1"] - df["th0"])
    df["a"] = np.rad2deg(df["a1"] - df["a0"])

    return df


class Track():
    def __init__(self, image_file, nucleus_channel=0):
        logger.info("Initializing nucleus track object")
        self.images, self.pix_per_um, self.dt, self.n_frames = sp.load_tiff(image_file)
        self.um_per_pix = 1 / self.pix_per_um
        self._ch = nucleus_channel
        self.im_f = os.path.basename(image_file)
        self.im_p = os.path.dirname(image_file)
        self.w = self.images[0].shape[0]
        self.h = self.images[0].shape[1]
        self._df = None
        self._linked = None
        self._segmented_nuclei = None
        self._nucleus_rotation = None

    @property
    def segmented_nuclei(self):
        if self._segmented_nuclei is not None: return self._segmented_nuclei

        logger.info("Segmenting nuclear boundary")
        self._segmented_nuclei = pd.DataFrame()
        for k, im in enumerate(self.images):
            if _DEBUG and k > 5: break

            img_lbl, detected = segment(im, radius=10 * self.pix_per_um)
            if detected is None: continue

            _df = pd.DataFrame(detected)
            _df.loc[:, "frame"] = k
            self._segmented_nuclei = self._segmented_nuclei.append(_df, sort=False, ignore_index=True)
        # TODO: convert everything to um space for dataframe construction
        return self._segmented_nuclei

    @property
    def nucleus_rotation(self):
        if self._nucleus_rotation is not None: return self._nucleus_rotation

        logger.info("Estimating nuclear rotation")
        dst = cv2.normalize(self.images, None, 0, 255, cv2.NORM_MINMAX)
        matches = keypoint_surf(sp.image_iterator(np.uint8(dst), channel=self._ch, number_of_frames=self.n_frames))
        # matches.loc[:, ['x0', 'y0', 'x1', 'y1']] /= self.pix_per_um

        # print(matches)
        for f in range(1, max(self.segmented_nuclei["frame"].max(), matches["frame"].max()) + 1):
            if _DEBUG and f > 5: break
            for _n, nuc in self.segmented_nuclei[self.segmented_nuclei["frame"] == f].iterrows():
                ix = matches.apply(lambda r: r['frame'] == f and nuc['boundary'].contains(r['pt0']), axis=1)
                if not ix.any(): continue
                matches.loc[ix, 'nucleus'] = nuc['boundary']

        matches.dropna(subset=['nucleus'], inplace=True)
        # matches.drop([['pt0', 'a0']], inplace=True)
        matches = matches.apply(_df_to_polar, axis=1)
        self._nucleus_rotation = matches
        # TODO: convert everything to um space for dataframe construction
        return self._nucleus_rotation

    def render(self, ax, frame, alpha=1):
        # ext = [0, self.w / self.pix_per_um, self.h / self.pix_per_um, 0]
        # im = np.max(self.images, axis=0) if frame is None else self.images[frame]
        # ax.imshow(im, interpolation='none', extent=ext, cmap=cm.gray, alpha=alpha)
        for _n, nuc in self.segmented_nuclei[self.segmented_nuclei["frame"] == frame].iterrows():
            # TODO: move conversion to where it should be
            n_bum = affinity.scale(nuc['boundary'], xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))

            x, y = n_bum.exterior.xy
            ax.plot(x, y, color='red', linewidth=1, solid_capstyle='round', zorder=1)
            cenc = n_bum.centroid
            ax.plot(cenc.x, cenc.y, color='red', marker='+', linewidth=1, solid_capstyle='round', zorder=2)

        for _i, r in self.nucleus_rotation[self.nucleus_rotation["frame"] == frame].iterrows():
            # TODO: move conversion to where it should be
            pt0 = r["pt0"]
            pt1 = r["pt1"]
            ax.annotate("", xy=(pt0.x * self.um_per_pix, pt0.y * self.um_per_pix),
                        xytext=(pt1.x * self.um_per_pix, pt1.y * self.um_per_pix),
                        arrowprops=dict(arrowstyle="->", facecolor='blue'))

        self._format_axes(ax)

    def _format_axes(self, ax):
        ax.set_aspect('equal')
        ax.set_xlim(0, self.w / self.pix_per_um)
        ax.set_ylim(0, self.h / self.pix_per_um)

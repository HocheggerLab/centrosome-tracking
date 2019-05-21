import os

import pandas as pd
import numpy as np
import cv2
from shapely import affinity
from matplotlib import cm
import trackpy as tp

from ._segment import optical_flow_lk_match, segment
from ._common import _DEBUG, logger
import plot_special_tools as sp


def df_to_polar(df):
    # FIXME: each (r, th) must be computed from their respective origin. Currently there are calculated from first nucleus reference frame
    nuc = df["nuc_bnd"].centroid
    dx = df["pt"].x - nuc.x
    dy = df["pt"].y - nuc.y
    df["r"] = np.sqrt(dx ** 2 + dy ** 2)
    th = np.arctan(dy / dx)
    df["th"] = th + np.pi if (dx < 0) else th + 2 * np.pi if (dx > 0 and dy < 0) else th

    return df


def angle_correction(df):
    df['th+'] = df['th']
    # from zero to 2*pi
    _ixd = (df['th+'].shift(-1) >= 3 * np.pi / 2) & (df['th+'] <= np.pi / 2)
    while _ixd.sum() > 0:
        df.loc[_ixd, 'th+'] += np.pi * 2
        _ixd = (df['th+'].shift(-1) >= 3 * np.pi / 2) & (df['th+'] <= np.pi / 2)
    # from 2*pi to zero
    _ixd = (df['th+'].shift(-1) <= np.pi / 2) & (df['th+'] >= 3 * np.pi / 2)
    while _ixd.sum() > 0:
        df.loc[_ixd, 'th+'] -= np.pi * 2
        _ixd = (df['th+'].shift(-1) <= np.pi / 2) & (df['th+'] >= 3 * np.pi / 2)

    df.loc[:, 'th_dev'] = df['th+'] - df['th+'].iloc[0]
    # df.loc[:, 'th_dev'] = np.abs(df['th+'] - df['th+'].iloc[0])

    return df


def velocity(df, time='time', frame='frame'):
    df = df.set_index(frame).sort_index()
    dx = df["pt"].apply(lambda p: p.x).diff()
    dy = df["pt"].apply(lambda p: p.y).diff()
    dt = df[time].diff()
    df.loc[:, 'Vx'] = dx / dt
    df.loc[:, 'Vy'] = dy / dt
    return df.reset_index()


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
    def boundary(self):
        if self._segmented_nuclei is not None: return self._segmented_nuclei

        logger.info("Segmenting nuclear boundary")
        self._segmented_nuclei = pd.DataFrame()
        for k, im in enumerate(sp.image_iterator(self.images, channel=self._ch, number_of_frames=self.n_frames)):
            if _DEBUG and k > 5: break

            img_lbl, detected = segment(im, radius=10 * self.pix_per_um)
            if detected is None: continue

            _df = pd.DataFrame(detected)
            _df.loc[:, "frame"] = k
            self._segmented_nuclei = self._segmented_nuclei.append(_df, sort=False, ignore_index=True)

        self._segmented_nuclei.loc[:, "frame"] = self._segmented_nuclei["frame"].astype(int)

        # TODO: convert everything to um space for dataframe construction
        # self._segmented_nuclei.loc[:, ["x", "y"]] *= self.um_per_pix
        # self._segmented_nuclei.loc[:, "boundary"] = affinity.scale(self._segmented_nuclei["boundary"],
        #                                                            xfact=self.um_per_pix, yfact=self.um_per_pix,
        #                                                            origin=(0, 0, 0))

        logger.info("Linking nuclei particles")
        self._segmented_nuclei.drop("id", axis=1, inplace=True)
        search_range = 1.5 * self.pix_per_um
        pred = tp.predict.NearestVelocityPredict(initial_guess_vels=0.5 * self.pix_per_um)
        self._segmented_nuclei = pred.link_df(self._segmented_nuclei, search_range, memory=2, link_strategy='auto',
                                              adaptive_stop=0.5)

        return self._segmented_nuclei

    @property
    def nucleus_rotation(self):
        if self._nucleus_rotation is not None: return self._nucleus_rotation

        logger.info("Estimating nuclear rotation")
        nrm = np.uint8(cv2.normalize(self.images, None, 0, 255, cv2.NORM_MINMAX))
        # matches = keypoint_surf(sp.image_iterator(nrm, channel=self._ch, number_of_frames=self.n_frames))
        matches = optical_flow_lk_match(sp.image_iterator(-nrm, channel=self._ch, number_of_frames=self.n_frames))
        # matches = pd.DataFrame()
        # for _pi, _sn in self.boundary.groupby("particle"):
        #     if len(_sn) < 2: continue
        #     print(_pi, _sn)
        #     _m = keypoint_surf(sp.image_iterator(np.uint8(nrm), channel=self._ch, number_of_frames=self.n_frames),
        #                        mask_polygons=_sn)
        #     matches = matches.append(_m, ignore_index=True, sort=False)
        # matches.loc[:, ['x0', 'y0', 'x1', 'y1']] /= self.pix_per_um

        for f in range(1, max(self.boundary["frame"].max(), matches["frame"].max()) + 1):
            if _DEBUG and f > 5: break
            for _n, nuc in self.boundary[self.boundary["frame"] == f].iterrows():
                ix = matches.apply(lambda r: r['frame'] == f and nuc['boundary'].contains(r['pt']), axis=1)
                if not ix.any(): continue
                matches.loc[ix, 'nucleus'] = nuc['particle']
                matches.loc[ix, 'nuc_bnd'] = nuc['boundary']
                matches.loc[ix, 'frame'] = f

        print(matches.columns)
        matches.dropna(subset=['nucleus'], inplace=True)
        matches.loc[:, 'nucleus'] = matches['nucleus'].astype(int)

        matches = matches.apply(df_to_polar, axis=1)
        # deal with angle jumps
        matches = matches.groupby(["nucleus", "particle"]).apply(angle_correction).reset_index(drop=True)
        # compute angular speed
        matches.loc[:, 'omega'] = matches['th+'].diff()

        matches.loc[:, "time"] = matches["frame"] * self.dt
        matches = matches.groupby('particle').apply(velocity).reset_index(drop=True)
        # matches.drop(['nucleus', 'pt1'], axis=1, inplace=True)

        print(matches)
        self._nucleus_rotation = matches
        # TODO: convert everything to um space for dataframe construction
        return self._nucleus_rotation

    def render(self, ax, frame, alpha=1):
        ext = [0, self.w / self.pix_per_um, self.h / self.pix_per_um, 0]
        im = sp.retrieve_image(self.images, frame, channel=2, number_of_frames=self.n_frames)
        ax.imshow(im, interpolation='none', extent=ext, cmap=cm.gray, alpha=alpha)

        # Create some random colors
        # color = sns.color_palette('bright', p0.shape[0])

        for _n, nuc in self.boundary[self.boundary["frame"] == frame].iterrows():
            # TODO: move conversion to where it should be
            n_bum = affinity.scale(nuc['boundary'], xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0))

            x, y = n_bum.exterior.xy
            ax.plot(x, y, color='red', linewidth=1, solid_capstyle='round', zorder=1)
            cenc = n_bum.centroid
            ax.plot(cenc.x, cenc.y, color='red', marker='+', linewidth=1, solid_capstyle='round', zorder=2)

        for _in, nucp in self.nucleus_rotation.groupby(["nucleus", "particle"]):
            pt0 = nucp.loc[nucp["frame"] == frame, "pt"].values
            pt1 = nucp.loc[nucp["frame"] == frame + 1, "pt"].values
            if pt0.size == 0 or pt1.size == 0: continue

            # TODO: move conversion to where it should be
            pt0 = pt0[0]
            pt1 = pt1[0]
            ax.annotate("", xy=(pt1.x * self.um_per_pix, pt1.y * self.um_per_pix),
                        xytext=(pt0.x * self.um_per_pix, pt0.y * self.um_per_pix),
                        arrowprops=dict(arrowstyle="->", color='yellow'))

        self._format_axes(ax)

    def _format_axes(self, ax):
        ax.set_aspect('equal')
        ax.set_xlim(0, self.w / self.pix_per_um)
        ax.set_ylim(0, self.h / self.pix_per_um)

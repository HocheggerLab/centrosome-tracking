import os

import pandas as pd
import numpy as np
import cv2
from shapely import affinity
from shapely.geometry.point import Point
from matplotlib import cm
import trackpy as tp
import seaborn as sns
import pysketcher as ps

import tools.image as image
from ._segment import optical_flow_lk_match, segment
from ._common import _DEBUG, logger
import tools.plot_tools as sp
import mechanics as m


def df_to_polar(df):
    nuc = df["boundary"].centroid
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
    def __init__(self, image_file, nucleus_channel=0, skip_frames=0):
        logger.info("Initializing nucleus track object")
        self.images, self.pix_per_um, self.dt, self.n_frames, self.n_channels = image.load_tiff(image_file)
        self.um_per_pix = 1 / self.pix_per_um
        self._ch = nucleus_channel
        self.im_f = os.path.basename(image_file)
        self.im_p = os.path.dirname(image_file)
        self.w = self.images[0].shape[0]
        self.h = self.images[0].shape[1]
        self._df = None
        self._linked = None
        self._boundary_pix = None
        self._features_pix = None
        self._rotation_pix = None

        # skip first skip_frames frames
        self.images = self.images[self.n_channels * skip_frames:, :]
        self.n_frames -= skip_frames

    def _segment_boundary(self):
        logger.info("Segmenting nuclear boundary")
        self._boundary_pix = pd.DataFrame()
        for k, im in enumerate(image.image_iterator(self.images, channel=self._ch, number_of_frames=self.n_frames)):
            if _DEBUG and k > 5: break

            img_lbl, detected = segment(im, radius=10 * self.pix_per_um)
            if detected is None: continue

            _df = pd.DataFrame(detected)
            _df.loc[:, "frame"] = k
            self._boundary_pix = self._boundary_pix.append(_df, sort=False, ignore_index=True)

        self._boundary_pix.loc[:, "frame"] = self._boundary_pix["frame"].astype(int)

        logger.info("Linking nuclei particles")
        self._boundary_pix.drop("id", axis=1, inplace=True)
        search_range = 1.5 * self.pix_per_um
        pred = tp.predict.NearestVelocityPredict(initial_guess_vels=0.5 * self.pix_per_um)
        self._boundary_pix = pred.link_df(self._boundary_pix, search_range, memory=2, link_strategy='auto',
                                          adaptive_stop=0.5)

    @property
    def position(self):
        if self._boundary_pix is None:
            self._segment_boundary()

        # convert everything to um space for dataframe construction
        nuc_um = self._boundary_pix[["frame", "particle", "x", "y"]].copy()
        nuc_um.loc[:, ["x", "y"]] *= self.um_per_pix

        return nuc_um  # .set_index(["frame", "particle"]).sort_index()

    @property
    def boundary(self):
        if self._boundary_pix is None:
            self._segment_boundary()

        # convert everything to um space for dataframe construction
        nuc_um = self._boundary_pix[["frame", "particle", "boundary"]].copy()
        nuc_um.loc[:, "boundary"] = nuc_um["boundary"].apply(
            lambda b: affinity.scale(b, xfact=self.um_per_pix, yfact=self.um_per_pix, origin=(0, 0, 0)))

        return nuc_um

    @property
    def features(self):
        if self._features_pix is None:

            if self._boundary_pix is None:
                self._segment_boundary()

            nrm = np.uint8(cv2.normalize(self.images, None, 0, 255, cv2.NORM_MINMAX))
            self._features_pix = pd.DataFrame()

            for _p, nuc in self._boundary_pix.groupby("particle"):
                pt_in_nuc = optical_flow_lk_match(
                    image.mask_iterator(
                        image.image_iterator(nrm, channel=self._ch, number_of_frames=self.n_frames),
                        list(nuc.set_index("frame").sort_index()["boundary"].items())
                    )
                )
                if pt_in_nuc.empty: continue

                pt_in_nuc.loc[:, 'nucleus'] = _p
                self._features_pix = self._features_pix.append(pt_in_nuc, ignore_index=True, sort=False)
            assert not self._features_pix.empty

            self._features_pix.dropna(subset=['particle'], inplace=True)
            self._features_pix.loc[:, 'particle'] = self._features_pix['particle'].astype(int)

        # convert everything to um space for dataframe construction
        feat_um = self._features_pix[["frame", "nucleus", "particle", "pt"]].copy()
        feat_um.loc[:, "pt"] = feat_um["pt"].apply(lambda p: Point(p.x * self.um_per_pix, p.y * self.um_per_pix))
        feat_um.loc[:, "x"] = feat_um["pt"].apply(lambda p: p.x)
        feat_um.loc[:, "y"] = feat_um["pt"].apply(lambda p: p.y)
        feat_um.loc[:, "time"] = feat_um["frame"] * self.dt
        # compute velocity and  MSD
        feat_um = feat_um.groupby(["nucleus", "particle"]).apply(velocity).reset_index(drop=True)
        feat_um = m.get_msd(feat_um, group=["nucleus", "particle"])
        return feat_um

    @property
    def rotation(self):
        if self._rotation_pix is not None: return self._rotation_pix

        logger.info("Estimating nuclear rotation")
        extended_feat = pd.merge(self.features, self.boundary.rename(columns={"particle": "nucleus"}),
                                 on=["frame", "nucleus"], how="left").dropna(subset=["boundary"])
        self._rotation_pix = extended_feat.apply(df_to_polar, axis=1)
        self._rotation_pix.drop("boundary", axis=1, inplace=True)
        # deal with angle jumps
        self._rotation_pix = self._rotation_pix.groupby(["nucleus", "particle"]).apply(angle_correction).reset_index(
            drop=True)
        # compute angular speed
        self._rotation_pix.loc[:, 'omega'] = self._rotation_pix['th+'].diff()

        return self._rotation_pix[["frame", "particle", "th+", "omega"]]

    def render(self, ax, frame, alpha=1):
        ext = [0, self.w / self.pix_per_um, self.h / self.pix_per_um, 0]
        im = image.retrieve_image(self.images, frame, channel=2, number_of_frames=self.n_frames)
        ax.imshow(im, interpolation='none', extent=ext, cmap=cm.gray, alpha=alpha)

        for _n, nuc in self.boundary[self.boundary["frame"] == frame].iterrows():
            x, y = nuc['boundary'].exterior.xy
            ax.plot(x, y, color='red', linewidth=1, solid_capstyle='round', zorder=1)
            cenc = nuc['boundary'].centroid
            ax.plot(cenc.x, cenc.y, color='red', marker='+', linewidth=1, solid_capstyle='round', zorder=2)

        # filter data
        df = self.rotation
        df = df[(~df['particle'].isin(df[df['msd'].diff() > 500]['particle'].unique()))]
        df = df[~df['particle'].isin(
            df.loc[(df['r'].diff() < 0) & (df['frame'] > 2) & (df['frame'] < 5), 'particle'].unique())]
        df = df[~df['particle'].isin(df.loc[(df['r'].diff() > 5) & (df['frame'] > 5), 'particle'].unique())]

        # plot frame reference for nucleus
        for _in, nucp in df.groupby(["nucleus"]):
            foi = nucp[nucp["frame"] == frame]
            if foi.empty: continue
            c_pt = self.boundary.loc[(self.boundary["frame"] == frame) & (self.boundary["particle"] == _in), "boundary"]
            if c_pt.empty: continue
            c_pt = (c_pt.iloc[0].centroid.x, c_pt.iloc[0].centroid.y)

            rotation = df.loc[df["frame"] == frame, "th+"].mean()
            x_axis = ps.Axis(c_pt, 10, 'Nx', rotation_angle=0)
            y_axis = ps.Axis(c_pt, 10, 'Ny', rotation_angle=90)
            x_axis.shapes['label'].fgcolor = 'white'
            y_axis.shapes['label'].fgcolor = 'white'
            system = ps.Composition({'x_axis': x_axis, 'y_axis': y_axis})
            system.set_linecolor('white')
            system.set_linecolor('white')
            system.rotate(np.rad2deg(rotation), c_pt)
            system.draw()

        particle_group = df.groupby(["nucleus", "particle"])
        # Create some random colors
        trace_colors = sns.color_palette('bright', len(particle_group))
        for pk, (_in, nucp) in enumerate(particle_group):
            pt0 = nucp.loc[nucp["frame"] == frame, "pt"].values
            pt1 = nucp.loc[nucp["frame"] == frame + 1, "pt"].values
            if pt0.size == 0 or pt1.size == 0: continue

            pt0, pt1 = pt0[0], pt1[0]
            ax.annotate("", xy=(pt1.x, pt1.y), xytext=(pt0.x, pt0.y), arrowprops=dict(arrowstyle="->", color='yellow'))

            # plot a path line up to frame
            pts = nucp.loc[nucp["frame"] <= frame, "pt"].values
            ax.plot([p.x * self.um_per_pix for p in pts], [p.y * self.um_per_pix for p in pts], color=trace_colors[pk],
                    alpha=0.5)
        self._format_axes(ax)

    def _format_axes(self, ax):
        ax.set_aspect('equal')
        ax.set_xlim(0, self.w / self.pix_per_um)
        ax.set_ylim(0, self.h / self.pix_per_um)

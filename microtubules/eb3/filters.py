import logging
import warnings

import pandas as pd
import numpy as np
from shapely.geometry import LinearRing, Point, Polygon
import geopandas as gp

from tools.measurements import integral_over_surface

warnings.filterwarnings("ignore", category=DeprecationWarning)
log = logging.getLogger(__name__)


def _angle(pt1, pt2):
    dx = pt2.x - pt1.x
    dy = pt2.y - pt1.y

    if dy >= 0:
        return np.arctan2(dy, dx)
    else:
        return np.arctan2(dy, dx) + 2 * np.pi


class Wheel:
    def __init__(self, df, image, radius=10, number_of_divisions=16):
        self.df = df
        self.dfg = None
        self.image = image
        self.radius = radius
        self.n = 2 ** int(np.log2(number_of_divisions))
        self.angle_delta = 2 * np.pi / number_of_divisions
        self.best = list()

    def triangle(self, x, y, angle_start=0):
        tri = Polygon([(x, y),
                       (x + self.radius * np.cos(angle_start), y + self.radius * np.sin(angle_start)),
                       (x + self.radius * np.cos(angle_start + self.angle_delta),
                        y + self.radius * np.sin(angle_start + self.angle_delta))
                       ])
        return tri

    def add(self, x, y):
        self.best.append((x, y))

    def filter_wheel(self, x, y, ax=None):
        inp = self.df
        in_idx = ((inp['x1'] > x - self.radius) | (inp['x2'] > x - self.radius)) & \
                 ((inp['x1'] < x + self.radius) | (inp['x2'] < x + self.radius)) & \
                 ((inp['y1'] > y - self.radius) | (inp['y2'] > y - self.radius)) & \
                 ((inp['y1'] < y + self.radius) | (inp['y2'] < y + self.radius))
        df = inp[in_idx]
        out = pd.DataFrame()
        if len(df) == 0: return out

        # fill four cuadrants
        divs_per_cuadrant = int(self.n / 4)
        for cuadrant in range(1, 5):
            pn_idx = df['theta'].apply(lambda t: t > 0 if (cuadrant == 1 or cuadrant == 3) else t < 0)
            dc = df[pn_idx]
            for i in range(divs_per_cuadrant):
                # if cuadrant != 4 or i != 1: continue
                ang_i = self.angle_delta * i
                ang_ii = ang_i + self.angle_delta
                ang_avg = (ang_ii + ang_i) / 2

                # build triangle
                c_ang = (cuadrant - 1) / 2 * np.pi
                tri = self.triangle(x, y, angle_start=ang_i + c_ang)

                in_idx = dc['pt1'].apply(lambda pt: pt.within(tri)) & dc['pt2'].apply(lambda pt: pt.within(tri))
                dcc = dc[in_idx]

                if cuadrant == 2 or cuadrant == 4:
                    dcc.loc[:, 'theta'] += np.pi / 2
                dcc = dcc[(dcc['theta'] + ang_avg - np.pi / 2) < self.angle_delta]
                if cuadrant == 1 or cuadrant == 2:
                    dcc.loc[:, 'x'] = dcc['x1']
                    dcc.loc[:, 'y'] = dcc['y1']
                if cuadrant == 3 or cuadrant == 4:
                    dcc.loc[:, 'x'] = dcc['x2']
                    dcc.loc[:, 'y'] = dcc['y2']
                dcc.drop(columns=['x1', 'y1', 'x1', 'x2'])
                out = out.append(dcc)

                if ax is not None:
                    ax.plot(tri.exterior.xy[0], tri.exterior.xy[1], lw=0.1, c='white')
                    ax.scatter(dcc['x'], dcc['y'], s=15, c='green', zorder=10)
                    ax.scatter(dcc['x1'], dcc['y1'], s=2, c='blue', zorder=20)
                    ax.scatter(dcc['x2'], dcc['y2'], s=2, c='red', zorder=20)
                    ax.plot([dcc['x1'], dcc['x2']], [dcc['y1'], dcc['y2']], lw=1, c='white', alpha=1)

        return out

    def count_lines_with_same_slope(self, x, y):
        box = Polygon(
            [(x - self.radius, y - self.radius),
             (x - self.radius, y + self.radius),
             (x + self.radius, y + self.radius),
             (x + self.radius, y - self.radius)])
        if min(box.bounds) < 0 or min(box.bounds) > 512: return [0]

        df = self.df
        log.debug('- 0', df['pt1'].count())
        in_idx = ((df['x1'] > x - self.radius) | (df['x2'] > x - self.radius)) & \
                 ((df['x1'] < x + self.radius) | (df['x2'] < x + self.radius)) & \
                 ((df['y1'] > y - self.radius) | (df['y2'] > y - self.radius)) & \
                 ((df['y1'] < y + self.radius) | (df['y2'] < y + self.radius))
        _df = df[in_idx]
        if len(_df) == 0: return [0]
        log.debug('- 1', len(_df))

        # fill four cuadrants
        cuadrants = list()
        divs_per_cuadrant = int(self.n / 4)
        for cuadrant, color in zip(range(1, 5), ['red', 'blue', 'green', 'yellow']):
            o = 0
            pn_idx = _df['theta'].apply(lambda t: t > 0 if (cuadrant == 1 or cuadrant == 3) else t < 0)
            dc = _df[pn_idx]
            log.debug('- 2', dc['pt1'].count())
            for i in range(divs_per_cuadrant):
                # if cuadrant != 1 or i != 0: continue
                ang_i = self.angle_delta * i
                ang_ii = ang_i + self.angle_delta
                ang_avg = (ang_ii + ang_i) / 2

                # build triangle
                c_ang = (cuadrant - 1) / 2 * np.pi
                tri = self.triangle(x, y, angle_start=ang_i + c_ang)

                in_idx = dc['pt1'].apply(lambda pt: pt.within(tri)) & dc['pt2'].apply(lambda pt: pt.within(tri))
                if len(dc[in_idx]) == 0: continue
                log.debug('- 3', dc[in_idx]['pt1'].count())

                if (cuadrant == 2 or cuadrant == 4):
                    dc.loc[in_idx, 'theta'] += np.pi / 2
                in_ang = np.abs(dc.loc[in_idx, 'theta'] + ang_avg - np.pi / 2) < self.angle_delta
                log.debug('- 4', dc[in_idx & in_ang]['pt1'].count())
                if len(dc[in_idx & in_ang]) == 0: continue
                o += len(dc[in_idx & in_ang])

            cuadrants.append(o)

        return cuadrants

    def dev_from_circle_angles(self, x, y):
        if self.dfg is None:
            self.dfg = gp.GeoDataFrame(self.df, geometry="l")
        center = Point(x, y)
        circle = LinearRing(list(center.buffer(4).exterior.coords))
        in_idx = self.dfg["l"].apply(lambda r: circle.crosses(r))
        df = self.dfg[in_idx]
        if len(df) == 0: return 0

        df.loc[:, "i"] = df["l"].intersection(circle)
        df = df[df["i"].apply(lambda r: type(r) == Point)]

        # df.loc[:, "angle"] = df["i"].apply(lambda r: np.arctan2(center.x - r.x, center.y - r.y))
        df.loc[:, "angle"] = df["i"].apply(lambda r: _angle(center, r))
        df.loc[(df["angle"] > np.pi) & (df["angle"] <= 2 * np.pi), "theta"] += np.pi
        df.loc[:, "adiff"] = np.abs(df["angle"] - df["theta"])

        fn = 1 / df["adiff"].sum() + len(df) + integral_over_surface(self.image, center.buffer(1)) * 1e-4

        return fn

    def plot(self, x, y, ax):
        # divs_per_cuadrant = int(self.n / 4)
        # for cuadrant, color in zip(range(1, 5), ['red', 'blue', 'green', 'yellow']):
        #     for i in range(divs_per_cuadrant):
        #         # build triangle
        #         ang_i = self.angle_delta * i
        #         c_ang = (cuadrant - 1) / 2 * np.pi
        #         tri = self.triangle(x, y, angle_start=ang_i + c_ang)
        #         ax.plot(tri.exterior.xy[0], tri.exterior.xy[1], lw=0.1, c='white')
        #         # for ix, row in dc[in_idx & in_ang].iterrows():
        #         #     ax.scatter(row['x1'], row['y1'], s=5, c='blue')
        #         #     ax.scatter(row['x2'], row['y2'], s=5, c='red')
        #         #     ax.plot([row['x1'], row['x2']], [row['y1'], row['y2']], lw=1, c='white', alpha=1)
        center = Point(x, y)
        circle = LinearRing(list(center.buffer(4).exterior.coords))
        in_idx = self.df["l"].apply(lambda r: circle.crosses(r))
        df = self.df[in_idx]

        # df.loc[:, "i"] = df["l"].intersection(circle)
        # df.loc[:, "angle"] = df["i"].apply(lambda r: _angle(center, r))
        # df.loc[(df["angle"] > np.pi) & (df["angle"] <= 2 * np.pi), "theta"] += np.pi
        # df = df.set_index("angle").sort_index().reset_index()
        # # df = df[(df["angle"] > 0) & (df["angle"] < np.pi / 2)]
        # # df = df[(df["angle"] > np.pi / 2) & (df["angle"] < np.pi)]
        # # df = df[(df["angle"] > np.pi) & (df["angle"] < 3 / 2 * np.pi)]
        # df = df[(df["angle"] > 3 / 2 * np.pi) & (df["angle"] < 2 * np.pi)]
        # print(df[["angle", "theta"]])
        # import matplotlib.pyplot as plt
        # df[["angle", "theta"]].plot()
        # plt.show()

        df["l"].plot(lw=0.5, color="red", ax=ax)
        ax.plot(circle.coords.xy[0], circle.coords.xy[1], c="white")

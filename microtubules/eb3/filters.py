import logging
import warnings

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import geopandas as gp

warnings.filterwarnings("ignore", category=DeprecationWarning)
log = logging.getLogger(__name__)


class Wheel:
    def __init__(self, df, radius=10, number_of_divisions=16):
        self.df = gp.GeoDataFrame(df, geometry="l")
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

    def filter_wheel(self, x, y):
        ang_diff = 2 * np.pi / self.n

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
            pn_idx = df['theta'].apply(lambda t: t < 0 if (cuadrant == 0 or cuadrant == 2) else t > 0)
            dc = df[pn_idx]
            for i in range(divs_per_cuadrant):
                ang_i = ang_diff * i
                ang_ii = ang_i + ang_diff
                ang_avg = (ang_ii + ang_i) / 2

                # build triangle
                c_ang = (cuadrant - 1) / 2 * np.pi
                tri = self.triangle(x, y, angle_start=ang_i + c_ang)

                in_idx = dc['pt1'].apply(lambda pt: pt.within(tri)) & dc['pt2'].apply(lambda pt: pt.within(tri))
                dcc = dc[in_idx]

                if cuadrant == 2 or cuadrant == 4:
                    dcc.loc[:, 'theta'] += np.pi / 2
                dcc = dcc[(dcc['theta'] + ang_avg - np.pi / 2) < ang_diff]
                if cuadrant == 1 or cuadrant == 2:
                    dcc.loc[:, 'x'] = dcc['x1']
                    dcc.loc[:, 'y'] = dcc['y1']
                if cuadrant == 3 or cuadrant == 4:
                    dcc.loc[:, 'x'] = dcc['x2']
                    dcc.loc[:, 'y'] = dcc['y2']
                dcc.drop(columns=['x1', 'y1', 'x1', 'x2'])
                out = out.append(dcc)

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

    def plot(self, x, y, ax):
        divs_per_cuadrant = int(self.n / 4)
        for cuadrant, color in zip(range(1, 5), ['red', 'blue', 'green', 'yellow']):
            for i in range(divs_per_cuadrant):
                # build triangle
                ang_i = self.angle_delta * i
                c_ang = (cuadrant - 1) / 2 * np.pi
                tri = self.triangle(x, y, angle_start=ang_i + c_ang)
                ax.plot(tri.exterior.xy[0], tri.exterior.xy[1], lw=0.1, c='white')
                # for ix, row in dc[in_idx & in_ang].iterrows():
                #     ax.scatter(row['x1'], row['y1'], s=5, c='blue')
                #     ax.scatter(row['x2'], row['y2'], s=5, c='red')
                #     ax.plot([row['x1'], row['x2']], [row['y1'], row['y2']], lw=1, c='white', alpha=1)

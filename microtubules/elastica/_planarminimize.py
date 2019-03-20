import logging

import numpy as np
import pandas as pd
from lmfit import Minimizer
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt

from ._imageplanar import ImagePlanarElastica
from tools.draggable import DraggableCircle

logger = logging.getLogger(__name__)


class PlanarImageMinimizerIVP(ImagePlanarElastica):
    def __init__(self, axes,
                 w=1.0, k0=1.0, alpha=np.pi / 2, phi=0.0,
                 L=1.0, E=0.625, J=1.0, F=1.0, x0=0.0, y0=0.0, m0=0.01, theta=3 * np.pi / 2,
                 image=None, callback=None):
        super().__init__(axes, w=w, k0=k0, alpha=alpha, phi=phi, L=L, E=E, J=J, F=F, x0=x0, y0=y0, m0=m0, theta=theta,
                         image=image, callback=callback)

        self._pt_fit = None  # point for doing the fit
        self.fit_pt = (x0 + 2, y0)

    def select(self):
        super().select()
        self._pt_fit.show()

    def unselect(self):
        super().unselect()
        self._pt_fit.hide()

    @property
    def fit_pt(self):
        return self._pt_fit.circle.center

    @fit_pt.setter
    def fit_pt(self, value):
        x, y = value
        ftc = plt.Circle((x, y), radius=0.5, fc='magenta', picker=1, zorder=100)
        self.ax.add_artist(ftc)
        self._pt_fit = DraggableCircle(ftc)
        self._pt_fit.connect()
        if not self.selected: self._pt_fit.hide()

    def fit(self):
        def objective(params):
            self.parameters = params

            lineintegral, number_of_pixels = self.get_line_integral_over_image()

            p = Point(self._pt_fit)
            c = Polygon([(x, y) for x, y in zip(self.curve_x, self.curve_y)])
            obj = 1 / lineintegral + p.distance(c)
            # obj = p.distance(c)

            params.add('f_obj', value=obj)
            self._param_exploration.append(params.copy())

            # logger.debug('objective function = {:06.4e}'.format(obj))
            return obj

        inip = self.parameters
        inip['x0'].min = inip['x0'].value - 0.2
        inip['x0'].max = inip['x0'].value + 0.2
        inip['y0'].min = inip['y0'].value - 0.2
        inip['y0'].max = inip['y0'].value + 0.2
        inip['phi'].min = inip['phi'].value * 0.8
        inip['phi'].max = inip['phi'].value * 1.2
        inip['L'].min = inip['L'].value * 0.8
        inip['L'].max = inip['L'].value * 1.2

        inip['alpha'].min = 0
        inip['alpha'].max = 2 * np.pi
        inip['w'].min = 0
        inip['w'].max = 20
        inip['k0'].min = -1
        inip['k0'].max = 1

        logger.info(inip)

        self._param_exploration = []

        fitter = Minimizer(objective, inip)
        # result = fitter.minimize(method='bgfs', params=inip)
        result = fitter.minimize(method='basinhopping', params=inip)
        # niter=10 ** 4, niter_success=10 ** 2)

        return result

    def plot_fit(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import cm
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        df = pd.DataFrame([p.valuesdict() for p in self._param_exploration])
        df = df.set_index('f_obj').sort_index(ascending=False).reset_index()
        print(df)
        print(df.set_index('f_obj').sort_index().iloc[0])

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(df["w"], df["k0"], df["alpha"], c=df["f_obj"], cmap='viridis')

        fig = plt.figure()
        ax = fig.gca()
        norm = mpl.colors.Normalize(vmin=df["f_obj"].min(), vmax=df["f_obj"].max())
        color = cm.ScalarMappable(norm=norm, cmap='viridis')
        for p in self._param_exploration:
            self.parameters = p
            self.eval()
            ax.plot(self.curve_x, self.curve_y, lw=1, c=color.to_rgba(p["f_obj"]))
        # fig.colorbar(color)

        plt.show(block=False)

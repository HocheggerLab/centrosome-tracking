import logging

import numpy as np
from lmfit import Parameters
import pandas as pd
from lmfit import Minimizer
from shapely.geometry import LineString, Point
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

        self.fit_pt = (x0 + 2, y0)
        self._param_exploration = []

    def select(self):
        super().select()
        for pt in self._pt_fit: pt.show()

    def unselect(self):
        super().unselect()
        for pt in self._pt_fit: pt.hide()

    @property
    def fit_pt(self):
        if type(self._pt_fit) == DraggableCircle:
            return [self._pt_fit.circle.center]
        elif type(self._pt_fit) == list:
            return [pt.circle.center for pt in self._pt_fit]

    @fit_pt.setter
    def fit_pt(self, value):
        if type(value) == tuple:
            x, y = value
            ftc = plt.Circle((x, y), radius=0.25, fc='magenta', picker=.25, zorder=100)
            self.ax.add_artist(ftc)
            ptf = DraggableCircle(ftc)
            ptf.connect()
            self._pt_fit = [ptf]
            if not self.selected: ptf.hide()
        elif type(value) == list:
            self._pt_fit = list()
            for pt in value:
                x, y = pt
                ftc = plt.Circle((x, y), radius=0.25, fc='magenta', picker=.25, zorder=100)
                self.ax.add_artist(ftc)
                ptf = DraggableCircle(ftc)
                ptf.connect()
                self._pt_fit.append(ptf)
                if not self.selected: ptf.hide()

    @property
    def parameters(self):
        params = Parameters()
        params.add('L', value=self.L, vary=False)
        params.add('x0', value=self.x0, vary=False)
        params.add('y0', value=self.y0, vary=False)
        params.add('phi', value=self.phi, vary=False)
        params.add('theta0', value=self.theta0)

        params.add('w', value=self.w)
        params.add('k0', value=self.k0)
        params.add('alpha', value=self.alpha)

        return params

    @parameters.setter
    def parameters(self, value):
        vals = value.valuesdict()

        self.L = vals['L']
        self.x0 = vals['x0']
        self.y0 = vals['y0']
        self.phi = vals['phi']
        self.theta0 = vals['theta0']

        self._w = vals['w']
        self._k0 = vals['k0']
        self.alpha = vals['alpha']
        self.w_was_calculated = False
        self.k0_was_calculated = False

    def fit(self):
        def objective(params):
            self.parameters = params

            self.eval(num_points=100)
            lineintegral, number_of_pixels = self.get_line_integral_over_image()
            c = LineString([(x, y) for x, y in zip(self.curve_x, self.curve_y)])

            o0 = 1 / lineintegral
            # o0 = 0
            o1 = 0
            for pt in self.fit_pt:
                o1 += Point(pt).distance(c)
            obj = o0 + o1

            params.add('f_obj', value=obj)
            self._param_exploration.append(params.copy())

            logger.debug('objective function = {:06.4e} + {:06.4e} = {:06.4e}'.format(o0, o1, obj))
            return obj

        inip = self.parameters
        # inip['x0'].min = inip['x0'].value - 0.2
        # inip['x0'].max = inip['x0'].value + 0.2
        # inip['y0'].min = inip['y0'].value - 0.2
        # inip['y0'].max = inip['y0'].value + 0.2
        inip['phi'].min = inip['phi'].value * 0.9
        inip['phi'].max = inip['phi'].value * 1.1
        # inip['theta0'].min = inip['theta0'].value * 0.8
        # inip['theta0'].max = inip['theta0'].value * 1.2

        inip['alpha'].min = 0
        inip['alpha'].max = 2 * np.pi
        inip['theta0'].min = 0
        inip['theta0'].max = 2 * np.pi
        inip['w'].min = 0
        inip['w'].max = 20
        inip['k0'].min = -1
        inip['k0'].max = 1

        logger.info(inip)
        self._param_exploration = []

        fitter = Minimizer(objective, inip)
        result = fitter.minimize(method='bgfs', params=inip)
        # result = fitter.minimize(method='basinhopping', params=inip)#, niter=10 ** 4, niter_success=1000)
        print(len(self._param_exploration))

        return result

    def plot_fit(self):
        if len(self._param_exploration) == 0: return
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import cm
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        df = pd.DataFrame([p.valuesdict() for p in self._param_exploration])
        df = df.set_index('f_obj').sort_index(ascending=False).reset_index()
        # print(df)
        print(df['f_obj'].describe())
        # print(df.set_index('f_obj').sort_index().iloc[0])

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
            ax.plot(self.curve_x, self.curve_y, lw=1, c=color.to_rgba(p["f_obj"]), zorder=10)
        # fig.colorbar(color)
        for pt in self.fit_pt:
            ax.scatter(pt[0], pt[1], marker='o', s=10, c='r', zorder=100)

        plt.savefig('fitting.pdf')
        plt.show(block=False)

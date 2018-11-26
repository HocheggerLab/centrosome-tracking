import logging

import numpy as np
import pandas as pd
from lmfit import Minimizer
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from ._imageplanar import ImagePlanarElastica

logger = logging.getLogger(__name__)


class PlanarImageMinimizerIVP(ImagePlanarElastica):

    def fit(self, pt=None):
        def objective(params):
            self.parameters = params

            lineintegral, number_of_pixels = self.get_line_integral_over_image()
            max_intensity = self.tiffimage.asarray().max()

            if pt is not None:
                p = Point(pt)
                c = Polygon([(x, y) for x, y in zip(self.curve_x, self.curve_y)])
                obj = 1 / lineintegral + p.distance(c)
                # obj = p.distance(c)
            else:
                obj = max_intensity * number_of_pixels - lineintegral

            params.add('f_obj', value=obj)
            self._param_exploration.append(params.copy())

            logger.debug('objective function = {:06.4e}'.format(obj))
            return obj

        inip = self.parameters
        inip['x0'].min = inip['x0'].value - 5
        inip['x0'].max = inip['x0'].value + 5
        inip['theta'].min = inip['x0'].value * 0.9
        inip['theta'].max = inip['x0'].value * 1.1

        inip['L'].vary = False
        inip['F'].vary = False
        inip['m0'].vary = False
        inip['phi'].vary = False

        # inip['alpha'].vary = False
        inip['alpha'].min = 0
        inip['alpha'].max = 2 * np.pi
        inip['w'].min = 0
        inip['w'].max = 20
        inip['k0'].min = -1
        inip['k0'].max = 1

        logger.info(inip)

        self._param_exploration = []

        fitter = Minimizer(objective, inip)
        result = fitter.minimize(method='bgfs', params=self.parameters)
        # result = fitter.minimize(method='basinhopping', params=self.parameters,
        #                          niter=10 ** 4, niter_success=10 ** 2)

        df = pd.DataFrame([p.valuesdict() for p in self._param_exploration])
        df = df.set_index('f_obj').sort_index(ascending=False).reset_index()
        print(df)
        print(df.set_index('f_obj').sort_index().iloc[0])

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib import cm
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

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

        return result

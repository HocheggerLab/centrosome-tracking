import logging

import numpy as np
import pandas as pd
from lmfit import Minimizer

from ._imageplanar import ImagePlanarElastica

logger = logging.getLogger(__name__)


class PlanarImageMinimizerIVP(ImagePlanarElastica):

    def fit(self):
        def objective(params):
            self.parameters = params

            lineintegral, number_of_pixels = self.get_line_integral_over_image()
            max_intensity = self.tiffimage.asarray().max()
            obj = max_intensity * number_of_pixels - lineintegral

            par = {'w': self._w, 'k0': self._k0, 'alpha': self.alpha, 'f_obj': obj}
            self._param_exploration.append(par)

            logger.debug('objective function = {:06.4e}'.format(obj))
            return obj

        inip = self.parameters.copy()
        inip['x0'].min = inip['x0'].value - 5
        inip['x0'].max = inip['x0'].value + 5
        inip['theta'].min = 0
        inip['theta'].max = 2 * np.pi
        # inip['L'].min = 4
        # inip['L'].max = 20

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
        # result = fitter.minimize(method='bgfs', params=self.parameters)
        result = fitter.minimize(method='basinhopping', params=self.parameters,
                                 niter=10 ** 4, niter_success=2 * 10 ** 2)

        df = pd.DataFrame(self._param_exploration)
        df = df.set_index('f_obj').sort_index(ascending=False).reset_index()
        print(df)

        import matplotlib.pyplot as plt
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(df["w"], df["k0"], df["alpha"], c=df["f_obj"], cmap='viridis')
        # plt.colorbar()
        plt.show(block=False)

        return result

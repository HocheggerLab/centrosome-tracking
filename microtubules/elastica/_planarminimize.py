import logging

from lmfit import Minimizer

from ._imageplanar import ImagePlanarElastica

logger = logging.getLogger(__name__)


class PlanarImageMinimizer(ImagePlanarElastica):

    def fit(self):
        def objective(params):
            self.parameters = params

            lineintegral, number_of_pixels = self.get_line_integral_over_image()
            max_intensity = self.tiffimage.asarray().max
            return max_intensity * number_of_pixels - lineintegral

        inip = self.parameters.copy()
        inip['x0'].min = inip['x0'].value - 5
        inip['x0'].max = inip['x0'].value + 5
        inip['xe'].min = inip['xe'].value - 5
        inip['ye'].max = inip['ye'].value + 5
        inip['N1'].min = 0.0
        inip['N1'].max = 100 * 1e-12
        inip['N2'].min = 0.0
        inip['N2'].max = 100 * 1e-12
        inip['E'].min = 1.0e9
        inip['E'].max = 1.9e9 * 100  # assuming N=10 fibers and a N^2 scaling
        inip['L'].min = inip['L'].value
        inip['L'].max = inip['L'].value * 2.0
        inip['phi'].min = inip['phi'].value * 0.8
        inip['phi'].max = inip['phi'].value * 1.2
        inip['theta0'].min = inip['theta0'].value * 0.8
        inip['theta0'].max = inip['theta0'].value * 1.2

        fitter = Minimizer(objective, inip)
        result = fitter.minimize(method='bgfs', params=self.parameters)
        # result = fitter.minimize(method='basinhopping', params=self.parameters)

        return result

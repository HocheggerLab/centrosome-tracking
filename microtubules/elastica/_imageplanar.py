import numpy as np
import tifffile as tf

from ._planar_ivp import PlanarElasticaIVPArtist


class ImagePlanarElastica(PlanarElasticaIVPArtist):
    def __init__(self, axes,
                 w=1.0, k0=1.0, alpha=np.pi / 2,
                 L=1.0, E=0.625, J=1.0, F=1.0, x0=0.0, y0=0.0, m0=0.01, theta=3 * np.pi / 2,
                 image=None):
        if image is None or type(image) != tf.tifffile.TiffPage:
            raise Exception('No point in creating this class without an image.')

        super().__init__(axes, w=w, k0=k0, alpha=alpha, L=L, E=E, J=J, F=F, x0=x0, y0=y0, m0=m0, theta=theta)
        self.tiffimage = image

    def get_line_integral_over_image(self, res=4.5):
        img = self.tiffimage.asarray()
        min_value = np.finfo(img.dtype).min, np.finfo(img.dtype).min
        try:
            self.eval(num_points=100)
            self.plot(self.ax)
        except Exception:
            return min_value

        if self.res.status == 2:
            return min_value

        pt = np.array([self.curve_x, self.curve_y]).T
        pt *= res
        pt = np.unique(pt.astype(np.int16), axis=0)

        minx, maxx = np.amin(pt[:, 0]), np.amax(pt[:, 0])
        miny, maxy = np.amin(pt[:, 1]), np.amax(pt[:, 1])
        if minx < 0 or maxx > img.shape[0] or miny < 0 or maxy > img.shape[1]:
            return min_value

        maxrows = img.shape[1] - 1
        i = img[maxrows - pt[:, 1], pt[:, 0]].sum()  # numpy arrays follow (row, col) convention

        return i, pt.shape[0]

    def paint_line_integral_over_image(self, ax, res=4.5):
        img = self.tiffimage.asarray()

        pt = np.array([self.curve_x, self.curve_y]).T
        pt *= res
        pt = np.unique(pt.astype(np.int16), axis=0)

        maxy = img.shape[1] - 1
        img[maxy - pt[:, 1], pt[:, 0]] = 10000

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(img)

        fig = plt.figure()
        ax = fig.gca()
        ax.hist(img)

        plt.show()

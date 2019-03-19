import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf

from microtubules.elastica import Aster
from microtubules import elastica as e

np.set_printoptions(1)

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal')

    ax.set_facecolor('b')

    with tf.TiffFile(
            '/Users/Fabio/data/lab/SRRF images for Fabio/15-2-19 Best Images/U2OS CDK1as +1NM+STLC PFA in CBS SRRF composite - Capture 11.tif') as tif:
        if tif.is_imagej is not None:
            sizeT, channels = tif.imagej_metadata['images'], tif.pages[0].imagedepth
            sizeZ, sizeX, sizeY = 1, tif.pages[0].imagewidth, tif.pages[0].imagelength
            print('N of frames=%d channels=%d, sizeZ=%d, sizeX=%d, sizeY=%d' % \
                  (sizeT, channels, sizeZ, sizeX, sizeY))
            res = 4.5

            sequence = tif.asarray().reshape([sizeT, channels, sizeX, sizeY])
            # print(sequence.shape)
            img = tif.pages[0].asarray()

            ax.imshow(img, extent=(0, sizeX / res, 0, sizeY / res))
            # ax.set_xlim([0, sizeX / res])
            # ax.set_ylim([0, sizeY / res])

    F = 30 * np.sqrt(2) * 1e-3
    x0, y0 = 56.5, 60
    fiber = e.PlanarImageMinimizerIVP(ax, w=1.0, k0=-1.0, alpha=np.pi / 6,
                                      m0=0.0, x0=x0, y0=y0, L=11.0,
                                      theta=2 * np.pi / 3, image=tif.pages[0])
    # fiber = e.PlanarElasticaIVPArtist(ax, L=10.2, E=1e9, J=1e-8, F=5e-1, theta=np.pi / 3)
    centrosome = Aster(ax, x0=x0, y0=y0)
    centrosome.add_fiber(fiber)

    plt.show()

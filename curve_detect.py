import argparse

import matplotlib.pyplot as plt
import tifffile as tf
from matplotlib import cm
from skimage.feature import canny
from skimage.filters import roberts, scharr, sobel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detects curves from image.')
    parser.add_argument('input', metavar='in', type=str, help='image file')
    args = parser.parse_args()

    with tf.TiffFile(args.input, fastij=True) as tif1:
        # Generating figure 2
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
        ax = axes.ravel()
        ims = []
        for i, page in enumerate(tif1.pages):
            image = page.asarray()
            edges_canny = canny(image, sigma=0.9, low_threshold=0.1, high_threshold=0.3, use_quantiles=True)
            edge_sobel = sobel(image)
            edge_scharr = scharr(image)
            edge_roberts = roberts(image)

            # edges_canny = exposure.equalize_hist(edges_canny)
            # edge_scharr = exposure.equalize_hist(edge_scharr)
            # edge_roberts = exposure.equalize_hist(edge_roberts)
            # edge_sobel = exposure.equalize_hist(edge_sobel)

            colormap = cm.cool
            ax[0].imshow(image, cmap=cm.gray)
            ax[0].set_title('Input image')

            ax[1].imshow(edges_canny, cmap=cm.gray)
            ax[1].set_title('Canny edges')

            ax[2].imshow(edge_scharr, cmap=colormap)
            ax[2].set_title('Scharr edges')

            ax[3].imshow(edge_roberts, cmap=colormap)
            ax[3].set_title('Roberts edges')

            ax[4].imshow(edge_sobel, cmap=colormap)
            ax[4].set_title('Sobel edges')

            for a in ax:
                a.set_axis_off()
                a.set_adjustable('box-forced')

            plt.tight_layout()
            plt.show()
            plt.savefig('out-%d.tif' % (i), format='tif')

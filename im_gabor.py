import cv2
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
import skimage.segmentation
import tifffile as tf
from scipy import ndimage as ndi
from skimage import exposure
from skimage.filters import gaussian
from skimage.morphology import erosion, square
from skimage.segmentation import active_contour



def cell_boundary(tubulin, hoechst, fig=None, threshold=80, markers=None):
    def build_gabor_filters():
        filters = []
        ksize = 9
        for theta in np.arange(0, np.pi, np.pi / 8):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 6.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= kern.sum()
            filters.append(kern)
        return filters

    def process_gabor(img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_16UC1, kern)
            np.maximum(accum, fimg, accum)
        return accum

    p2 = np.percentile(tubulin, 2)
    p98 = np.percentile(tubulin, 98)
    tubulin = exposure.rescale_intensity(tubulin, in_range=(p2, p98))
    p2 = np.percentile(hoechst, 2)
    p98 = np.percentile(hoechst, 98)
    hoechst = exposure.rescale_intensity(hoechst, in_range=(p2, p98))

    img = np.maximum(tubulin, 0.8 * hoechst)

    img = erosion(img, square(3))
    filters = build_gabor_filters()
    gabor = process_gabor(img, filters)

    gabor = cv2.convertScaleAbs(gabor, alpha=(255.0 / 65535.0))
    ret, bin1 = cv2.threshold(gabor, threshold, 255, cv2.THRESH_BINARY)

    # gaussian blur on gabor filter result
    ksize = 31
    blur = cv2.GaussianBlur(bin1, (ksize, ksize), 0)
    ret, bin2 = cv2.threshold(blur, threshold, 255, cv2.THRESH_OTSU)
    # ret, bin2 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY)

    if markers is None:
        # get markers for watershed from hoescht channel
        hoechst_8 = cv2.convertScaleAbs(hoechst, alpha=(255.0 / 65535.0))
        blur_nuc = cv2.GaussianBlur(hoechst_8, (ksize, ksize), 0)
        ret, bin_nuc = cv2.threshold(blur_nuc, 0, 255, cv2.THRESH_OTSU)
        markers = ndi.label(bin_nuc)[0]

    # label cell boundaries starting from each nucleus, using the watershed algorithm
    # distance = ndi.distance_transform_edt(bin2)
    # distance = distance / np.linalg.norm(distance)
    # distance = distance / np.max(distance)
    # distance = np.logical_and(distance, distance > 0.09)
    # distance = ndi.distance_transform_edt(distance)
    # gabor_proc = distance * gabor
    # gabor_proc = cv2.GaussianBlur(gabor, (ksize, ksize), 0)
    # gabor_proc = process_gabor(gabor_proc, filters)
    gabor_proc = gabor

    labels = skimage.morphology.watershed(-gabor_proc, markers, mask=bin2)

    image = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    boundaries_list = list()
    # loop over the labels
    for (i, l) in enumerate([l for l in np.unique(labels) if l > 0]):
        # find contour of mask
        cell_boundary = np.zeros(shape=labels.shape, dtype=np.uint8)
        cell_boundary[labels == l] = 255
        cnts = cv2.findContours(cell_boundary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contour = cnts[0]

        do_snake = False
        if do_snake:
            snake = np.array([[x, y] for x, y in [i[0] for i in contour]], dtype=np.float32)
            contour = active_contour(gaussian(gabor, 3), snake, alpha=0.015, beta=0.1, gamma=0.1, w_line=-1.0)

            cx, cy = 0, 0
            boundaries_list.append({'id': l, 'boundary': contour, 'centroid': (cx, cy)})
        else:
            # draw the contour
            ((x, y), _) = cv2.minEnclosingCircle(contour)
            cv2.putText(color, '#{}'.format(i + 1), (int(x) - 10, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.drawContours(color, contour, -1, (0, 255, 0), 2)

            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(color, (cx, cy), 5, (0, 255, 0), thickness=-1)

            boundary = np.array([[x, y] for x, y in [i[0] for i in contour]], dtype=np.float32)
            boundaries_list.append({'id': l, 'boundary': boundary, 'centroid': (cx, cy)})

    if fig is not None:
        # set-up matplotlib axes
        gs = matplotlib.gridspec.GridSpec(2, 3)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])
        ax4 = plt.subplot(gs[1, 0])
        ax5 = plt.subplot(gs[1, 1])
        ax6 = plt.subplot(gs[1, 2])
        plt.tight_layout()

        # show the output image
        ax1.imshow(img)
        ax2.imshow(gabor)
        ax3.imshow(bin2)

        ax4.imshow(markers)
        ax5.imshow(labels)
        ax6.imshow(color)

        ax1.set_title('original')
        ax2.set_title('gabor filter')
        ax3.set_title('threshold')
        ax4.set_title('nuclei seeds')
        ax5.set_title('cell boundary')
        ax6.set_title('render')

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.pause(0.5)

    return boundaries_list, gabor_proc


if __name__ == '__main__':
    with tf.TiffFile('/Users/Fabio/data/lab/pc-100.tif', fastij=True) as tif:
        if tif.is_imagej is not None:
            sizeT, channels = tif.pages[0].imagej_tags.frames, tif.pages[0].imagej_tags.channels
            sizeZ, sizeX, sizeY = 1, tif.pages[0].image_width, tif.pages[0].image_length
            print 'N of frames=%d channels=%d, sizeZ=%d, sizeX=%d, sizeY=%d' % \
                  (sizeT, channels, sizeZ, sizeX, sizeY)

            res = 'n/a'
            if tif.pages[0].resolution_unit == 'centimeter':
                # asuming square pixels
                xr = tif.pages[0].x_resolution
                res = float(xr[0]) / float(xr[1])  # pixels per cm
                res = res / 1e4  # pixels per um
            elif tif.pages[0].imagej_tags.unit == 'micron':
                # asuming square pixels
                xr = tif.pages[0].x_resolution
                res = float(xr[0]) / float(xr[1])  # pixels per um

            sequence = tif.pages[0].asarray().reshape([sizeT, channels, sizeX, sizeY])
            for img in sequence:
                ch1 = img[0]
                ch2 = img[1]
                ch3 = img[2]
                tubulin = ch2
                hoescht = ch1
                try:
                    cell_boundary(tubulin, hoescht, fig=plt.gcf())
                except:
                    pass

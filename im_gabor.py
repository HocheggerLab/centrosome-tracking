import cv2
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
from scipy import ndimage as ndi


def build_filters():
    filters = []
    ksize = 51
    for theta in np.arange(0, np.pi, np.pi / 32):
        kern = cv2.getGaborKernel((ksize, ksize), 3.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_16UC1, kern)
        np.maximum(accum, fimg, accum)
    return accum


if __name__ == '__main__':
    file = '/Users/Fabio/data/lab/test-gabor.tif'
    file2 = '/Users/Fabio/data/lab/test-gabor-2.tif'
    tubulin = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    hoescht = cv2.imread(file2, cv2.IMREAD_UNCHANGED)
    if tubulin is None or hoescht is None:
        print 'Failed to load image file:', file

    img = tubulin + hoescht

    # set-up matplotlib axes
    gs = matplotlib.gridspec.GridSpec(2, 4)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[0, 3])
    ax5 = plt.subplot(gs[1, 0])
    ax6 = plt.subplot(gs[1, 1])
    ax7 = plt.subplot(gs[1, 2])
    ax8 = plt.subplot(gs[1, 3])
    plt.tight_layout()

    thr_ = 50
    filters = build_filters()
    gabor = process(img, filters)
    gabor = cv2.convertScaleAbs(gabor, alpha=(255.0 / 65535.0))
    ret, bin1 = cv2.threshold(gabor, thr_, 255, cv2.THRESH_BINARY)

    # gaussian blur on gabor filter result
    ksize = 31
    blur = cv2.GaussianBlur(bin1, (ksize, ksize), 0)
    ret, bin2 = cv2.threshold(blur, thr_, 255, cv2.THRESH_OTSU)

    # get markers for watershed from hoescht channel
    hoescht_8 = cv2.convertScaleAbs(hoescht, alpha=(255.0 / 65535.0))
    blur_nuc = cv2.GaussianBlur(hoescht_8, (ksize, ksize), 0)
    ret, bin_nuc = cv2.threshold(blur_nuc, 0, 255, cv2.THRESH_OTSU)

    # label cell boundaries starting from each nucleus, using the watershed algorithm
    distance = ndi.distance_transform_edt(bin2)
    markers = ndi.label(bin_nuc)[0]
    labels = skimage.morphology.watershed(-distance, markers, mask=bin2)

    image = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # loop over the labels
    for (i, l) in enumerate([l for l in np.unique(labels) if l > 0]):
        # find contour of mask
        cell_boundary = np.zeros(shape=labels.shape, dtype=np.uint8)
        cell_boundary[labels == l] = 255
        cnts = cv2.findContours(cell_boundary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contour = cnts[0]

        # draw the contour
        ((x, y), _) = cv2.minEnclosingCircle(contour)
        cv2.putText(color, '#{}'.format(i + 1), (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.drawContours(color, [contour], -1, (0, 255, 0), 2)

        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(color, (cx, cy), 5, (0, 255, 0), thickness=-1)

        # compute convex hull
        hull = cv2.convexHull(contour)
        cv2.drawContours(color, [hull], -1, (255, 0, 0), 2)

        M = cv2.moments(hull)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(color, (cx, cy), 5, (255, 0, 0), thickness=-1)

    # show the output image
    ax1.imshow(gabor)
    ax2.imshow(blur)
    ax3.imshow(bin2)
    ax4.imshow(distance)

    ax5.imshow(bin_nuc)
    ax6.imshow(markers)
    ax7.imshow(labels)
    ax8.imshow(color)

    ax7.set_title('cell boundary')

    # for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

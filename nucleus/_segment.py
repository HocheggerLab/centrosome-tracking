import math

import cv2
import pandas as pd
import scipy.ndimage as ndi
import skimage.feature as feature
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology
import skimage.segmentation as segmentation
import skimage.transform as tf
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
from scipy.ndimage.morphology import distance_transform_edt

from ._common import _DEBUG, logger


def segment(image, radius=10):
    # apply threshold
    thresh_val = filters.threshold_otsu(image)
    thresh = image >= thresh_val
    thresh = morphology.remove_small_holes(thresh)
    thresh = morphology.remove_small_objects(thresh)

    # remove artifacts connected to image border
    cleared = segmentation.clear_border(thresh)

    if len(cleared[cleared > 0]) == 0: return None, None

    distance = distance_transform_edt(cleared)
    local_maxi = feature.peak_local_max(distance, indices=False, labels=cleared,
                                        min_distance=radius / 4, exclude_border=False)
    markers, num_features = ndi.label(local_maxi)
    if num_features == 0:
        logger.info('no nuclei found for current stack')
        return None, None

    labels = morphology.watershed(-distance, markers, watershed_line=True, mask=cleared)

    # store all contours found
    contours = measure.find_contours(labels, 0.9)
    tform = tf.SimilarityTransform(rotation=math.pi / 2, scale=1.0)

    _list = list()
    for k, contr in enumerate(contours):
        contr = tform(contr)
        contr[:, 0] *= -1
        bnd = Polygon(contr)
        _list.append({
            'id': k,
            'boundary': bnd,
            # 'x': bnd.centroid.x,
            # 'y': bnd.centroid.y
        })

    return labels, _list


def keypoint_surf(image_it):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    _im = image_it.__next__()
    # _im = cv2.equalizeHist(_im)
    _im = clahe.apply(_im)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # BFMatcher with default params
    bf = cv2.BFMatcher()

    dfmatch = pd.DataFrame()

    for fr, im in enumerate(image_it):
        if _DEBUG and fr > 5: break

        # im = cv2.equalizeHist(im)
        im = clahe.apply(im)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(_im, None)
        kp2, des2 = sift.detectAndCompute(im, None)

        #  match descriptors using Brute-Force Matching
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test and add match to dataframe is successful
        # good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                # good.append([m])
                img2_idx = m.trainIdx
                img1_idx = m.queryIdx
                (x0, y0) = kp1[img1_idx].pt
                (x1, y1) = kp2[img2_idx].pt

                d = pd.DataFrame(data={
                    'frame': [fr + 1],
                    'pt0': [Point(x0, y0)],
                    # 'x0': [x0],
                    # 'y0': [y0],
                    'a0': [kp1[img1_idx].angle],
                    'pt1': [Point(x1, y1)],
                    # 'x1': [x1],
                    # 'y1': [y1],
                    'a1': [kp2[img2_idx].angle]
                })
                dfmatch = dfmatch.append(d, ignore_index=True, sort=False)

        # # cv2.drawMatchesKnn expects list of lists as matches.
        # import matplotlib.pyplot as plt
        # img3 = cv2.drawMatchesKnn(_im, kp1, im, kp2, good, None, flags=2)
        # plt.imshow(img3), plt.show()

        _im = im
    return dfmatch

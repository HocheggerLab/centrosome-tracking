import math
import itertools

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import skimage.exposure as exposure
import skimage.feature as feature
import skimage.filters as filters
import skimage.measure as measure
import skimage.morphology as morphology
import skimage.segmentation as segmentation
import skimage.transform as tf
from skimage import img_as_ubyte
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
from scipy.ndimage.morphology import distance_transform_edt

from ._common import _DEBUG, logger


def exclude_contained(polygons):
    if polygons is None: return []
    for p in polygons:
        p['valid'] = True
    for p1, p2 in itertools.combinations(polygons, 2):
        if not p1['valid'] or not p2['valid']: continue
        if p1['boundary'].contains(p2['boundary']):
            p2['valid'] = False
        if p2['boundary'].contains(p1['boundary']):
            p1['valid'] = False
    polygons = [p for p in polygons if p['valid']]
    for p in polygons:
        del p['valid']
    return polygons


# noinspection PyTypeChecker
def segment(image, radius=10):
    image = cv2.GaussianBlur(image, (int(radius), int(radius)), 2)
    image = exposure.rescale_intensity(image)

    # apply threshold
    thresh_val = filters.threshold_otsu(image)
    thresh = image >= thresh_val
    thresh = morphology.remove_small_holes(thresh)
    thresh = morphology.remove_small_objects(thresh, min_size=np.pi * radius ** 2 / 2)

    # remove artifacts connected to image border
    cleared = segmentation.clear_border(thresh)

    if len(cleared[cleared > 0]) == 0: return None, None

    distance = distance_transform_edt(cleared)
    local_maxi = feature.peak_local_max(distance, indices=False, labels=cleared,
                                        min_distance=radius / 6, exclude_border=False)
    markers, num_features = ndi.label(local_maxi)
    if num_features == 0:
        logger.info('no nuclei found for current stack')
        return None, None

    distance = exposure.rescale_intensity(distance)
    distance = cv2.GaussianBlur(distance, (int(radius), int(radius)), 2)
    distance = exposure.rescale_intensity(distance)
    # plt.imshow(distance), plt.show()

    thresh_val = filters.threshold_otsu(distance)
    thresh = distance >= thresh_val
    thresh = morphology.remove_small_holes(thresh)
    thresh = morphology.remove_small_objects(thresh)
    thresh = img_as_ubyte(thresh)

    labels = morphology.watershed(-thresh, markers, watershed_line=True, mask=cleared)

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
            'x': bnd.centroid.x,
            'y': bnd.centroid.y
        })

    return labels, exclude_contained(_list)


def keypoint_surf_match(image_it):
    _im = image_it.__next__()
    _im = exposure.rescale_intensity(_im)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # BFMatcher with default params
    bf = cv2.BFMatcher()

    matches = dict()
    mkey = 0
    for fr, im in enumerate(image_it):
        if _DEBUG and fr > 5: break
        im = exposure.rescale_intensity(im)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(_im, None)
        kp2, des2 = sift.detectAndCompute(im, None)

        #  match descriptors using Brute-Force Matching
        if des1 is None or des2 is None: continue
        knnmatches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test and add match to dataframe is successful
        # good = []
        for _m, n in knnmatches:
            if _m.distance < 0.75 * n.distance:
                # good.append([m])
                img2_idx = _m.trainIdx
                img1_idx = _m.queryIdx
                (x0, y0) = kp1[img1_idx].pt
                (x1, y1) = kp2[img2_idx].pt

                pt0 = Point(x0, y0)
                found = False
                for _mk, md in matches.items():
                    if pt0 == md['pts'][-1]:
                        found = True
                        matches[_mk]['pts'].append(Point(x1, y1))
                        matches[_mk]['frames'].append(fr + 1)
                if not found:
                    matches[mkey] = {'pts': [], 'frames': []}
                    matches[mkey]['frames'].append(fr)
                    matches[mkey]['pts'].append(pt0)
                    matches[mkey]['pts'].append(Point(x1, y1))
                    matches[mkey]['frames'].append(fr + 1)
                    mkey += 1

        # # cv2.drawMatchesKnn expects list of lists as matches.
        # import matplotlib.pyplot as plt
        # img3 = cv2.drawMatchesKnn(_im, kp1, im, kp2, good, None, flags=2)
        # plt.imshow(img3), plt.show()
        _im = im

    dfmatch = pd.DataFrame()
    for mkey, md in matches.items():
        y0 = pd.DataFrame(data={
            'frame': [fr for fr in md['frames']],
            'pt': [pt for pt in md['pts']],
            'x': [pt.x for pt in md['pts']],
            'y': [pt.y for pt in md['pts']],
            'particle': [mkey] * len(md['frames']),
        })
        dfmatch = dfmatch.append(y0, ignore_index=True, sort=False)
    return dfmatch


def optical_flow_lk_match(image_it):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=200,
                          qualityLevel=0.005,
                          minDistance=7,
                          blockSize=15)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=0,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 4, 0.03))

    # Take first frame and find corners in it
    old_gray = image_it.__next__()
    old_gray = exposure.rescale_intensity(old_gray)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # print(p0.shape)
    # plt.imshow(old_gray)
    # for i, new in enumerate(p0):
    #     x1, y1 = new.ravel()
    #     plt.plot(x1, y1, marker='o', markersize=2, color='red')
    # plt.show()

    # Create a mask image for drawing purposes
    # mask = np.zeros_like(old_gray)

    matches = dict()
    mkey = 0
    for fr, gray in enumerate(image_it):
        if _DEBUG and fr > 5: break
        gray = exposure.rescale_intensity(gray)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        # plt.imshow(gray)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            x1, y1 = new.ravel()
            x0, y0 = old.ravel()
            # plt.plot((x1, x0), (y1, y0), color=color[i])
            # plt.plot(x1, y1, marker='o', markersize=5, color='green')
            # plt.plot(x0, y0, marker='+', markersize=5, color='blue')

            # search old point in lists
            pt0 = Point(x0, y0)
            found = False
            for _mk, md in matches.items():
                if pt0 == md['pts'][-1]:
                    found = True
                    matches[_mk]['pts'].append(Point(x1, y1))
                    matches[_mk]['frames'].append(fr + 1)
            if not found:
                matches[mkey] = {'pts': [], 'frames': []}
                matches[mkey]['frames'].append(fr)
                matches[mkey]['pts'].append(pt0)
                matches[mkey]['pts'].append(Point(x1, y1))
                matches[mkey]['frames'].append(fr + 1)
                mkey += 1

        # plt.show()

        # Now update the previous frame and previous points
        old_gray = gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    dfmatch = pd.DataFrame()
    for mkey, md in matches.items():
        y0 = pd.DataFrame(data={
            'frame': [fr for fr in md['frames']],
            'pt': [pt for pt in md['pts']],
            'x': [pt.x for pt in md['pts']],
            'y': [pt.y for pt in md['pts']],
            'particle': [mkey] * len(md['frames']),
        })
        dfmatch = dfmatch.append(y0, ignore_index=True, sort=False)

    return dfmatch

import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import shapely.geometry
from shapely.wkt import dumps

import plots
import tools.image as image
from tools.interactor import PolygonInteractor
from tools.draggable import DraggableCircle


def write_centrosome_angle_csv(fname, nucleus, centrosomes):
    nuc = shapely.geometry.Polygon(nucleus.poly.xy)
    c1 = shapely.geometry.Point(centrosomes[0])
    c2 = shapely.geometry.Point(centrosomes[1])
    cn = nuc.centroid
    angle = math.atan2(c1.y - cn.y, c1.x - cn.x) - math.atan2(c2.y - cn.y, c2.x - cn.x)

    df = pd.DataFrame(data={'file': [os.path.basename(fname)[-4]],
                            'c1': dumps(c1, rounding_precision=1),
                            'c2': dumps(c2, rounding_precision=1),
                            'nucleus': dumps(nuc, rounding_precision=1),
                            'angle_rad': [angle],
                            'angle_deg': [math.degrees(angle)],
                            })
    df.to_csv(fname, index=False)


def pick_centrosomes_and_save(file):
    imgs, pix_per_um, dt, n_frames, n_channels, series = image.load_tiff(file)
    img = image.retrieve_from_pageseries(series[0], n_frames - 1)
    n_ztacks, _, sizeX, sizeY = img.shape
    # z max project and split images into channels
    tub, act, pact = np.nanmax(img, axis=0)

    print(tub.shape, pact.shape)
    ext = (0, sizeX / pix_per_um, 0, sizeY / pix_per_um)
    ax.imshow(pact, extent=ext, cmap='uscope_magenta', interpolation='none')
    ax.imshow(tub, extent=ext, cmap='uscope_green', alpha=0.9, interpolation='none')

    theta = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r = 15
    xs = r * np.cos(theta) + ext[1] / 2
    ys = r * np.sin(theta) + ext[3] / 2
    poly = Polygon(np.column_stack([xs, ys]), alpha=0.3, lw=0.1, ls='--', animated=True)

    ax.add_patch(poly)
    nucleus = PolygonInteractor(ax, poly)

    ci1 = DraggableCircle(plt.Circle(xy=(ext[1] / 2 - 10, ext[3] / 2 + 10), radius=1, fc='r', picker=5))
    ci2 = DraggableCircle(plt.Circle(xy=(ext[1] / 2 + 10, ext[3] / 2 - 10), radius=1, fc='b', picker=5))
    ax.add_artist(ci1.circle)
    ax.add_artist(ci2.circle)
    ci1.connect()
    ci2.connect()

    ax.set_title('Click and drag a point to move it')
    plt.show()

    csv_filename = os.path.join(os.path.dirname(file), os.path.basename(file) + '.csv')
    asters = [ci1.circle.center, ci2.circle.center]
    write_centrosome_angle_csv(csv_filename, nucleus, asters)


if __name__ == '__main__':
    fig = plt.figure(dpi=200)
    ax = fig.gca()
    ax.set_aspect('equal')

    file = 'S03 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 3 partial sep.tif'
    file = 'S04 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 4.tif'
    file = 'S05 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 5.tif'
    file = 'S06 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 6.tif'
    file = 'S06 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 6.2.tif'
    file = 'S07 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 7.tif'
    file = 'S07 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 7.2.tif'
    file = 'S08 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 8.tif'
    file = 'S08 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 8.2.tif'
    file = 'S08 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 8.3.tif'
    file = 'S08 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 8.4.tif'
    file = 'S09 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 9.tif'
    file = 'S11 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 11.tif'
    file = 'S23 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 4 (1).tif'
    file = 'S23 U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 4 (1).2.tif'
    file = '/Volumes/Kidbeat/data/lab/centrosomes-fhod1/20180515/' + file

    file = 'U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 6.tif'
    file = 'U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 6.2.tif'
    file = 'U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 8.tif'
    file = 'U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 15.tif'
    file = 'U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 15.2.tif'
    file = 'U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 19.tif'
    file = 'U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 24.tif'
    file = 'U2OS CDK1as a-tub PACT +1NM +sir-act +FHOD1 +STLC on, washout STLC.sld - Capture 1 - Position 28.tif'
    file = '/Volumes/Kidbeat/data/lab/centrosomes-fhod1/20180516/' + file
    pick_centrosomes_and_save(file)

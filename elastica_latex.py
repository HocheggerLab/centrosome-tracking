# !/opt/usr/local/python3
import logging
import math
from functools import partial

import latexipy as lp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
from pathlib import Path

import elastica as e

logging.basicConfig(level=logging.INFO)

DIRECTORY = Path(__file__).parent / 'img'
np.set_printoptions(3, suppress=True)

lp.latexify()

winered = '#C01F56'
mycyan = '#3DCCFB'
mygray = '#808080'


# functions for plotting angle in matplotlib
def get_angle_text(angle_plot, text=None):
    if text is None:
        angle = angle_plot.get_label()[:-1]  # Excluding the degree symbol
        angle = "%0.2f" % float(angle) + u"\u00b0"  # Display angle upto 2 decimal places
    else:
        angle = text

    # Get the vertices of the angle arc
    vertices = angle_plot.get_verts()

    # Get the midpoint of the arc extremes
    x_width = (vertices[0][0] + vertices[-1][0]) / 2.0
    y_width = (vertices[0][1] + vertices[-1][1]) / 2.0

    print ('%0.2f,%0.2f' % (x_width, y_width))

    separation_radius = max(x_width / 2.0, y_width / 2.0)

    return [x_width + separation_radius, y_width + separation_radius, angle]


def get_angle_plot(line1, line2, offset=1, color=None, origin=[0, 0], len_x_axis=1, len_y_axis=1):
    l1xy = line1.get_xydata()

    # Angle between line1 and x-axis
    slope1 = (l1xy[1][1] - l1xy[0][1]) / float(l1xy[1][0] - l1xy[0][0])
    angle1 = abs(math.degrees(math.atan(slope1)))  # Taking only the positive angle

    l2xy = line2.get_xydata()

    # Angle between line2 and x-axis
    slope2 = (l2xy[1][1] - l2xy[0][1]) / float(l2xy[1][0] - l2xy[0][0])
    angle2 = abs(math.degrees(math.atan(slope2)))

    theta1 = min(angle1, angle2)
    theta2 = max(angle1, angle2)

    angle = theta2 - theta1

    if color is None:
        color = line1.get_color()  # Uses the color of line 1 if color parameter is not passed.

    return Arc(origin, len_x_axis * offset, len_y_axis * offset, 0, theta1, theta2, color=color,
               label=str(angle) + u"\u00b0")


def drawLine2P(x, y, ax=plt.gca()):
    xlims = ax.get_xlim()
    xrange = np.arange(xlims[0], xlims[1], 0.1)
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y)[0]
    yrange = k * xrange + b
    # ax.plot(xrange, k * xrange + b, lw=1, c=mygray, ls='--', zorder=1)
    line = Line2D([xrange[0], xrange[-1]], [yrange[0], yrange[-1]], lw=1, linestyle='--', c=mygray, zorder=1)
    return line


def generate_figures(suffix, figure=lp.figure):
    with figure('elastica' + suffix):
        # create some test data
        L, a1, a2, E, F, gamma, x0, y0, theta = 10.0, 0.1, 0.6, 1.0, 0.1, np.pi / 8, 200, 250, np.pi / 16
        Np = 100
        s = np.linspace(0, L, Np)
        r = e.heavy_planar_bvp(s, F=F, E=E, gamma=gamma)
        pol = r.sol
        xo = pol(s)[0:2, :]
        ys = e.eval_heavy_planar(s, pol, a1, a2)[0:2, :]
        yn = e.gen_test_data(a1, a2, L, E, F, gamma, x0, y0, theta, Np)

        # deal with rotations and translations
        sinth, costh = np.sin(theta), np.cos(theta)
        M = np.array([[costh, -sinth], [sinth, costh]])
        ys = np.matmul(M, ys) + np.array([x0, y0]).reshape((2, 1))
        xo = np.matmul(M, xo) + np.array([x0, y0]).reshape((2, 1))

        ax = plt.gca()
        ax.plot(xo[0], xo[1], lw=1, c=winered, zorder=2)
        ax.plot(ys[0], ys[1], lw=2, c=mycyan, zorder=2)
        ax.scatter(yn[0], yn[1], c=mycyan, marker='+', lw=0.1, zorder=3)
        # ax.plot(xo[0, [0, -1]], xo[1, [0, -1]], lw=1, c=mygray, ls='--', zorder=1)
        ax.scatter(xo[0, -1], xo[1, -1], marker='o', c=mygray, zorder=4)

        xlims = ax.get_xlim()
        line_1 = drawLine2P(xo[0, [0, -1]], xo[1, [0, -1]])
        line_2 = drawLine2P([xlims[0], xlims[1]], [xo[1, -1], xo[1, -1]])
        angle_plot = get_angle_plot(line_1, line_2, 1, origin=[xo[0, 0], xo[1, 0]])
        angle_text = get_angle_text(angle_plot)
        ax.add_line(line_1)
        ax.add_line(line_2)
        ax.add_patch(angle_plot)  # To display the angle arc
        ax.text(*angle_text)  # To display the angle value

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_aspect('equal', 'datalim')


if __name__ == '__main__':
    # If you are repeating arguments, you might want to register a partial.
    figure = partial(lp.figure, directory=DIRECTORY)

    # You can generate figures without calling lp.latexify().
    generate_figures('_latex', figure)

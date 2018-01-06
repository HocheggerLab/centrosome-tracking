import ConfigParser
import json
import os

import numpy as np
import pandas as pd
from PyQt4 import QtGui
from PyQt4.QtCore import QPoint, QRect, QSize, Qt
from PyQt4.QtGui import QBrush, QColor, QPainter, QPen, QRubberBand

import matlab_eb3 as eb3

# profile decorator support when not profiling
try:
    profile
except NameError:
    profile = lambda x: x


class Eb3ImageQLabel(QtGui.QLabel):
    def __init__(self, parent, exp_folder=None):
        QtGui.QLabel.__init__(self, parent)
        self.selected = True
        self.exp_folder = exp_folder
        self.dataHasChanged = False
        self.condition = None
        self.run = None
        self.frame = None
        self.resolution = None
        self.image_pixmap = None
        self.dwidth = 0
        self.dheight = 0
        self._bkgimg = None
        self._bkgres = None
        self._bkgdt = None
        self.df = None
        self.ix = None
        self.selected = set()
        self.fname = '/Users/Fabio/data/lab/eb3trk.selec.txt'

        if not os.path.isfile(self.fname):
            with open(self.fname, 'w') as configfile:
                config = ConfigParser.RawConfigParser()
                config.add_section('General')
                config.set('General', 'Version', 'v0.1')
                config.set('General', 'Mode', 'Manual selection of Eb3 track data.')
                config.write(configfile)

        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()

        self.clear()

    def clear(self):
        imgarr = np.zeros(shape=(512, 512), dtype=np.uint32)
        qtimage = QtGui.QImage(imgarr.data, imgarr.shape[1], imgarr.shape[0], imgarr.strides[0],
                               QtGui.QImage.Format_RGB32)
        self.image_pixmap = QtGui.QPixmap(qtimage)
        self.setPixmap(self.image_pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = QPoint(event.pos())
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()

    def mouseMoveEvent(self, event):
        if not self.origin.isNull():
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            modifiers = QtGui.QApplication.keyboardModifiers()
            self.rubberBand.hide()
            if self.df is not None:
                rect = QRect(self.origin, event.pos()).normalized()
                x = rect.x() / self.resolution
                y = rect.y() / self.resolution
                w = rect.width() / self.resolution
                h = rect.height() / self.resolution

                # To establish whether a given point(Xtest, Ytest) overlaps
                # with a rectangle (XBL, YBL, XTR, YTR) by testing both:
                # Xtest >= XBL & & Xtest <= XTR
                # Ytest >= YBL & & Ytest <= YTR
                df = self.df[self.ix]
                ixx = (x <= df['x']) & (df['x'] <= x + w)
                ixy = (y <= df['y']) & (df['y'] <= y + h)
                if modifiers == Qt.ControlModifier:
                    for i in df[ixx & ixy]['trk'].unique():
                        if i in self.selected:
                            self.selected.remove(int(i))
                else:
                    for i in df[ixx & ixy]['trk'].unique():
                        self.selected.add(int(i))
                self.save_selection()
                self.draw_measurements()
                self.setPixmap(self.image_pixmap)
        # self.emit(QtCore.SIGNAL('clicked()'))

    def save_selection(self):
        with open(self.fname, 'r') as configfile:
            config = ConfigParser.ConfigParser()
            config.readfp(configfile)

        with open(self.fname, 'w') as configfile:
            section = self.run
            if not config.has_section(section):
                config.add_section(section)
            config.set(section, 'selected', json.dumps(list(self.selected)))
            config.write(configfile)

    def paintEvent(self, event):
        if self.dataHasChanged:
            self.dataHasChanged = False

            bkg = self._bkgimg
            qtimage = QtGui.QImage(bkg.repeat(4), bkg.shape[1], bkg.shape[0], QtGui.QImage.Format_RGB32)
            self.image_pixmap = QtGui.QPixmap(qtimage)
            # self.image_pixmap = self.image_pixmap.transformed(QtGui.QTransform().rotate(180))
            self.draw_measurements()
            self.setPixmap(self.image_pixmap)
        return QtGui.QLabel.paintEvent(self, event)

    def render_frame(self, condition, run, frame):
        if self.condition != condition or self.run != run:
            self.condition, self.frame = condition, frame
            self.run = run[0:-4] if run[-4:] == '.tif' else run
            self.dataHasChanged = True
            img, res, dt = eb3.find_image(run, os.path.join(self.exp_folder, self.condition))
            self.dwidth, self.dheight, _ = img.shape
            # print img.shap8e
            # map the data range to 0 - 255
            img_8bit = ((img - img.min()) / (img.ptp() / 255.0)).astype(np.uint8)
            self._bkgimg = img_8bit[:, :, 0]
            self.resolution = res
            self._bkgdt = dt

            self.df = pd.read_pickle('/Users/Fabio/data/lab/eb3filter.pandas')
            self.ix = (self.df['condition'] == self.condition) & (self.df['tag'] == self.run)

            # get selection from disk
            with open(self.fname, 'r') as configfile:
                config = ConfigParser.ConfigParser()
                config.readfp(configfile)
                if config.has_section(self.run):
                    self.selected = set(json.loads(config.get(self.run, 'selected')))
                else:
                    self.selected = set()

            self.repaint()

    @profile
    def draw_measurements(self):
        df = self.df[self.ix]

        painter = QPainter()
        painter.begin(self.image_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        for id, dft in df.groupby('trk'):
            if id in self.selected:
                painter.setPen(QPen(QBrush(QColor(0, 255, 0, 125)), 2))
                painter.setBrush(QColor('green'))
            else:
                painter.setPen(QPen(QBrush(QColor(255, 255, 0, 125)), 2))
                painter.setBrush(QColor('yellow'))

            x = dft['x'].values * self.resolution
            y = dft['y'].values * self.resolution
            for i in range(len(dft) - 1):
                x0, y0, x1, y1 = x[i], y[i], x[i + 1], y[i + 1]
                painter.drawLine(x0, y0, x1, y1)

        painter.end()

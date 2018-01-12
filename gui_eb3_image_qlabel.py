import ConfigParser
import json
import logging
import os

import numpy as np
import pandas as pd
from PyQt4 import QtGui
from PyQt4.QtCore import QPoint, QRect, QSize, Qt
from PyQt4.QtGui import QBrush, QColor, QPainter, QPen, QRubberBand

import plot_special_tools as sp

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
        self.measurements_pixmap = None
        self.dwidth = 0
        self.dheight = 0
        self.bkg_pixmap = None
        self._bkgres = None
        self._bkgdt = None
        self.df = None
        self.ix = None
        self.selected = set()
        self.fname = '/Users/Fabio/data/lab/eb3trk.selec.txt'
        self.cal = pd.read_excel('/Users/Fabio/data/lab/eb3/eb3_calibration.xls')

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
        self.image_pixmap.fill(QColor(0, 0, 255, 0))
        if self.measurements_pixmap is not None:
            self.measurements_pixmap.fill(QColor(0, 0, 0, 0))
            self.bkg_pixmap.fill(QColor(0, 0, 255, 0))
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
                x = rect.x()
                y = rect.y()
                w = rect.width()
                h = rect.height()

                # To establish whether a given point(Xtest, Ytest) overlaps
                # with a rectangle (XBL, YBL, XTR, YTR) by testing both:
                # Xtest >= XBL & & Xtest <= XTR
                # Ytest >= YBL & & Ytest <= YTR
                ixx = (x <= self.df['x']) & (self.df['x'] <= x + w)
                ixy = (y <= self.df['y']) & (self.df['y'] <= y + h)
                if modifiers == Qt.ControlModifier:
                    for i in self.df[ixx & ixy]['particle'].unique():
                        if i in self.selected:
                            self.selected.remove(int(i))
                else:
                    for i in self.df[ixx & ixy]['particle'].unique():
                        self.selected.add(int(i))
                self.save_selection()
                self.measurements_pixmap = QtGui.QPixmap(self.dwidth, self.dheight)
                self.measurements_pixmap.fill(QColor(0, 0, 0, 0))
                self.draw_measurements()
                self.setPixmap(self.image_pixmap)

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

            painter = QPainter()
            painter.begin(self.image_pixmap)
            painter.drawImage(0, 0, self.bkg_pixmap.toImage())
            painter.drawImage(0, 0, self.measurements_pixmap.toImage())
            painter.end()

            self.setPixmap(self.image_pixmap)
        return QtGui.QLabel.paintEvent(self, event)

    def render_frame(self, condition, run, frame):
        if self.frame != frame:
            self.frame = frame
            self.dataHasChanged = True

            img, res, dt = sp.find_image(run, os.path.join(self.exp_folder, condition))
            img = img[frame]
            self.dwidth, self.dheight = img.shape

            # map the data range to 0 - 255
            bkg = ((img - img.min()) / (img.ptp() / 255.0)).astype(np.uint8)
            self.bkg_pixmap = QtGui.QImage(bkg.repeat(4), bkg.shape[1], bkg.shape[0], QtGui.QImage.Format_RGB32)
            self.bkg_pixmap = QtGui.QPixmap(self.bkg_pixmap)
            self.resolution = res
            self._bkgdt = dt

        calp = self.cal[self.cal['filename'] == run].iloc[0]
        run = run[0:-4] if run[-4:] == '.tif' else run
        if self.condition != condition or self.run != run:
            logging.debug('condition or run changed. condition=%s\r\nrun=%s' % (self.condition, self.run))
            self.condition, self.run = condition, run
            self.dataHasChanged = True

            self.df = pd.read_pickle('/Users/Fabio/data/lab/eb3filter.pandas')
            ix = (self.df['condition'] == self.condition) & (self.df['tag'] == self.run)
            self.df = self.df[ix]
            # consider 1.6X magification of optivar system
            if calp['optivar'] == 'yes':
                logging.info('file with optivar configuration selected!')
                self.df['x'] *= 1.6
                self.df['y'] *= 1.6
            self.df['x'] *= self.resolution
            self.df['y'] *= self.resolution

            self.dfg = self.df.groupby('particle')

            # get selection from disk
            with open(self.fname, 'r') as configfile:
                config = ConfigParser.ConfigParser()
                config.readfp(configfile)
                if config.has_section(self.run):
                    self.selected = set(json.loads(config.get(self.run, 'selected')))
                else:
                    self.selected = set()

            self.measurements_pixmap = QtGui.QPixmap(self.dwidth, self.dheight)
            self.measurements_pixmap.fill(QColor(0, 0, 0, 0))
            self.draw_measurements()

        self.repaint()

    @profile
    def draw_measurements(self):

        painter = QPainter()
        painter.begin(self.measurements_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setPen(QPen(QBrush(QColor('white')), 2))
        painter.drawText(10, 30, '%02d - (%03d,%03d)' % (self.frame, self.dwidth, self.dheight))

        for id, g in self.dfg:
            if id in self.selected:
                painter.setPen(QPen(QBrush(QColor(0, 255, 0, 125)), 2))
            else:
                painter.setPen(QPen(QBrush(QColor(255, 0, 0, 30)), 2))

            x = g['x'].values
            y = g['y'].values
            for i in range(len(g) - 1):
                x0, y0, x1, y1 = x[i], y[i], x[i + 1], y[i + 1]
                painter.drawLine(x0, y0, x1, y1)

        painter.end()

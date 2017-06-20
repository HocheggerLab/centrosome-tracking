import os
import re
import h5py
# import cv2
import skimage
import skimage.color
import skimage.exposure
import skimage.io

import pandas as pd
import numpy as np
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QImage, QPixmap, QLabel, QPainter, QBrush, QPen, QColor


class RenderedCentrosomeQLabel(QtGui.QLabel):
    def __init__(self, parent, df=None):
        QtGui.QLabel.__init__(self, parent)
        self.selected = True
        self._df = df

    def mouseReleaseEvent(self, ev):
        self.emit(QtCore.SIGNAL('clicked()'))

    def paintEvent(self, event):
        QtGui.QLabel.paintEvent(self, event)
        painter = QtGui.QPainter(self)
        if not self.selected:
            painter.setOpacity(0.5)
            ovrl = QtGui.QPixmap(300, 300)
            ovrl.fill(QtGui.QColor(255, 0, 0))
            painter.drawPixmap(0, 0, ovrl)
        if self._df is not None:
            painter.setPen(QtGui.QColor(255, 255, 255))
            painter.setOpacity(1.0)
            painter.drawText(130, 0, 'circ: %0.2f' % self._df['circ'].values[0])
            painter.drawText(130, 15, 'signal: %0.2f' % self._df['sig1_mean'].values[0])
            painter.drawText(130, 30, 'sig2: %0.2f' % self._df['sig2_mean'].values[0])
            painter.drawText(130, 45, 'sig2 outer ring: %0.2f' % self._df['sig2_out_mean'].values[0])


class ExperimentsList(QtGui.QWidget):
    def __init__(self, path):
        QtGui.QWidget.__init__(self)
        uic.loadUi('ExperimentsSelectionWidget.ui', self)

        imgarr = np.zeros(shape=(512, 512), dtype=np.uint32)
        qtimage = QtGui.QImage(imgarr.data, imgarr.shape[1], imgarr.shape[0], imgarr.strides[0],
                               QtGui.QImage.Format_RGB32)
        qtpixmap = QtGui.QPixmap(qtimage)
        self.movieImgLabel.setPixmap(qtpixmap)

        self.frame = 0
        self.nucleiSelected = None
        self.hdf5file = path

        self.populateExperiments()

    def populateExperiments(self):
        model = QtGui.QStandardItemModel()
        self.experimentsTreeView.setModel(model)
        self.experimentsTreeView.setUniformRowHeights(True)
        with h5py.File(self.hdf5file, "r") as f:
            for cond in f.iterkeys():
                conditem = QtGui.QStandardItem(cond)
                for run in f[cond].iterkeys():
                    runitem = QtGui.QStandardItem(run)
                    self.experimentsTreeView.expand(runitem.index())
                    conditem.appendRow(runitem)
                if conditem.rowCount() > 0:
                    model.appendRow(conditem)
                    # expand list
                    index = model.indexFromItem(conditem)
                    self.experimentsTreeView.expand(index)
        # select first row
        self.experimentsTreeView.setCurrentIndex(model.index(0, 1))
        QtCore.QObject.connect(self.experimentsTreeView.selectionModel(),
                               QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.expChange)

    @QtCore.pyqtSlot('QModelIndex, QModelIndex')
    def expChange(self, current, previous):
        self.condition = current.parent().data().toString()
        self.run = current.data().toString()
        self.frame = 0
        self.populateFramesList()
        self.renderFrame()
        self.populateNuclei()

    def populateFramesList(self):
        model = QtGui.QStandardItemModel()
        self.frameListView.setModel(model)
        with h5py.File(self.hdf5file, "r") as f:
            sel = f['%s/%s/raw' % (self.condition, self.run)]
            for fr in range(len(sel)):
                # nuc = sel[nucID]
                conditem = QtGui.QStandardItem('%d' % fr)
                model.appendRow(conditem)
        QtCore.QObject.connect(self.frameListView.selectionModel(),
                               QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.frameChange)

    @QtCore.pyqtSlot('QModelIndex, QModelIndex')
    def frameChange(self, current, previous):
        self.frame = int(current.data().toString())
        self.renderFrame()

    def populateNuclei(self):
        model = QtGui.QStandardItemModel()
        self.nucleiListView.setModel(model)
        with h5py.File(self.hdf5file, "r") as f:
            nuc = f['%s/%s/measurements/nuclei' % (self.condition, self.run)]
            sel = f['%s/%s/selection' % (self.condition, self.run)]
            for nucID in nuc:
                conditem = QtGui.QStandardItem(nucID)
                # conditem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                if nucID in sel:
                    conditem.setData(QtCore.QVariant(Qt.Checked), Qt.CheckStateRole)
                else:
                    conditem.setData(QtCore.QVariant(Qt.Unchecked), Qt.CheckStateRole)
                model.appendRow(conditem)
        QtCore.QObject.connect(self.nucleiListView.selectionModel(),
                               QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.nucleiChange)

    @QtCore.pyqtSlot('QModelIndex, QModelIndex')
    def nucleiChange(self, current, previous):
        self.nucleiSelected = int(current.data().toString()[1:])
        self.populateCentrosomes()
        self.renderFrame()

    def populateCentrosomes(self):
        model = QtGui.QStandardItemModel()
        self.centrosomeListView.setModel(model)
        with h5py.File(self.hdf5file, "r") as f:
            centrosome_list = f['%s/%s/measurements/centrosomes' % (self.condition, self.run)]
            sel = f['%s/%s/selection/N%02d' % (self.condition, self.run, self.nucleiSelected)]
            for cntrID in centrosome_list:
                cntr = centrosome_list[cntrID]
                cfxy = cntr['pos'].value
                cnt_frames = cfxy.T[0]

                if self.frame in cnt_frames:
                    conditem = QtGui.QStandardItem(cntrID)
                    if cntrID in sel:
                        conditem.setData(QtCore.QVariant(Qt.Checked), Qt.CheckStateRole)
                    else:
                        conditem.setData(QtCore.QVariant(Qt.Unchecked), Qt.CheckStateRole)
                    model.appendRow(conditem)

        QtCore.QObject.connect(self.centrosomeListView.selectionModel(),
                               QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.nucleiChange)

    def renderFrame(self):
        with h5py.File(self.hdf5file, "r") as f:
            ch2 = f['%s/%s/raw/%03d/channel-2' % (self.condition, self.run, self.frame)]
            data = ch2[:]
            self.resolution = ch2.parent.attrs['resolution']

            img_8bit = ((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8)  # map the data range to 0 - 255
            qtimage = QtGui.QImage(img_8bit.repeat(4), 512, 512, QtGui.QImage.Format_RGB32)
            self.imagePixmap = QPixmap(qtimage)
            self.drawMeasurements()
            self.movieImgLabel.setPixmap(self.imagePixmap)

    def drawMeasurements(self):
        print 'drawMeasurements call'
        with h5py.File(self.hdf5file, "r") as f:
            nuclei_list = f['%s/%s/measurements/nuclei' % (self.condition, self.run)]
            centrosome_list = f['%s/%s/measurements/centrosomes' % (self.condition, self.run)]
            sel = f['%s/%s/selection' % (self.condition, self.run)]

            painter = QPainter()
            painter.begin(self.imagePixmap)
            painter.setRenderHint(QPainter.Antialiasing)

            for nucID in nuclei_list:
                nuc = nuclei_list[nucID]
                nfxy = nuc['pos'].value
                nuc_frames = nfxy.T[0]
                if self.frame in nuc_frames:
                    fidx = nuc_frames.searchsorted(self.frame)
                    nx = nfxy[fidx][1] * self.resolution
                    ny = nfxy[fidx][2] * self.resolution

                    flagIsSelectedNuclei = int(nucID[1:]) == self.nucleiSelected

                    painter.setPen(QPen(QBrush(QColor('transparent')), 2))
                    if (self.nucleiSelected is None and nucID in sel) or \
                            (self.nucleiSelected is not None and flagIsSelectedNuclei):
                        painter.setBrush(QColor('blue'))
                    else:
                        painter.setBrush(QColor('gray'))
                    painter.drawEllipse(nx - 5, ny - 5, 10, 10)

                    painter.setPen(QPen(QBrush(QColor('white')), 2))
                    painter.drawText(nx + 10, ny + 5, nucID)

            for cntrID in centrosome_list:
                cntr = centrosome_list[cntrID]
                cfxy = cntr['pos'].value
                cnt_frames = cfxy.T[0]

                if self.frame in cnt_frames:
                    fidx = cnt_frames.searchsorted(self.frame)
                    cx = cfxy[fidx][1] * self.resolution
                    cy = cfxy[fidx][2] * self.resolution

                    for nucID in sel:
                        flagIsSelectedNuclei = int(nucID[1:]) == self.nucleiSelected

                        painter.setBrush(QColor('transparent'))
                        if self.nucleiSelected is None or (
                                            self.nucleiSelected is not None and flagIsSelectedNuclei and cntrID in sel[
                                    nucID]):
                            painter.setPen(QPen(QBrush(QColor('green')), 2))
                            painter.drawEllipse(cx - 5, cy - 5, 10, 10)
                        # else:
                        #     painter.setPen(QPen(QBrush(QColor('gray')), 2))
                        #     painter.drawEllipse(cx - 5, cy - 5, 10, 10)

                        painter.setPen(QPen(QBrush(QColor('white')), 2))
                        painter.drawText(cx + 10, cy + 5, cntrID)

            painter.end()


if __name__ == '__main__':
    import sys

    from PyQt4.QtCore import QT_VERSION_STR
    from PyQt4.Qt import PYQT_VERSION_STR

    base_path = os.path.abspath('%s' % os.getcwd())
    print('Qt version:', QT_VERSION_STR)
    print('PyQt version:', PYQT_VERSION_STR)
    print('Working dir:', os.getcwd())
    print('Base dir:', base_path)
    print skimage.io.available_plugins
    os.chdir(base_path)

    app = QtGui.QApplication(sys.argv)

    folders = ExperimentsList('centrosomes.nexus.hdf5')
    # folders.setGeometry(1200, 100, 900, 800)
    folders.show()

    # imgWindow = ImagesWindow(path, df_sel)
    # sa = QtGui.QScrollArea()
    # sa.setGeometry(700, 100, 910, 900)
    # sa.setWidgetResizable(True)
    # sa.setWidget(imgWindow)
    # sa.show()
    # QtCore.QObject.connect(folders, QtCore.SIGNAL('fileSelected(QString)'), imgWindow.applyFilter)
    # QtCore.QObject.connect(app, QtCore.SIGNAL('aboutToQuit()'), closing)

    sys.exit(app.exec_())

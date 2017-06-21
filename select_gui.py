import os

import h5py
import numpy as np
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QAbstractItemView, QBrush, QColor, QPainter, QPen, QPixmap

import hdf5_nexus as hdf


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
        self.centrosomeDropped = False

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
        self.populateCentrosomes()

    def populateFramesList(self):
        with h5py.File(self.hdf5file, "r") as f:
            sel = f['%s/%s/raw' % (self.condition, self.run)]
            self.frameHSlider.setMaximum(len(sel) - 1)
        QtCore.QObject.connect(self.frameHSlider, QtCore.SIGNAL('valueChanged(int)'), self.frameSliderChange)

    @QtCore.pyqtSlot('int')
    def frameSliderChange(self, value):
        self.frame = value
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
                conditem.setCheckable(True)
                model.appendRow(conditem)
        QtCore.QObject.connect(self.nucleiListView.selectionModel(),
                               QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.nucleiChange)
        model.itemChanged.connect(self.nucleiTickChange)

    @QtCore.pyqtSlot('QModelIndex, QModelIndex')
    def nucleiChange(self, current, previous):
        self.nucleiSelected = int(current.data().toString()[1:])
        self.populateCentrosomes()
        self.renderFrame()

    @QtCore.pyqtSlot('QStandardItem')
    def nucleiTickChange(self, item):
        self.nucleiSelected = int(item.text()[1:])
        with h5py.File(self.hdf5file, 'a') as f:
            sel = f['%s/%s/selection' % (self.condition, self.run)]
            del sel['N%02d' % self.nucleiSelected]

        self.populateNuclei()
        self.populateCentrosomes()
        self.renderFrame()

    def populateCentrosomes(self):
        model = QtGui.QStandardItemModel()
        modelA = QtGui.QStandardItemModel()
        modelB = QtGui.QStandardItemModel()

        self.centrosomeListView.setModel(model)
        self.centrosomeListView.setDragEnabled(True)
        self.centrosomeListView.setDragDropMode(QAbstractItemView.InternalMove)

        self.centrosomeListView_A.setModel(modelA)
        self.centrosomeListView_A.setAcceptDrops(True)
        # self.centrosomeListView_A.setDropIndicatorShown(True)
        self.centrosomeListView_B.setModel(modelB)
        self.centrosomeListView_B.setAcceptDrops(True)
        with h5py.File(self.hdf5file, "r") as f:
            centrosome_list = f['%s/%s/measurements/centrosomes' % (self.condition, self.run)].keys()
            sel = f['%s/%s/selection' % (self.condition, self.run)]
            if 'N%02d' % self.nucleiSelected in sel:
                sel = sel['N%02d' % self.nucleiSelected]
                for cntrID in centrosome_list:
                    centrInA = cntrID in sel['A']
                    centrInB = cntrID in sel['B']

                    item = QtGui.QStandardItem(cntrID)
                    if centrInA:
                        item.setData(QtCore.QVariant(Qt.Checked), Qt.CheckStateRole)
                        item.setCheckable(True)
                        modelA.appendRow(item)
                    elif centrInB:
                        item.setData(QtCore.QVariant(Qt.Checked), Qt.CheckStateRole)
                        item.setCheckable(True)
                        modelB.appendRow(item)

        hlab = hdf.LabHDF5NeXusFile(filename='/Users/Fabio/centrosomes.nexus.hdf5')
        for cntrID in centrosome_list:
            if not hlab.isCentrosomeAssociated(cntrID, self.condition, self.run):
                item = QtGui.QStandardItem(cntrID)
                model.appendRow(item)

        QtCore.QObject.connect(modelA, QtCore.SIGNAL('itemChanged(QStandardItem)'), self.centrosomeTickChange)
        modelA.itemChanged.connect(self.centrosomeTickChange)
        modelB.itemChanged.connect(self.centrosomeTickChange)

        QtCore.QObject.connect(modelA, QtCore.SIGNAL("rowsInserted(QModelIndex,int,int)"), self.centrosomeADrop)
        QtCore.QObject.connect(modelB, QtCore.SIGNAL("rowsInserted(QModelIndex,int,int)"), self.centrosomeBDrop)

    @QtCore.pyqtSlot('QStandardItem')
    def centrosomeTickChange(self, item):
        self.centrosomeSelected = str(item.text())
        if self.centrosomeDropped:
            self.centrosomeDropped = False
            hlab = hdf.LabHDF5NeXusFile(filename='/Users/Fabio/centrosomes.nexus.hdf5')
            c = int(self.centrosomeSelected[1:])
            hlab.associateCentrosomeWithNuclei(c, self.nucleiSelected, self.condition, self.run, self.centrosomeGroup)
        elif item.checkState() == QtCore.Qt.Unchecked:
            hlab = hdf.LabHDF5NeXusFile(filename='/Users/Fabio/centrosomes.nexus.hdf5')
            hlab.deleteAssociation(self.centrosomeSelected, self.nucleiSelected, self.condition, self.run)

        self.populateNuclei()
        self.populateCentrosomes()
        self.renderFrame()

    @QtCore.pyqtSlot('QModelIndex,int,int')
    def centrosomeADrop(self, item, start, end):
        self.centrosomeDropped = True
        self.centrosomeGroup = 0

    @QtCore.pyqtSlot('QModelIndex,int,int')
    def centrosomeBDrop(self, item, start, end):
        self.centrosomeDropped = True
        self.centrosomeGroup = 1

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
        # print 'drawMeasurements call'
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

                        centrInA = cntrID in sel['%s/A' % nucID]
                        centrInB = cntrID in sel['%s/B' % nucID]
                        painter.setBrush(QColor('transparent'))
                        if self.nucleiSelected is None:
                            painter.setPen(QPen(QBrush(QColor('green')), 2))
                        elif flagIsSelectedNuclei and centrInA:
                            painter.setPen(QPen(QBrush(QColor('orange')), 2))
                        elif flagIsSelectedNuclei and centrInB:
                            painter.setPen(QPen(QBrush(QColor('red')), 2))
                        else:
                            painter.setPen(QPen(QBrush(QColor('gray')), 2))
                        painter.drawEllipse(cx - 5, cy - 5, 10, 10)

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
    os.chdir(base_path)

    app = QtGui.QApplication(sys.argv)

    folders = ExperimentsList('/Users/Fabio/centrosomes.nexus.hdf5')
    # folders.setGeometry(1200, 100, 900, 800)
    folders.show()

    sys.exit(app.exec_())

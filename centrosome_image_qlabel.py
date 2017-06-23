import h5py
import numpy as np
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QBrush, QColor, QPainter, QPen, QPixmap


class CentrosomeImageQLabel(QtGui.QLabel):
    def __init__(self, parent, hdf_file=None):
        QtGui.QLabel.__init__(self, parent)
        self.selected = True
        self._hdf5file = hdf_file
        self.nucleiSelected = None
        self.dataHasChanged = False
        self.condition = None
        self.run = None
        self.frame = None
        self.resolution = None
        self.imagePixmap = None

        self.clear()

    @property
    def hdf5file(self):
        return self._hdf5file

    @hdf5file.setter
    def hdf5file(self, hdf_file):
        if hdf_file is not None:
            print("Setting value")
            self._hdf5file = hdf_file
            with h5py.File(hdf_file, "r") as f:
                self.condition = f.keys()[0]
                self.run = f[self.condition].keys()[0]
                self.frame = int(f['%s/%s/raw' % (self.condition, self.run)].keys()[0])

    def clear(self):
        imgarr = np.zeros(shape=(512, 512), dtype=np.uint32)
        qtimage = QtGui.QImage(imgarr.data, imgarr.shape[1], imgarr.shape[0], imgarr.strides[0],
                               QtGui.QImage.Format_RGB32)
        self.imagePixmap = QtGui.QPixmap(qtimage)
        self.setPixmap(self.imagePixmap)

    def mouseReleaseEvent(self, ev):
        self.emit(QtCore.SIGNAL('clicked()'))

    def paintEvent(self, event):
        if self.dataHasChanged:
            # print 'paintEvent reloading data from file %s' % self.hdf5file
            self.dataHasChanged = False
            with h5py.File(self.hdf5file, "r") as f:
                ch2 = f['%s/%s/raw/%03d/channel-2' % (self.condition, self.run, self.frame)]
                data = ch2[:]
                self.resolution = ch2.parent.attrs['resolution']

                img_8bit = ((data - data.min()) / (data.ptp() / 255.0)).astype(
                    np.uint8)  # map the data range to 0 - 255
                qtimage = QtGui.QImage(img_8bit.repeat(4), 512, 512, QtGui.QImage.Format_RGB32)
                self.imagePixmap = QPixmap(qtimage)
                self.draw_measurements()
                self.setPixmap(self.imagePixmap)
        return QtGui.QLabel.paintEvent(self, event)

    def render_frame(self, condition, run, frame, nuclei_selected=None):
        self.condition, self.run, self.frame = condition, run, frame
        self.nucleiSelected = nuclei_selected
        self.dataHasChanged = True

    def draw_measurements(self):
        with h5py.File(self._hdf5file, "r") as f:
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

                    is_in_selected_nuclei = int(nucID[1:]) == self.nucleiSelected

                    painter.setPen(QPen(QBrush(QColor('transparent')), 2))
                    if (self.nucleiSelected is None and nucID in sel) or \
                            (self.nucleiSelected is not None and is_in_selected_nuclei):
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

                    nuclei_sel = [n for n in sel if n != 'pandas_dataframe']
                    if len(nuclei_sel) > 0:
                        for nucID in nuclei_sel:
                            is_in_selected_nuclei = int(nucID[1:]) == self.nucleiSelected

                            centr_in_a = cntrID in sel['%s/A' % nucID]
                            centr_in_b = cntrID in sel['%s/B' % nucID]
                            painter.setBrush(QColor('transparent'))
                            if self.nucleiSelected is None:
                                painter.setPen(QPen(QBrush(QColor('green')), 2))
                            elif is_in_selected_nuclei and centr_in_a:
                                painter.setPen(QPen(QBrush(QColor('orange')), 2))
                            elif is_in_selected_nuclei and centr_in_b:
                                painter.setPen(QPen(QBrush(QColor('red')), 2))
                            else:
                                painter.setPen(QPen(QBrush(QColor('gray')), 2))
                            painter.drawEllipse(cx - 5, cy - 5, 10, 10)

                            painter.setPen(QPen(QBrush(QColor('white')), 2))
                            painter.drawText(cx + 10, cy + 5, cntrID)
                    else:
                        painter.setBrush(QColor('transparent'))
                        painter.setPen(QPen(QBrush(QColor('gray')), 2))
                        painter.drawEllipse(cx - 5, cy - 5, 10, 10)
                        painter.setPen(QPen(QBrush(QColor('white')), 2))
                        painter.drawText(cx + 10, cy + 5, cntrID)

            painter.end()

import h5py
import numpy as np
import pandas as pd
from PyQt4 import Qt, QtCore, QtGui
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
        self.image_pixmap = None

        self.clear()

    @property
    def hdf5file(self):
        return self._hdf5file

    @hdf5file.setter
    def hdf5file(self, hdf_file):
        if hdf_file is not None:
            print('Setting value')
            self._hdf5file = hdf_file
            with h5py.File(hdf_file, 'r') as f:
                self.condition = f.keys()[0]
                self.run = f[self.condition].keys()[0]
                self.frame = int(f['%s/%s/raw' % (self.condition, self.run)].keys()[0])

    def clear(self):
        imgarr = np.zeros(shape=(512, 512), dtype=np.uint32)
        qtimage = QtGui.QImage(imgarr.data, imgarr.shape[1], imgarr.shape[0], imgarr.strides[0],
                               QtGui.QImage.Format_RGB32)
        self.image_pixmap = QtGui.QPixmap(qtimage)
        self.setPixmap(self.image_pixmap)

    def mouseReleaseEvent(self, ev):
        self.emit(QtCore.SIGNAL('clicked()'))

    def paintEvent(self, event):
        if self.dataHasChanged:
            # print 'paintEvent reloading data from file %s' % self.hdf5file
            self.dataHasChanged = False
            with h5py.File(self.hdf5file, 'r') as f:
                ch2 = f['%s/%s/raw/%03d/channel-2' % (self.condition, self.run, self.frame)]
                data = ch2[:]
                self.resolution = ch2.parent.attrs['resolution']
                # map the data range to 0 - 255
                img_8bit = ((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8)
                qtimage = QtGui.QImage(img_8bit.repeat(4), 512, 512, QtGui.QImage.Format_RGB32)
                self.image_pixmap = QPixmap(qtimage)
                self.draw_measurements()
                self.setPixmap(self.image_pixmap)
        return QtGui.QLabel.paintEvent(self, event)

    def render_frame(self, condition, run, frame, nuclei_selected=None):
        self.condition, self.run, self.frame = condition, run, frame
        self.nucleiSelected = nuclei_selected
        self.dataHasChanged = True
        self.repaint()

    def draw_measurements(self):
        df = pd.read_hdf(self.hdf5file, key='%s/%s/measurements/pandas_dataframe' % (self.condition, self.run),
                         mode='r')
        with h5py.File(self._hdf5file, 'r') as f:
            nuclei_list = f['%s/%s/measurements/nuclei' % (self.condition, self.run)]
            centrosome_list = f['%s/%s/measurements/centrosomes' % (self.condition, self.run)]
            sel = f['%s/%s/selection' % (self.condition, self.run)]

            painter = QPainter()
            painter.begin(self.image_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)

            for nucID in nuclei_list:
                nuc = nuclei_list[nucID]
                nid = int(nucID[1:])
                if nid == 0: continue
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

                    # get nuclei boundary as a polygon
                    df_nucfr = df[(df['Nuclei'] == nid) & (df['Frame'] == self.frame)]
                    if len(df_nucfr['NuclBound'].values) > 0:
                        nuc_boundary_str = df_nucfr['NuclBound'].values[0]
                        if nuc_boundary_str[1:-1] != '':
                            nucb_points = eval(nuc_boundary_str[1:-1])
                            nucb_qpoints = [Qt.QPoint(x * self.resolution, y * self.resolution) for x, y in nucb_points]
                            nucb_poly = Qt.QPolygon(nucb_qpoints)

                            painter.setPen(QPen(QBrush(QColor('red')), 2))
                            painter.setBrush(QColor('transparent'))
                            painter.drawPolygon(nucb_poly)

            for cntrID in centrosome_list:
                cntr = centrosome_list[cntrID]
                cfxy = cntr['pos'].value
                cnt_frames = cfxy.T[0]

                if self.frame in cnt_frames:
                    fidx = cnt_frames.searchsorted(self.frame)
                    cx = cfxy[fidx][1] * self.resolution
                    cy = cfxy[fidx][2] * self.resolution

                    nuclei_sel = [n for n in sel if (n != 'pandas_dataframe' and n != 'pandas_masks')]
                    painter.setBrush(QColor('transparent'))
                    if len(nuclei_sel) > 0:
                        painter.setPen(QPen(QBrush(QColor('gray')), 1))
                        painter.drawEllipse(cx - 5, cy - 5, 10, 10)
                        painter.setPen(QPen(QBrush(QColor('white')), 1))
                        painter.drawText(cx + 10, cy + 5, cntrID)
                    else:
                        painter.setPen(QPen(QBrush(QColor('gray')), 1))
                        painter.drawEllipse(cx - 5, cy - 5, 10, 10)
                        painter.setPen(QPen(QBrush(QColor('white')), 1))
                        painter.drawText(cx + 10, cy + 5, cntrID)

            # draw selection
            for nuclei_str in f['%s/%s/selection' % (self.condition, self.run)]:
                if nuclei_str == 'pandas_dataframe' or nuclei_str == 'pandas_masks': continue
                nuclei_id = int(nuclei_str[1:])
                centrosomes_of_nuclei_a = f['%s/%s/selection/%s/A' % (self.condition, self.run, nuclei_str)].keys()
                centrosomes_of_nuclei_b = f['%s/%s/selection/%s/B' % (self.condition, self.run, nuclei_str)].keys()
                painter.setPen(QPen(QBrush(QColor('orange')), 2))
                for centr_str in centrosomes_of_nuclei_a:
                    cntr = centrosome_list[centr_str]
                    cfxy = cntr['pos'].value
                    cnt_frames = cfxy.T[0]
                    if self.frame in cnt_frames:
                        fidx = cnt_frames.searchsorted(self.frame)
                        cx = cfxy[fidx][1] * self.resolution
                        cy = cfxy[fidx][2] * self.resolution
                        painter.drawEllipse(cx - 5, cy - 5, 10, 10)

                painter.setPen(QPen(QBrush(QColor('red')), 2))
                for centr_str in centrosomes_of_nuclei_b:
                    cntr = centrosome_list[centr_str]
                    cfxy = cntr['pos'].value
                    cnt_frames = cfxy.T[0]
                    if self.frame in cnt_frames:
                        fidx = cnt_frames.searchsorted(self.frame)
                        cx = cfxy[fidx][1] * self.resolution
                        cy = cfxy[fidx][2] * self.resolution
                        painter.drawEllipse(cx - 5, cy - 5, 10, 10)

            painter.end()

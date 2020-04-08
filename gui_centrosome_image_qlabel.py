import logging

import h5py
import numpy as np
import pandas as pd

from PyQt5 import QtGui
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QPolygon
from PyQt5.QtWidgets import QLabel


class CentrosomeImageQLabel(QLabel):
    def __init__(self, parent, hdf_file=None):
        QLabel.__init__(self, parent)
        self.selected = True
        self._hdf5file = hdf_file
        self.nucleiSelected = None
        self.dataHasChanged = False
        self.condition = None
        self.run = None
        self.frame = None
        self.resolution = None
        self.image_pixmap = None
        self.dwidth = 0
        self.dheight = 0

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
                self.condition = list(f.keys())[0]
                self.run = list(f[self.condition].keys())[0]
                self.frame = int(list(f['%s/%s/raw' % (self.condition, self.run)].keys())[0])

    def clear(self):
        imgarr = np.zeros(shape=(512, 512), dtype=np.uint32)
        qtimage = QtGui.QImage(imgarr.data, imgarr.shape[1], imgarr.shape[0], imgarr.strides[0],
                               QtGui.QImage.Format_RGB32)
        self.image_pixmap = QtGui.QPixmap(qtimage)
        self.setPixmap(self.image_pixmap)

    def mouseReleaseEvent(self, ev):
        self.clicked.emit()

    def paintEvent(self, event):
        if self.dataHasChanged:
            # print 'paintEvent reloading data from file %s' % self.hdf5file
            self.dataHasChanged = False
            with h5py.File(self.hdf5file, 'r') as f:
                ch2 = f['%s/%s/raw/%03d/channel-2' % (self.condition, self.run, self.frame)]
                data = ch2[:]
                self.resolution = ch2.parent.attrs['resolution']

            self.dwidth, self.dheight = data.shape
            # map the data range to 0 - 255
            img_8bit = ((data - data.min()) / (data.ptp() / 255.0)).astype(np.uint8)
            qtimage = QtGui.QImage(img_8bit.repeat(3), self.dwidth, self.dheight, QtGui.QImage.Format_RGB888)
            self.image_pixmap = QPixmap(qtimage)
            self.draw_measurements()
            self.setPixmap(self.image_pixmap)
        return QLabel.paintEvent(self, event)

    def render_frame(self, condition, run, frame, nuclei_selected=None):
        self.condition, self.run, self.frame = condition, run, frame
        self.nucleiSelected = nuclei_selected
        self.dataHasChanged = True
        self.repaint()

    def draw_measurements(self):
        with h5py.File(self._hdf5file, 'r') as f:
            if 'pandas_dataframe' not in f['%s/%s/measurements' % (self.condition, self.run)]:
                raise KeyError('No data for selected condition-run.')

            df = pd.read_hdf(self.hdf5file, key='%s/%s/measurements/pandas_dataframe' % (self.condition, self.run))
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
                    painter.drawText(10, 30, '%02d - (%03d,%03d)' % (self.frame, self.dwidth, self.dheight))

                    # get nuclei boundary as a polygon
                    df_nucfr = df[(df['Nuclei'] == nid) & (df['Frame'] == self.frame)]
                    if len(df_nucfr['NuclBound'].values) > 0:
                        cell_boundary = df_nucfr['NuclBound'].values[0]
                        if cell_boundary[1:-1] != '':
                            nucb_points = eval(cell_boundary[1:-1])
                            nucb_qpoints = [QPoint(x * self.resolution, y * self.resolution) for x, y in nucb_points]
                            nucb_poly = QPolygon(nucb_qpoints)

                            if nid == self.nucleiSelected:
                                painter.setPen(QPen(QBrush(QColor('yellow')), 2))
                            else:
                                painter.setPen(QPen(QBrush(QColor('red')), 1))
                            painter.setBrush(QColor('transparent'))
                            painter.drawPolygon(nucb_poly)

                    if 'boundary' in f['%s/%s/processed' % (self.condition, self.run)]:
                        try:
                            k = '%s/%s/processed/boundary' % (self.condition, self.run)
                            dfbound = pd.read_hdf(self.hdf5file, key=k)
                            dfbound = dfbound[(dfbound['Nuclei'] == nid) & (dfbound['Frame'] == self.frame)]
                            if not dfbound.empty:
                                cell_bnd_str = dfbound.iloc[0]['CellBound']
                                if type(cell_bnd_str) == str:
                                    cell_boundary = np.array(eval(cell_bnd_str)) * self.resolution
                                    cell_centroid = dfbound.iloc[0][['CellX', 'CellY']].values * self.resolution
                                    nucb_qpoints = [QPoint(x, y) for x, y in cell_boundary]
                                    nucb_poly = QPolygon(nucb_qpoints)

                                    painter.setBrush(QColor('transparent'))
                                    painter.setPen(QPen(QBrush(QColor(0, 255, 0)), 2))
                                    painter.drawPolygon(nucb_poly)

                                    painter.drawText(cell_centroid[0] + 5, cell_centroid[1], 'C%02d' % (nid))

                                    painter.setBrush(QColor(0, 255, 0))
                                    painter.drawEllipse(cell_centroid[0] - 5, cell_centroid[1] - 5, 10, 10)
                        except Exception as e:
                            # pass
                            logging.error('Found a problem rendering cell boundary' + str(e))
                            # del f['%s/%s/processed/boundary' % (self.condition, self.run)]

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
                # nuclei_id = int(nuclei_str[1:])
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

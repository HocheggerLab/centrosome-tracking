import logging
import os

print(os.__file__)

import h5py
import pandas as pd
import seaborn as sns
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import QTimer, Qt

import hdf5_nexus as hdf
import plot_special_tools as sp
from imagej_pandas import ImagejPandas

pd.options.display.max_colwidth = 10
logging.basicConfig(level=logging.DEBUG)


class ExperimentsList(QtGui.QWidget):
    def __init__(self, path):
        QtGui.QWidget.__init__(self)
        uic.loadUi('gui_eb3.ui', self)

        self.frame = 0
        self.total_frames = 0
        self.condition = None
        self.run = None
        self.nuclei_selected = None
        self.initial_dir = path

        self.movieImgLabel.exp_folder = self.initial_dir
        self.movieImgLabel.clear()

        self.populate_experiments()

        # Call a QTimer for animations
        self.timer = QTimer()
        self.timer.start(200)

        QtCore.QObject.connect(self.exportPandasButton, QtCore.SIGNAL('pressed()'), self.on_export_pandas_button)
        QtCore.QObject.connect(self.timer, QtCore.SIGNAL('timeout()'), self.anim)

    def anim(self):
        if self.total_frames > 0:
            self.frame = (self.frame + 1) % self.total_frames
            logging.debug('rendering frame %d' % self.frame)
            self.movieImgLabel.render_frame(self.condition, self.run, self.frame)

    def populate_experiments(self):
        model = QtGui.QStandardItemModel()
        self.experimentsTreeView.setModel(model)
        self.experimentsTreeView.setUniformRowHeights(True)
        for file in os.listdir(self.initial_dir):
            path = os.path.join(self.initial_dir, file)
            if os.path.isdir(path):
                # print('adding %s' % file)
                conditem = QtGui.QStandardItem(file)
                for root, directories, files in os.walk(path):
                    for i in files:
                        ext = i.split('.')[-1]
                        if ext == 'tif' and i[0:6] != 'Result':
                            # print('adding %s' % i)
                            runitem = QtGui.QStandardItem(i)
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
                               QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.on_exp_change)

    @QtCore.pyqtSlot('QModelIndex, QModelIndex')
    def on_exp_change(self, current, previous):
        self.condition = str(current.parent().data().toString())
        self.run = str(current.data().toString())
        self.frame = 0
        if len(self.condition) > 0:
            self.populate_frames_list()
            self.movieImgLabel.render_frame(self.condition, self.run, self.frame)
            self.currImageLabel.setText(self.run)
        else:
            self.nucleiListView.model().clear()
            self.movieImgLabel.clear()
            self.currImageLabel.setText('')

        # self.timer.start(200)

    def populate_frames_list(self):
        img, res, dt = sp.find_image(self.run, os.path.join(self.initial_dir, self.condition))
        self.total_frames = img.shape[0]
        logging.debug('image has %d frames' % self.total_frames)

    def populate_nuclei(self):
        model = QtGui.QStandardItemModel()
        self.nucleiListView.setModel(model)
        with h5py.File(self.hdf5file, 'r') as f:
            nuc = f['%s/%s/measurements/nuclei' % (self.condition, self.run)]
            sel = f['%s/%s/selection' % (self.condition, self.run)]
            for nucID in nuc:
                conditem = QtGui.QStandardItem(nucID)
                if nucID in sel:
                    conditem.setData(QtCore.QVariant(Qt.Checked), Qt.CheckStateRole)
                else:
                    conditem.setData(QtCore.QVariant(Qt.Unchecked), Qt.CheckStateRole)
                conditem.setCheckable(True)
                model.appendRow(conditem)
        QtCore.QObject.connect(self.nucleiListView.selectionModel(),
                               QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.on_nuclei_change)
        model.itemChanged.connect(self.on_nucleitick_change)

    def plot_tracks_of_nuclei(self, nuclei):
        self.mplDistance.clear()
        with h5py.File(self.hdf5file, 'r') as f:
            if 'pandas_dataframe' in f['%s/%s/processed' % (self.condition, self.run)]:
                df = pd.read_hdf(self.hdf5file, key='%s/%s/processed/pandas_dataframe' % (self.condition, self.run))
                mask = pd.read_hdf(self.hdf5file, key='%s/%s/processed/pandas_masks' % (self.condition, self.run))
                ixdf = df['Nuclei'] == nuclei
                ixmsk = mask['Nuclei'] == nuclei
                time, frame, dist = ImagejPandas.get_contact_time(df[ixdf], 10)
                sp.distance_to_nucleus(df[ixdf], self.mplDistance.canvas.ax, mask=mask[ixmsk], time_contact=time)

                # plot distance with respect to cell centroid
                if 'boundary' in f['%s/%s/processed' % (self.condition, self.run)]:
                    df = pd.read_hdf(self.hdf5file, key='%s/%s/processed/boundary' % (self.condition, self.run))
                    df = df[df['Nuclei'] == nuclei]
                    # logging.info('columnas de boundary: ' + str(df.columns))
                    if 'DistCell' in df:
                        df = df.set_index('Time').sort_index()
                        pal = sns.color_palette()
                        for k, [(centr_lbl), _df] in enumerate(df.groupby(['Centrosome'])):
                            color = pal[k % len(pal)]
                            dlbl = 'N%d-C%d' % (nuclei, centr_lbl)
                            _df['DistCell'].plot(ax=self.mplDistance.canvas.ax, label=dlbl, marker='<', sharex=True,
                                                 c=color)

        self.mplDistance.canvas.draw()

    @QtCore.pyqtSlot()
    def on_export_pandas_button(self):
        fname = QtGui.QFileDialog.getSaveFileName(self, caption='Save centrosome file',
                                                  directory='/Users/Fabio/boundary.pandas')
        mname = QtGui.QFileDialog.getSaveFileName(self, caption='Save mask dataframe file',
                                                  directory='/Users/Fabio/boundary-mask.pandas')
        fname = str(fname)
        mname = str(mname)

        self.reprocess_selections()
        print 'saving masks to %s' % (mname)
        hlab = hdf.LabHDF5NeXusFile(filename=self.hdf5file)
        msk = hlab.mask
        msk.to_pickle(mname)
        print 'saving centrosomes to %s' % (fname)
        df = hlab.dataframe
        df.to_pickle(fname)
        print 'export finished.'

    def reprocess_selections(self):
        with h5py.File(self.hdf5file, 'r') as f:
            conditions = f.keys()
        for cond in conditions:
            with h5py.File(self.hdf5file, 'a') as f:
                runs = f[cond].keys()
            for run in runs:
                try:
                    hlab = hdf.LabHDF5NeXusFile(filename=self.hdf5file)
                    hlab.process_selection_for_run(cond, run)
                except KeyError:
                    print 'skipping %s due to lack of data.'


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

    folders = ExperimentsList('/Users/Fabio/data/lab/eb3')
    folders.show()

    sys.exit(app.exec_())

import ConfigParser
import os
import re
from threading import Lock

import h5py
import pandas as pd
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import Qt, QTimer
from PyQt4.QtGui import QAbstractItemView

import hdf5_nexus as hdf
import special_plots as spc


class ExperimentsList(QtGui.QWidget):
    def __init__(self, path):
        QtGui.QWidget.__init__(self)
        uic.loadUi('ExperimentsSelectionWidget.ui', self)

        self.frame = 0
        self.total_frames = 0
        self.nuclei_selected = None
        self.hdf5file = path
        self.centrosome_dropped = False

        self.movieImgLabel.hdf5file = path
        self.movieImgLabel.clear()

        self.populate_experiments()

        # Call a QTimer for animations
        self.rendering_lock = Lock()
        QTimer.singleShot(1000, self.anim)

        QtCore.QObject.connect(self.exportPandasButton, QtCore.SIGNAL('pressed()'), self.on_export_pandas_button)
        QtCore.QObject.connect(self.exportSelectionButton, QtCore.SIGNAL('pressed()'), self.on_export_sel_button)
        QtCore.QObject.connect(self.importSelectionButton, QtCore.SIGNAL('pressed()'), self.on_import_sel_button)
        QtCore.QObject.connect(self.clearRunButton, QtCore.SIGNAL('pressed()'), self.on_clear_run_button)

    def anim(self):
        with self.rendering_lock:
            if self.total_frames > 0:
                self.frame = (self.frame + 1) % self.total_frames
                self.movieImgLabel.render_frame(self.condition, self.run, self.frame,
                                                nuclei_selected=self.nuclei_selected)
            QTimer.singleShot(500, self.anim)


    def populate_experiments(self):
        model = QtGui.QStandardItemModel()
        self.experimentsTreeView.setModel(model)
        self.experimentsTreeView.setUniformRowHeights(True)
        with h5py.File(self.hdf5file, 'r') as f:
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
                               QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.on_exp_change)

    @QtCore.pyqtSlot('QModelIndex, QModelIndex')
    def on_exp_change(self, current, previous):
        self.condition = str(current.parent().data().toString())
        self.run = str(current.data().toString())
        self.frame = 0
        self.mplDistance.canvas.ax.cla()
        if len(self.condition) > 0:
            self.populate_frames_list()
            self.movieImgLabel.render_frame(self.condition, self.run, self.frame)
            self.populate_nuclei()
            self.populate_centrosomes()
            self.mplDistance.clear()
        else:
            self.nucleiListView.model().clear()
            self.centrosomeListView.model().clear()
            self.centrosomeListView_A.model().clear()
            self.centrosomeListView_B.model().clear()
            self.movieImgLabel.clear()

        with h5py.File(self.hdf5file, 'r') as f:
            sel = f['%s/%s/raw' % (self.condition, self.run)]
            self.total_frames = len(sel)
        self.anim()

    def populate_frames_list(self):
        with h5py.File(self.hdf5file, 'r') as f:
            sel = f['%s/%s/raw' % (self.condition, self.run)]
            self.total_frames = len(sel)
            self.frameHSlider.setMaximum(self.total_frames - 1)
        QtCore.QObject.connect(self.frameHSlider, QtCore.SIGNAL('valueChanged(int)'), self.on_frame_slider_change)

    @QtCore.pyqtSlot('int')
    def on_frame_slider_change(self, value):
        self.frame = value
        self.movieImgLabel.render_frame(self.condition, self.run, self.frame, nuclei_selected=self.nuclei_selected)

    def populate_nuclei(self):
        model = QtGui.QStandardItemModel()
        self.nucleiListView.setModel(model)
        with h5py.File(self.hdf5file, 'r') as f:
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
                               QtCore.SIGNAL('currentChanged(QModelIndex, QModelIndex)'), self.on_nuclei_change)
        model.itemChanged.connect(self.on_nucleitick_change)

    @QtCore.pyqtSlot('QModelIndex, QModelIndex')
    def on_nuclei_change(self, current, previous):
        self.nuclei_selected = int(current.data().toString()[1:])
        self.populate_centrosomes()
        self.movieImgLabel.render_frame(self.condition, self.run, self.frame, nuclei_selected=self.nuclei_selected)
        self.plot_tracks_of_nuclei(self.nuclei_selected)

    @QtCore.pyqtSlot('QStandardItem')
    def on_nucleitick_change(self, item):
        self.nuclei_selected = int(item.text()[1:])
        if item.checkState() == QtCore.Qt.Unchecked:
            with h5py.File(self.hdf5file, 'a') as f:
                sel = f['%s/%s/selection' % (self.condition, self.run)]
                del sel['N%02d' % self.nuclei_selected]
            hlab = hdf.LabHDF5NeXusFile(filename=self.hdf5file)
            hlab.process_selection_for_run(self.condition, self.run)
            self.mplDistance.clear()

        self.populate_nuclei()
        self.populate_centrosomes()
        self.movieImgLabel.render_frame(self.condition, self.run, self.frame, nuclei_selected=self.nuclei_selected)

    def plot_tracks_of_nuclei(self, nuclei):
        self.mplDistance.clear()
        with h5py.File(self.hdf5file, 'r') as f:
            if 'pandas_dataframe' in f['%s/%s/processed' % (self.condition, self.run)]:
                read_ok = True
            else:
                read_ok = False
        if read_ok:
            df = pd.read_hdf(self.hdf5file, key='%s/%s/processed/pandas_dataframe' % (self.condition, self.run))
            mask = pd.read_hdf(self.hdf5file, key='%s/%s/processed/pandas_masks' % (self.condition, self.run))
            df = df[df['Nuclei'] == nuclei]
            mask = mask[mask['Nuclei'] == nuclei]
            spc.plot_distance_to_nucleus(df, self.mplDistance.canvas.ax, mask=mask, draw_interpolated=False)
            self.mplDistance.canvas.draw()

    def populate_centrosomes(self):
        model = QtGui.QStandardItemModel()
        modelA = QtGui.QStandardItemModel()
        modelB = QtGui.QStandardItemModel()

        self.centrosomeListView.setModel(model)
        self.centrosomeListView.setDragEnabled(True)
        self.centrosomeListView.setDragDropMode(QAbstractItemView.InternalMove)

        self.centrosomeListView_A.setModel(modelA)
        self.centrosomeListView_A.setAcceptDrops(True)
        self.centrosomeListView_B.setModel(modelB)
        self.centrosomeListView_B.setAcceptDrops(True)
        with h5py.File(self.hdf5file, 'r') as f:
            centrosome_list = f['%s/%s/measurements/centrosomes' % (self.condition, self.run)].keys()
            sel = f['%s/%s/selection' % (self.condition, self.run)]
            if self.nuclei_selected is not None and 'N%02d' % self.nuclei_selected in sel:
                sel = sel['N%02d' % self.nuclei_selected]
                for cntrID in centrosome_list:
                    centr_in_a = cntrID in sel['A']
                    centr_in_b = cntrID in sel['B']

                    item = QtGui.QStandardItem(cntrID)
                    if centr_in_a:
                        item.setData(QtCore.QVariant(Qt.Checked), Qt.CheckStateRole)
                        item.setCheckable(True)
                        modelA.appendRow(item)
                    elif centr_in_b:
                        item.setData(QtCore.QVariant(Qt.Checked), Qt.CheckStateRole)
                        item.setCheckable(True)
                        modelB.appendRow(item)

        hlab = hdf.LabHDF5NeXusFile(filename=self.hdf5file)
        for cntrID in centrosome_list:
            if not hlab.is_centrosome_associated(cntrID, self.condition, self.run):
                item = QtGui.QStandardItem(cntrID)
                model.appendRow(item)

        QtCore.QObject.connect(modelA, QtCore.SIGNAL('itemChanged(QStandardItem)'), self.on_centrosometick_change)
        modelA.itemChanged.connect(self.on_centrosometick_change)
        modelB.itemChanged.connect(self.on_centrosometick_change)

        QtCore.QObject.connect(modelA, QtCore.SIGNAL('rowsInserted(QModelIndex,int,int)'), self.on_centrosome_a_drop)
        QtCore.QObject.connect(modelB, QtCore.SIGNAL('rowsInserted(QModelIndex,int,int)'), self.on_centrosome_b_drop)

    @QtCore.pyqtSlot('QStandardItem')
    def on_centrosometick_change(self, item):
        self.centrosome_selected = str(item.text())
        hlab = hdf.LabHDF5NeXusFile(filename=self.hdf5file)
        c = int(self.centrosome_selected[1:])
        if self.centrosome_dropped:
            self.centrosome_dropped = False
            hlab.associate_centrosome_with_nuclei(c, self.nuclei_selected, self.condition, self.run,
                                                  self.centrosome_group)
            hlab.process_selection_for_run(self.condition, self.run)
        elif item.checkState() == QtCore.Qt.Unchecked:
            hlab.delete_association(c, self.nuclei_selected, self.condition, self.run)

        self.populate_nuclei()
        self.populate_centrosomes()
        self.movieImgLabel.render_frame(self.condition, self.run, self.frame)
        self.plot_tracks_of_nuclei(self.nuclei_selected)

    @QtCore.pyqtSlot('QModelIndex,int,int')
    def on_centrosome_a_drop(self, item, start, end):
        self.centrosome_dropped = True
        self.centrosome_group = 0

    @QtCore.pyqtSlot('QModelIndex,int,int')
    def on_centrosome_b_drop(self, item, start, end):
        self.centrosome_dropped = True
        self.centrosome_group = 1

    @QtCore.pyqtSlot()
    def on_clear_run_button(self):
        if self.condition is not None and self.run is not None:
            hlab = hdf.LabHDF5NeXusFile(filename=self.hdf5file)
            hlab.clear_associations(self.condition, self.run)
            hlab.process_selection_for_run(self.condition, self.run)
            self.populate_nuclei()
            self.populate_centrosomes()

    @QtCore.pyqtSlot()
    def on_export_pandas_button(self):
        fname = QtGui.QFileDialog.getSaveFileName(self, caption='Save file',
                                                  directory='/Users/Fabio/centrosomes.pandas')
        fname = str(fname)

        print 'saving to %s' % fname
        self.reprocess_selections()
        hlab = hdf.LabHDF5NeXusFile(filename=self.hdf5file)
        df = hlab.dataframe
        df.to_pickle(fname)

    @QtCore.pyqtSlot()
    def on_export_sel_button(self):
        fname = QtGui.QFileDialog.getSaveFileName(self, caption='Save selection file',
                                                  directory='/Users/Fabio/centrosomes_selection.txt')
        fname = str(fname)
        print 'saving to %s' % fname

        config = ConfigParser.RawConfigParser()
        with h5py.File(self.hdf5file, 'r') as f:
            for cond in f:
                for run in f[cond]:
                    for nuclei_str in f['%s/%s/selection' % (cond, run)]:
                        if nuclei_str == 'pandas_dataframe' or nuclei_str == 'pandas_masks': continue
                        section = '%s.%s.%s' % (cond, run, nuclei_str)
                        config.add_section(section)
                        centrosomes_of_nuclei_a = f['%s/%s/selection/%s/A' % (cond, run, nuclei_str)].keys()
                        centrosomes_of_nuclei_b = f['%s/%s/selection/%s/B' % (cond, run, nuclei_str)].keys()
                        config.set(section, 'A', [c.encode('ascii', 'ignore') for c in centrosomes_of_nuclei_a])
                        config.set(section, 'B', [c.encode('ascii', 'ignore') for c in centrosomes_of_nuclei_b])

        # Write our configuration file
        with open(fname, 'w') as configfile:
            config.write(configfile)
        print '\r\nexport done.'

    def on_import_sel_button(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, caption='Load selection file', filter='Text (*.txt)',
                                                  directory='/Users/Fabio/centrosomes_selection.txt')
        if not fname: return
        fname = str(fname)

        print 'deleting old selection'
        with h5py.File(self.hdf5file, 'a') as f:
            for cond in f:
                for run in f[cond]:
                    for o in f['%s/%s/selection' % (cond, run)]:
                        del f['%s/%s/selection/%s' % (cond, run, o)]

        print 'opening %s' % fname
        selection = ConfigParser.ConfigParser()
        selection.read(fname)
        for sel in selection.sections():
            print sel
            cond, run, nucl = re.search('^(.+)\.(.+)\.N(.+)$', sel).groups()
            hlab = hdf.LabHDF5NeXusFile(filename=self.hdf5file)

            _A = eval(selection.get(sel, 'a'))
            _B = eval(selection.get(sel, 'b'))
            for c in _A:
                hlab.associate_centrosome_with_nuclei(int(c[1:]), int(nucl), cond, run, centrosome_group=0)
            for c in _B:
                hlab.associate_centrosome_with_nuclei(int(c[1:]), int(nucl), cond, run, centrosome_group=1)
        self.reprocess_selections()
        print '\r\ndone importing selection.'

    def reprocess_selections(self):
        with h5py.File(self.hdf5file, 'r') as f:
            conditions = f.keys()
        for cond in conditions:
            with h5py.File(self.hdf5file, 'a') as f:
                runs = f[cond].keys()
            for run in runs:
                hlab = hdf.LabHDF5NeXusFile(filename=self.hdf5file)
                hlab.process_selection_for_run(cond, run)


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
    folders.show()

    sys.exit(app.exec_())

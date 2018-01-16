import ConfigParser
import logging
import os
import re

import coloredlogs
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import QTimer, Qt
from matplotlib.backends.backend_pdf import PdfPages

import hdf5_nexus as hdf
import im_gabor
import mechanics as m
import plot_special_tools as sp
from imagej_pandas import ImagejPandas

pd.options.display.max_colwidth = 10
coloredlogs.install(fmt='%(levelname)s:%(funcName)s - %(message)s', level=logging.DEBUG)


class ExperimentsList(QtGui.QWidget):
    def __init__(self, path):
        QtGui.QWidget.__init__(self)
        uic.loadUi('gui_cell_boundary.ui', self)

        self.frame = 0
        self.total_frames = 0
        self.condition = None
        self.run = None
        self.nuclei_selected = None
        self.hdf5file = path
        self.centrosome_dropped = False

        self.movieImgLabel.hdf5file = path
        self.movieImgLabel.clear()

        self.populate_experiments()

        # Call a QTimer for animations
        self.timer = QTimer()
        self.timer.start(200)

        QtCore.QObject.connect(self.exportPandasButton, QtCore.SIGNAL('pressed()'), self.on_export_pandas_button)
        QtCore.QObject.connect(self.renderBoxplotButton, QtCore.SIGNAL('pressed()'), self.on_render_boxplot_button)
        QtCore.QObject.connect(self.exportSelectionButton, QtCore.SIGNAL('pressed()'), self.on_export_sel_button)
        QtCore.QObject.connect(self.importSelectionButton, QtCore.SIGNAL('pressed()'), self.on_import_sel_button)
        QtCore.QObject.connect(self.timer, QtCore.SIGNAL('timeout()'), self.anim)
        QtCore.QObject.connect(self.gaborLineEdit, QtCore.SIGNAL('editingFinished()'), self.on_gabor_change)

    def anim(self):
        if self.total_frames > 0:
            self.frame = (self.frame + 1) % self.total_frames
            self.movieImgLabel.render_frame(self.condition, self.run, self.frame,
                                            nuclei_selected=self.nuclei_selected)

    def populate_experiments(self):
        model = QtGui.QStandardItemModel()
        self.experimentsTreeView.setModel(model)
        self.experimentsTreeView.setUniformRowHeights(True)
        with h5py.File(self.hdf5file, 'r') as f:
            for cond in reversed(sorted(f.iterkeys())):
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
            self.mplDistance.clear()
        else:
            self.nucleiListView.model().clear()
            self.movieImgLabel.clear()

        with h5py.File(self.hdf5file, 'r') as f:
            sel = f['%s/%s/raw' % (self.condition, self.run)]
            self.total_frames = len(sel)
        self.timer.start(200)

    def populate_frames_list(self):
        with h5py.File(self.hdf5file, 'r') as f:
            sel = f['%s/%s/raw' % (self.condition, self.run)]
            self.total_frames = len(sel)

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

    @QtCore.pyqtSlot('QModelIndex, QModelIndex')
    def on_nuclei_change(self, current, previous):
        self.nuclei_selected = int(current.data().toString()[1:])

        with h5py.File(self.hdf5file, 'r+') as f:
            nuc_sel = f['%s/%s/selection/N%02d' % (self.condition, self.run, self.nuclei_selected)]
            gabor_thr = nuc_sel.attrs['gabor_threshold'] if 'gabor_threshold' in nuc_sel.attrs else None

        hlab = hdf.LabHDF5NeXusFile(filename=self.hdf5file)
        hlab.process_selection_for_run(self.condition, self.run)
        if gabor_thr is not None:
            self.gaborLineEdit.setText(str(gabor_thr))
            with h5py.File(self.hdf5file, 'r+') as f:
                nuc_sel = f['%s/%s/selection/N%02d' % (self.condition, self.run, self.nuclei_selected)]
                nuc_sel.attrs['gabor_threshold'] = gabor_thr
        else:
            self.gaborLineEdit.setText('70')

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
        self.movieImgLabel.render_frame(self.condition, self.run, self.frame, nuclei_selected=self.nuclei_selected)

    @QtCore.pyqtSlot('QStandardItem')
    def on_gabor_change(self):
        if self.condition is None or self.run is None or self.nuclei_selected is None:
            self.timer.start(200)
            return
        logging.info('on gabor: condition %s, run %s, nuclei %s' % (self.condition, self.run, self.nuclei_selected))

        if self.timer.isActive():
            logging.info('freezing timer.')
            self.timer.stop()
        df = pd.DataFrame()
        with h5py.File(self.hdf5file, 'r+') as f:
            if not 'N%02d' % self.nuclei_selected in f['%s/%s/selection' % (self.condition, self.run)]:
                self.timer.start(200)
                return

            if 'boundary' in f['%s/%s/processed' % (self.condition, self.run)]:
                df = pd.read_hdf(self.hdf5file, key='%s/%s/processed/boundary' % (self.condition, self.run))
            elif 'pandas_dataframe' in f['%s/%s/processed' % (self.condition, self.run)]:
                df = pd.read_hdf(self.hdf5file, key='%s/%s/processed/pandas_dataframe' % (self.condition, self.run))
                # df = df.drop(['CentX', 'CentY', 'CNx', 'CNy', 'NuclX', 'NuclY',
                #               'Dist', 'Speed', 'Acc', 'NuclBound'], axis=1)
            else:
                self.timer.start(200)
                return

            nuc_sel = f['%s/%s/selection/N%02d' % (self.condition, self.run, self.nuclei_selected)]
            new_gabor_thr = int(self.gaborLineEdit.text())
            old_gabor_thr = nuc_sel.attrs['gabor_threshold'] if 'gabor_threshold' in nuc_sel.attrs else None
            logging.debug('old gabor thr: %s, new thr: %s new==old %s' %
                          (old_gabor_thr, new_gabor_thr, old_gabor_thr == new_gabor_thr))

            if old_gabor_thr == new_gabor_thr:
                self.timer.start(200)
                return
            nuc_sel = f['%s/%s/selection/N%02d' % (self.condition, self.run, self.nuclei_selected)]
            nuc_sel.attrs['gabor_threshold'] = new_gabor_thr

            if new_gabor_thr == 0:
                logging.info('deleting data for nuclei N%02d' % self.nuclei_selected)
                ix = df['Nuclei'] == self.nuclei_selected
                df.loc[ix, 'CellBound'] = np.NaN
                df.loc[ix, 'CellX'] = np.NaN
                df.loc[ix, 'CellY'] = np.NaN

            else:
                logging.info('computing cell boundary.')
                for frame in f['%s/%s/raw' % (self.condition, self.run)]:
                    ch1 = f['%s/%s/raw/%s/channel-1' % (self.condition, self.run, frame)]
                    ch2 = f['%s/%s/raw/%s/channel-2' % (self.condition, self.run, frame)]
                    hoechst = ch1[:]
                    tubulin = ch2[:]
                    resolution = ch2.parent.attrs['resolution']

                    frame = int(frame)
                    marker = np.zeros(hoechst.shape, dtype=np.uint8)
                    nuclei_list = f['%s/%s/measurements/nuclei' % (self.condition, self.run)]
                    for nucID in nuclei_list:
                        nuc = nuclei_list[nucID]
                        nid = int(nucID[1:])
                        if nid == 0: continue
                        nfxy = nuc['pos'].value
                        nuc_frames = nfxy.T[0]
                        if frame in nuc_frames:
                            fidx = nuc_frames.searchsorted(frame)
                            nx = int(nfxy[fidx][1] * resolution)
                            ny = int(nfxy[fidx][2] * resolution)
                            cv2.circle(marker, (nx, ny), 5, nid, thickness=-1)

                    boundary_list, gabor = im_gabor.cell_boundary(tubulin, hoechst, markers=marker,
                                                                  threshold=new_gabor_thr)
                    for b in boundary_list:
                        nuc = b['id']
                        if nuc == self.nuclei_selected:
                            np.set_printoptions(threshold=int(np.prod(b['boundary'].shape)))
                            boundcell = np.array2string(b['boundary'] / resolution, separator=',')
                            cx, cy = b['centroid'] / resolution
                            ix = (df['Frame'] == frame) & (df['Nuclei'] == nuc)
                            if np.any(ix):
                                df.loc[ix, 'CellBound'] = boundcell
                                df.loc[ix, 'CellX'] = cx
                                df.loc[ix, 'CellY'] = cy

            df = df.rename(columns={'DistCell': 'dist', 'SpdCell': 'speed', 'AccCell': 'acc'})
            df = m.get_speed_acc_rel_to(df, x='CentX', y='CentY', rx='CellX', ry='CellY',
                                        time='Time', frame='Frame', group='Centrosome')
            df = df.rename(columns={'dist': 'DistCell', 'speed': 'SpdCell', 'acc': 'AccCell'})

            if 'boundary' in f['%s/%s/processed' % (self.condition, self.run)]:
                del f['%s/%s/processed/boundary' % (self.condition, self.run)]

        logging.debug('nuclei with cell boundary: %s' % df.loc[~df['CellBound'].isnull(), 'Nuclei'].unique())
        df.to_hdf(self.hdf5file, key='%s/%s/processed/boundary' % (self.condition, self.run), mode='r+')
        self.plot_tracks_of_nuclei(self.nuclei_selected)
        logging.info('animating again.')
        self.timer.start(200)

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
    def on_render_boxplot_button(self):
        logging.info('Rendering boxplot')
        df_out = pd.DataFrame()
        with h5py.File(self.hdf5file, 'r') as f:
            for experiment_tag in f:
                for run in f['%s' % experiment_tag]:
                    if 'boundary' in f['%s/%s/processed' % (experiment_tag, run)]:
                        df = pd.read_hdf(self.hdf5file, key='%s/%s/processed/boundary' % (experiment_tag, run))
                        df['condition'] = experiment_tag
                        df['run'] = run
                        df_out = df_out.append(df)
        df_valid = df_out.loc[~df_out['CellBound'].isnull(), :]
        logging.debug(df_valid[ImagejPandas.NUCLEI_INDIV_INDEX])

        dt_before_contact = 30
        t_per_frame = 5
        with PdfPages('/Users/Fabio/dist_nucleus_cell.pdf') as pdf:
            fig = plt.gcf()
            fig.clf()
            fig.set_size_inches(9.3, 9.3 / 2)
            ax = plt.gca()
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)

            stats = pd.DataFrame()
            idx = ImagejPandas.MASK_INDEX
            idx.append('run')
            for i, (id, idf) in enumerate(df_valid.groupby(ImagejPandas.NUCLEI_INDIV_INDEX)):
                if 'CentrLabel' in idf:
                    idf = idf.drop('CentrLabel', axis=1)
                s = idf.set_index(idx).sort_index()
                u = s.unstack('Centrosome')
                u = u.interpolate().stack().reset_index()
                idf = ImagejPandas.vel_acc_nuclei(u)

                d_thr = ImagejPandas.DIST_THRESHOLD * 3
                time_of_c, frame_of_c, dist_of_c = ImagejPandas.get_contact_time(idf, d_thr)
                logging.debug(i, id, time_of_c, frame_of_c, dist_of_c)

                if time_of_c is not None:
                    frame_before = frame_of_c - dt_before_contact / t_per_frame
                    if frame_before < 0:
                        frame_before = 0
                    dists_before_contact = list(idf[idf['Frame'] == frame_before]['Dist'])
                    min_dist = max_dist = time_before = np.NaN
                    if len(dists_before_contact) > 0:
                        max_dist = max(dists_before_contact)
                        min_dist = min(dists_before_contact)
                        time_before = idf[idf['Frame'] == frame_before]['Time'].unique()[0]
                else:
                    frame_before = time_before = min_dist = max_dist = np.NaN

                ini_frame = idf.set_index('Frame').sort_index().index[0]
                ini_time = idf[idf['Frame'] == ini_frame]['Time'].unique()[0]
                ini_dist_min = min(idf[idf['Frame'] == ini_frame]['Dist'])

                logging.debug(idf.loc[idf['Frame'] == frame_of_c, 'DistCell'].min())
                df_rown = pd.DataFrame({'Tag': [id],
                                        'Nuclei': idf['Nuclei'].unique()[0],
                                        'Frame': [frame_of_c],
                                        'Time': [time_of_c],
                                        'Stat': 'Contact',
                                        'Type': 'Nucleus',
                                        'Dist': [dist_of_c]})
                df_rowc = pd.DataFrame({'Tag': [id],
                                        'Nuclei': idf['Nuclei'].unique()[0],
                                        'Frame': [frame_of_c],
                                        'Time': [time_of_c],
                                        'Stat': 'Contact',
                                        'Type': 'Cell Boundary',
                                        'Dist': idf.loc[idf['Frame'] == frame_of_c, 'DistCell'].min()})
                stats = stats.append(df_rown, ignore_index=True)
                stats = stats.append(df_rowc, ignore_index=True)

            sdata = stats[(stats['Stat'] == 'Contact') & (stats['Dist'].notnull())][['Dist', 'Type']]
            sdata['Dist'] = sdata.Dist.astype(np.float64)  # fixes a bug of seaborn
            sns.boxplot(data=sdata, y='Dist', x='Type', whis=np.inf, width=0.3)
            for i, artist in enumerate(ax.artists):
                artist.set_facecolor('None')
                artist.set_edgecolor(sp.SUSSEX_COBALT_BLUE)
                artist.set_zorder(5000)
            for i, artist in enumerate(ax.lines):
                artist.set_color(sp.SUSSEX_COBALT_BLUE)
                artist.set_zorder(5000)

            ax = sns.swarmplot(data=sdata, y='Dist', x='Type', zorder=100, color=sp.SUSSEX_CORAL_RED)
            ax.set_xlabel('')
            ax.set_ylabel('D(time of contact) $[\mu m]$')

            pdf.savefig(transparent=True)

    @QtCore.pyqtSlot()
    def on_export_pandas_button(self):
        fname = QtGui.QFileDialog.getSaveFileName(self, caption='Save centrosome file',
                                                  directory='/Users/Fabio/boundary.pandas')
        mname = QtGui.QFileDialog.getSaveFileName(self, caption='Save mask dataframe file',
                                                  directory='/Users/Fabio/boundary-mask.pandas')
        fname = str(fname)
        mname = str(mname)

        self.reprocess_selections()
        logging.info('saving masks to %s' % (mname))
        hlab = hdf.LabHDF5NeXusFile(filename=self.hdf5file)
        msk = hlab.mask
        msk.to_pickle(mname)
        logging.info('saving centrosomes to %s' % (fname))
        df = hlab.dataframe
        df.to_pickle(fname)
        logging.info('export finished.')

    @QtCore.pyqtSlot()
    def on_export_sel_button(self):
        fname = QtGui.QFileDialog.getSaveFileName(self, caption='Save selection file',
                                                  directory='/Users/Fabio/boundary_selection.txt')
        fname = str(fname)
        logging.info('saving to %s' % fname)

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
        logging.info('export done.')

    def on_import_sel_button(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, caption='Load selection file', filter='Text (*.txt)',
                                                  directory='/Users/Fabio/boundary_selection.txt')
        if not fname: return
        fname = str(fname)

        logging.info('deleting old selection')
        with h5py.File(self.hdf5file, 'a') as f:
            for cond in f:
                for run in f[cond]:
                    for o in f['%s/%s/selection' % (cond, run)]:
                        del f['%s/%s/selection/%s' % (cond, run, o)]

        logging.info('opening %s' % fname)
        selection = ConfigParser.ConfigParser()
        selection.read(fname)
        for sel in selection.sections():
            logging.debug(sel)
            cond, run, nucl = re.search('^(.+)\.(.+)\.N(.+)$', sel).groups()
            hlab = hdf.LabHDF5NeXusFile(filename=self.hdf5file)

            _A = eval(selection.get(sel, 'a'))
            _B = eval(selection.get(sel, 'b'))
            for c in _A:
                hlab.associate_centrosome_with_nuclei(int(c[1:]), int(nucl), cond, run, centrosome_group=0)
            for c in _B:
                hlab.associate_centrosome_with_nuclei(int(c[1:]), int(nucl), cond, run, centrosome_group=1)
        self.reprocess_selections()
        logging.info('done importing selection.')

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
                    logging.warning('skipping %s due to lack of data.')


if __name__ == '__main__':
    import sys

    from PyQt4.QtCore import QT_VERSION_STR
    from PyQt4.Qt import PYQT_VERSION_STR

    base_path = os.path.abspath('%s' % os.getcwd())
    logging.info('Qt version:', QT_VERSION_STR)
    logging.info('PyQt version:', PYQT_VERSION_STR)
    logging.info('Working dir:', os.getcwd())
    logging.info('Base dir:', base_path)
    os.chdir(base_path)

    app = QtGui.QApplication(sys.argv)

    folders = ExperimentsList('/Users/Fabio/centrosomes.nexus.hdf5')
    folders.show()

    sys.exit(app.exec_())

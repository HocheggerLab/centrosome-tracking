import configparser
import json
import logging
import os

import coloredlogs
import numpy as np
import pandas as pd
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import QTimer

import tools.image as image
import mechanics as m
import parameters

pd.options.display.max_colwidth = 10
logging.basicConfig(level=logging.DEBUG)
coloredlogs.install()


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
        img, res, dt, _, _, _ = image.find_image(self.run, os.path.join(self.initial_dir, self.condition))
        self.total_frames = img.shape[0]
        logging.debug('image has %d frames' % self.total_frames)

    @QtCore.pyqtSlot()
    def on_export_pandas_button(self):
        logging.info('exporting selected dataset...')
        # fname = QtGui.QFileDialog.getSaveFileName(self, caption='Save selected Eb3 file',
        #                                           directory=constants.data_dir+'eb3_selected.pandas')
        # fname = str(fname)

        df = pd.read_pickle(parameters.compiled_data_dir + 'eb3filter.pandas')
        # get selection from disk
        with open(parameters.compiled_data_dir + 'eb3trk.selec.txt', 'r') as configfile:
            config = configparser.ConfigParser()
            config.readfp(configfile)

            selected = pd.DataFrame()
            for tag, t in df.groupby('tag'):
                if config.has_section(tag):
                    selection = set(json.loads(config.get(tag, 'selected')))
                    ix_sel = (df['tag'] == tag) & (df['particle'].isin(selection))
                    sel = df[ix_sel]
                    selected = selected.append(sel)
            selected.to_pickle(parameters.compiled_data_dir + 'eb3_selected.pandas')
            df = selected

        indiv_idx = ['condition', 'tag', 'particle']
        df = m.get_speed_acc(df, group=indiv_idx)
        df = m.get_center_df(df, group=indiv_idx)
        dfi = df.set_index('frame').sort_index()
        dfi['speed'] = dfi['speed'].abs()
        df_avg = dfi.groupby(indiv_idx)['time', 'speed'].mean()
        df_avg.loc[:, 'time'] = dfi.groupby(indiv_idx)['time'].first()
        df_avg.loc[:, 'trk_len'] = dfi.groupby(indiv_idx)['x'].count()
        df_avg.loc[:, 'length'] = dfi.groupby(indiv_idx)['s'].agg(np.sum)
        df_avg = df_avg.reset_index()
        df_avg.to_pickle(parameters.compiled_data_dir + 'eb3stats_sel.pandas')

        print('export finished.')


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

    folders = ExperimentsList(parameters.experiments_dir + 'eb3/')
    folders.show()

    sys.exit(app.exec_())

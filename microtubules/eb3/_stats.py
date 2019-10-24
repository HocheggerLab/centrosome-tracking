import logging

import numpy as np
import mechanics as m

log = logging.getLogger(__name__)


class Tracks():
    def __init__(self, df, time='time', x='x', y='y', msd='msd', y_lim=None, x_lim=None,
                 condition='condition', track_group=['indv'], cell_group=None):
        self.df = df
        self.xlim = y_lim
        self.ylim = x_lim
        self.condition = condition
        assert np.all(
            [a in self.df.columns for a in [time, x, y, condition]]), "Some columns missing."
        self.track_group = track_group
        self.cell_group = cell_group

        self.df = m.get_speed_acc(self.df, group=self.track_group)

    def per_track(self):
        cols_to_work = ['x', 'y', 'speed', 'acc'] + self.track_group
        data = self.df[cols_to_work]
        data.loc[:, 'speed'] = data['speed'].apply(np.abs)
        data.loc[:, 'acc'] = data['acc'].apply(np.abs)
        stats = (data.groupby(self.track_group)
                 .agg({'speed': [np.nanmean, np.nanstd, len]})
                 .rename(columns={'nanmean': 'mean', 'nanstd': 'std', 'len': 'N'}, level=1)
                 .reset_index()
                 )
        stats.columns = stats.columns.map('|'.join).str.strip('|')
        return stats

    def per_cell(self):
        assert self.cell_group is not None, "Cell group columns are required when object was constructed."

        cols_to_work = ['x', 'y', 'speed', 'acc'] + self.cell_group
        data = self.df[cols_to_work]
        data.loc[:, 'speed'] = data['speed'].apply(np.abs)
        data.loc[:, 'acc'] = data['acc'].apply(np.abs)
        stats = (data.groupby(self.cell_group)
                 .agg({'speed': [np.nanmean, np.nanstd, len]})
                 .rename(columns={'nanmean': 'mean', 'nanstd': 'std', 'len': 'N'}, level=1)
                 .reset_index()
                 )
        stats.columns = stats.columns.map('|'.join).str.strip('|')
        return stats

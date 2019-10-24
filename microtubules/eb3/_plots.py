import logging

import traces
import numpy as np
import pandas as pd

# from tools.matplotlib_essentials import plt
import matplotlib.ticker as ticker

import seaborn as sns

log = logging.getLogger(__name__)


class MSD():
    def __init__(self, df, time='time', x='x', y='y', msd='msd', y_lim=None, x_lim=None,
                 condition='condition', track_group=['indv'], cell_group=None,
                 uneven_times=True):
        self.df = df
        self.xlim = y_lim
        self.ylim = x_lim
        self.condition = condition
        assert np.all(
            [a in self.df.columns for a in [time, x, y, msd, condition]]), "Some columns missing."
        self.track_group = track_group
        self.cell_group = cell_group

        # construct timeseries using trace library and average moving window
        self.timeseries = pd.DataFrame()
        if not uneven_times:
            self.timeseries = self.df
            # self.timeseries.loc[:,"x"]=self.df[x]
            # self.timeseries.loc[:,"y"]=self.df[y]
            # self.timeseries.loc[:,"time"]=self.df[time]
            # self.timeseries.loc[:,"msd"]=self.df[msd]
            pass
        else:
            period = np.round(self.df.groupby(self.track_group).diff()['time'].median(), decimals=0)
            log.debug("resampling timeseries with a new period of %0.2f" % period)

            idv_k = 0
            for _ix_cond, _cond in self.df.groupby(self.condition):
                _ts_list = list()
                for ix, _df in _cond.groupby(self.track_group):
                    ts = (
                        traces.TimeSeries(data=[(t, y) for t, y in zip(_df['time'], _df['msd'])])
                            .moving_average(placement='left', sampling_period=period, start=0, pandas=True)
                            .reset_index()
                            .rename(columns={'index': 'time', 0: 'msd'})
                            .assign(condition=_ix_cond, indv=idv_k)
                    )

                    self.timeseries = self.timeseries.append(ts, ignore_index=True, sort=False)
                    idv_k += 1

            self.timeseries["time"] = self.timeseries["time"].apply(lambda v: np.round(v, decimals=1))

    @staticmethod
    def format_axes(ax, time_minor=5, time_major=15, msd_minor=25, msd_major=50, x_lim=None, y_lim=None):
        ax.legend(title=None, loc='upper left')

        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(time_major))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(time_minor))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(msd_major))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(msd_minor))

        ax.set_ylabel('MSD')
        ax.set_xlabel('time [s]')

    def sampled_times(self, ax):
        pass

    def track_each(self, ax, order=None):
        # plot of MSD for each track
        sns.lineplot(data=self.timeseries, x='time', y='msd',
                     units='indv', estimator=None,
                     hue=self.condition, hue_order=order, lw=0.2, alpha=1, ax=ax)
        self.format_axes(ax, x_lim=[self.df['time'].min(), self.df['time'].max()],
                         y_lim=[self.df['msd'].min(), self.df['msd'].max() * 1.2])

    def track_average(self, ax, order=None):
        # plot MSD track_average per condition
        sns.lineplot(data=self.timeseries, x='time', y='msd',
                     hue=self.condition, hue_order=order, lw=2, ax=ax)
        self.format_axes(ax, x_lim=[self.df['time'].min(), 30], time_major=10,
                         y_lim=[0, 15], msd_minor=5, msd_major=10, )

import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ImagejPandas(object):
    DIST_THRESHOLD = 1.0  # um before 1 frame of contact
    TIME_BEFORE_CONTACT = 30

    def __init__(self, filename):
        self.path_csv = filename
        self.df_centrosome = pd.read_csv(self.path_csv)
        base_path = os.path.dirname(filename)
        bname = os.path.basename(filename)
        self.fname = re.search('(.+)-table.csv$', bname).group(1)
        self.path_nuclei = '%s/%s-nuclei.csv' % (base_path, self.fname)

        self.df_nuclei = pd.read_csv(self.path_nuclei)
        self.centrosome_replacements = dict()

        self.df_centrosome.loc[self.df_centrosome.index, 'Time'] /= 60.0  # time in minutes
        self.df_centrosome[['Frame', 'Nuclei', 'Centrosome']].astype(np.int64, inplace=True)
        self.df_nuclei[['Frame', 'Nuclei']].astype(np.int64, inplace=True)

        # merge with nuclei data
        self.merged_df = self.df_centrosome.merge(self.df_nuclei)
        self.merged_df = self.merged_df.drop(['ValidCentroid'], axis=1)

    @staticmethod
    def get_contact_time(df, threshold):
        # get all distances less than a threshold, order them by time and pick the earlier one
        cent_list = df.groupby('Centrosome').size().index
        if len(cent_list) <= 1:
            return 0, 0, 0
        elif len(cent_list) == 2:
            dsf = ImagejPandas.dist_vel_acc_centrosomes(df)
            dsr = dsf[dsf['DistCentr'] <= threshold]

            if dsr.size > 0:
                zeros_df = dsr[dsr['DistCentr'] == 0]
                if zeros_df.size > 0:
                    dsr = zeros_df.set_index('Frame').sort_index()
                    frame = dsr.index[0]
                    time = list(dsr[dsr.index == frame]['Time'])[0]
                    dist = list(dsr[dsr.index == frame]['DistCentr'])[0]
                else:
                    dsr = dsr.set_index('DistCentr').sort_index()
                    frame = list(dsr['Frame'])[0]
                    time = list(dsr['Time'])[0]
                    dist = list(dsf[dsf['Frame'] == frame]['DistCentr'])[0]

                return time, frame, dist
        return None, None, None

    @staticmethod
    def vel_acc_nuclei(df):
        df = df.set_index('Frame').sort_index()
        df['CNx'] = df['NuclX'] - df['CentX']
        df['CNy'] = df['NuclY'] - df['CentY']
        for c_i in df.groupby('Centrosome').groups.keys():
            dc = df[df['Centrosome'] == c_i]
            dc['Dist'] = np.sqrt(dc.CNx ** 2 + dc.CNy ** 2)  # relative to nuclei centroid
            d = dc[['CNx', 'CNy', 'Dist', 'Time']].diff()

            df.loc[df['Centrosome'] == c_i, 'Dist'] = np.sqrt(dc.CNx ** 2 + dc.CNy ** 2)  # relative to nuclei centroid
            df.loc[df['Centrosome'] == c_i, 'Speed'] = d.Dist / d.Time
            df.loc[df['Centrosome'] == c_i, 'Acc'] = d.Dist.diff() / d.Time

        return df.reset_index()

    @staticmethod
    def dist_vel_acc_centrosomes(df):
        cent_list = df.groupby('Centrosome').size().index
        if (len(cent_list) != 2) | (df.groupby(['Frame', 'Time', 'Centrosome']).size().max() > 1):
            # we accept just 1 value per (frame,centrosome)
            ds = pd.DataFrame()
            ds['Frame'] = np.NaN
            ds['Time'] = np.NaN
            ds['DistCentr'] = np.NaN
            ds['SpeedCentr'] = np.NaN
            ds['AccCentr'] = np.NaN
            return ds

        dc = df.set_index(['Frame', 'Time', 'Centrosome']).unstack()
        ddx = dc['CNx'][cent_list[0]] - dc['CNx'][cent_list[1]]
        ddy = dc['CNy'][cent_list[0]] - dc['CNy'][cent_list[1]]
        ds = pd.DataFrame()
        ds['DistCentr'] = np.sqrt(ddx ** 2 + ddy ** 2)

        ds = ds.reset_index().set_index('Frame')
        d = ds.diff()
        ds['SpeedCentr'] = d.DistCentr / d.Time
        ds['AccCentr'] = d.DistCentr.diff() / d.Time

        return ds.reset_index()

    @staticmethod
    def interpolate_data(df):
        if df.groupby(['Frame', 'Time', 'Nuclei', 'Centrosome']).size().max() > 1:
            # we accept just 1 value per (frame,centrosome)
            return df, df.isnull()

        s = df.set_index(['Frame', 'Time', 'Nuclei', 'Centrosome']).sort_index()
        u = s.unstack('Centrosome')
        umask = u.isnull()  # true for interpolated values
        u = u.interpolate(limit=30, limit_direction='backward')

        return u.stack().reset_index(), umask.stack().reset_index()

    @staticmethod
    def join_tracks(df, cn, cm):
        u = df[df['Centrosome'].isin([cn, cm])]
        # search for sup{lny}
        supdn = u.groupby('Centrosome')['Time'].max()
        # get the time of the minimum of the two values
        tc = supdn[supdn == supdn.min()].values[0]

        s = u.set_index(['Time', 'Nuclei', 'Centrosome']).unstack('Centrosome')
        # where are the NaN's?
        nans_are_in = s['Frame'].transpose().isnull().any(axis=1)
        nans_are_in = nans_are_in.keys().values

        mask = s.isnull().stack().reset_index()
        s = u.set_index(['Frame', 'Time', 'Nuclei', 'Centrosome']).unstack('Centrosome')

        if nans_are_in.size > 0:
            if nans_are_in[0] == cm:
                g = s[s.index > tc].transpose().fillna(method='ffill').transpose()
            else:
                g = s[s.index > tc].transpose().fillna(method='bfill').transpose()
            s[s.index > tc] = g
        u = s.stack()

        return u.reset_index(), mask

    def plot_nucleus_dataframe(self, nuclei_df, mask, filename=None):
        nucleus_id = nuclei_df['Nuclei'].min()

        plt.figure(5, figsize=[13, 7])
        plt.clf()
        gs = matplotlib.gridspec.GridSpec(6, 2)
        ax1 = plt.subplot(gs[0:2, 0])
        ax2 = plt.subplot(gs[2, 0])
        ax22 = plt.subplot(gs[2, 1])
        ax3 = plt.subplot(gs[3, 0])
        ax4 = plt.subplot(gs[4, 0])
        ax42 = plt.subplot(gs[4, 1])
        ax5 = plt.subplot(gs[5, 0])

        # get time of contact
        time_contact, frame_contact, dist_contact = self.get_contact_time(nuclei_df, ImagejPandas.DIST_THRESHOLD)

        # plot distance between centrosomes
        dsf = ImagejPandas.dist_vel_acc_centrosomes(nuclei_df)
        dsf = dsf.set_index('Time').sort_index()
        try:
            color = sns.color_palette()[-1]
            tmask = mask.set_index(['Time', 'Centrosome'])['Frame'].unstack().transpose().all()
            dsf['DistCentr'].plot(ax=ax3, label='Dist N%d' % (nucleus_id), marker=None, sharex=True, c=color)
            dsf[tmask]['DistCentr'].plot(ax=ax3, label='Original', marker='o', linewidth=0, sharex=True, c=color)

            ax4.axhline(y=0, color='k', linestyle='--', linewidth=0.1)
            ax42.axhline(y=0, color='k', linestyle='--', linewidth=0.1)
            dsf['SpeedCentr'].plot(ax=ax4, label='Dist N%d' % (nucleus_id), marker=None, sharex=True, c=color)
            dsf[tmask]['SpeedCentr'].plot(ax=ax4, label='Original', marker='o', linewidth=0, sharex=True, c=color)

            dsf['AccCentr'].plot(ax=ax42, label='Dist N%d' % (nucleus_id), marker=None, sharex=True, c=color)
            dsf[tmask]['AccCentr'].plot(ax=ax42, label='Original', marker='o', linewidth=0, sharex=True, c=color)

            if len(dsf[~tmask]['DistCentr']) > 0:
                dsf[~tmask]['DistCentr'].plot(ax=ax3, label='Gen', marker='<', linewidth=0, sharex=True, c=color)
                dsf[~tmask]['SpeedCentr'].plot(ax=ax4, label='Gen', marker='<', linewidth=0, sharex=True, c=color)
                dsf[~tmask]['AccCentr'].plot(ax=ax42, label='Gen', marker='<', linewidth=0, sharex=True, c=color)
        except Exception as e:
            print 'Error printing in try block: ' + str(e)
            pass

        in_yticks = list()
        in_yticks_lbl = list()
        for [(lblCentr), _df], k in zip(nuclei_df.groupby(['Centrosome']),
                                        range(len(nuclei_df.groupby(['Centrosome'])))):
            track = _df.set_index('Time').sort_index()

            tmask = mask[mask['Centrosome'] == lblCentr].set_index(['Time'])['Frame']

            color = sns.color_palette()[k]
            track['Dist'].plot(ax=ax1, label='N%d-C%d' % (nucleus_id, lblCentr), marker=None, sharex=True, c=color)
            track['Speed'].plot(ax=ax2, label='N%d-C%d' % (nucleus_id, lblCentr), sharex=True, c=color)
            track['Acc'].plot(ax=ax22, label='N%d-C%d' % (nucleus_id, lblCentr), sharex=True, c=color)

            ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.1)
            ax22.axhline(y=0, color='k', linestyle='--', linewidth=0.1)
            if len(tmask) > 0:
                if len(track['Dist'][tmask]) > 0:
                    track['Dist'][tmask].plot(ax=ax1, label='Original', marker='o', linewidth=0, sharex=True, c=color)
                if len(track['Dist'][~tmask]) > 0:
                    track['Dist'][~tmask].plot(ax=ax1, label='Gen', marker='<', linewidth=0, sharex=True, c=color)
                    track['Speed'][~tmask].plot(ax=ax2, label='Gen', marker='<', linewidth=0, sharex=True, c=color)
                    track['Acc'][~tmask].plot(ax=ax22, label='Gen', marker='<', linewidth=0, sharex=True, c=color)
            else:
                track['Dist'].plot(ax=ax1, label='Original', marker='o', linewidth=0, sharex=True, c=color)

            # plot time of contact
            if time_contact is not None:
                ax1.axvline(x=time_contact, color='dimgray', linestyle='--')
                ax1.axvline(x=time_contact - ImagejPandas.TIME_BEFORE_CONTACT, color='lightgray', linestyle='--')
                ax2.axvline(x=time_contact, color='dimgray', linestyle='--')
                ax3.axvline(x=time_contact, color='dimgray', linestyle='--')
                ax4.axvline(x=time_contact, color='dimgray', linestyle='--')
                ax5.axvline(x=time_contact, color='dimgray', linestyle='--')

            i_n = track.WhereInNuclei + 3 * k
            in_yticks.append(3 * k)
            in_yticks.append(3 * k + 1)
            in_yticks.append(3 * k + 2)
            in_yticks_lbl.append('Outside')
            in_yticks_lbl.append('Touching')
            in_yticks_lbl.append('Inside')
            i_n.plot(ax=ax5, label='N%d-C%d' % (nucleus_id, lblCentr), marker='o', ylim=[-0.5, 2 * k + 1.5],
                     sharex=True)

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        ax1.set_ylabel('Dist to Nuclei $[\mu m]$')
        ax2.set_ylabel('Speed $[\\frac{\mu m}{min}]$')
        ax22.set_ylabel('Accel $[\\frac{\mu m}{min^2}]$')
        ax3.set_ylabel('Between $[\mu m]$')
        ax4.set_ylabel('Speed $[\\frac{\mu m}{min}]$')
        ax42.set_ylabel('Accel $[\\frac{\mu m}{min^2}]$')
        ax5.set_yticks(in_yticks)
        ax5.set_yticklabels(in_yticks_lbl)

        ax22.set_ylim([-0.5, 0.5])
        ax42.set_ylim([-0.5, 0.5])

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, format='svg')
        plt.close(5)

    @staticmethod
    def process_dataframe(df,
                          nuclei_list=None,
                          centrosome_exclusion_dict=None,
                          centrosome_inclusion_dict=None,
                          joined_tracks=None,
                          max_time_dict=None):

        df = ImagejPandas.vel_acc_nuclei(df)
        # filter non wanted centrosomes
        if centrosome_exclusion_dict is not None:
            for nuId in centrosome_exclusion_dict.keys():
                for centId in centrosome_exclusion_dict[nuId]:
                    df[(df['Nuclei'] == nuId) & (df['Centrosome'] == centId)] = np.NaN
        # include wanted centrosomes
        if centrosome_inclusion_dict is not None:
            for nuId in centrosome_inclusion_dict.keys():
                for centId in centrosome_inclusion_dict[nuId]:
                    df.loc[df['Centrosome'] == centId, 'Nuclei'] = nuId

        df.dropna(how='all', inplace=True)
        df_filtered_nucs, df_masks = pd.DataFrame(), pd.DataFrame()
        for (nucleusID), filtered_nuc_df in df.groupby(['Nuclei']):
            if nucleusID in nuclei_list:
                if (max_time_dict is not None) and (nucleusID in max_time_dict):
                    filtered_nuc_df = filtered_nuc_df[filtered_nuc_df['Time'] <= max_time_dict[nucleusID]]

                filtered_nuc_df, imask = ImagejPandas.interpolate_data(filtered_nuc_df)

                # reset mask variables in each loop
                jmask = None
                # join tracks if asked
                if joined_tracks is not None:
                    if nucleusID in joined_tracks.keys():
                        for centId in joined_tracks[nucleusID]:
                            filtered_nuc_df, jmask = ImagejPandas.join_tracks(filtered_nuc_df, centId[0],
                                                                              centId[1])

                # compute mask as logical AND of joined track mask and interpolated data mask
                im = imask.set_index(['Frame', 'Time', 'Nuclei', 'Centrosome'])
                if jmask is not None:
                    jm = jmask.set_index(['Frame', 'Time', 'Nuclei', 'Centrosome'])
                    mask = (~im & ~jm).reset_index()
                else:
                    mask = (~im).reset_index()

                # compute velocity again with interpolated data
                filtered_nuc_df = ImagejPandas.vel_acc_nuclei(filtered_nuc_df)
                df_filtered_nucs = df_filtered_nucs.append(filtered_nuc_df)
                df_masks = df_masks.append(mask)
        return df_filtered_nucs, df_masks

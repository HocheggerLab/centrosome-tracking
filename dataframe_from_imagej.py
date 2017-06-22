import os
import re

import jinja2 as j2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DataFrameFromImagej(object):
    def __init__(self, filename, stats_df=None):
        self.path_csv = filename
        self.df_csv = pd.read_csv(self.path_csv)
        base_path = os.path.dirname(filename)
        parent_path = os.path.abspath(os.path.join(os.path.dirname(filename), '..'))
        bname = os.path.basename(filename)
        self.fname = re.search('(.+)-table.csv$', bname).group(1)
        self.path_csv_nuclei = '%s/%s-nuclei.csv' % (base_path, self.fname)

        self.df_nuclei_csv = pd.read_csv(self.path_csv_nuclei)
        if stats_df is None:
            stats_df = pd.DataFrame()
        self.stats = stats_df
        self.dt_before_contact = 30
        self.t_per_frame = 5
        self.d_threshold = 1.0  # um before 1 frame of contact
        self.centrosome_replacements = dict()

        # merge with nuclei data
        self.merged_df = self.df_csv.merge(self.df_nuclei_csv)
        self.merged_df = self.merged_df.drop(['WhereInNuclei', 'ValidCentroid'], axis=1)

    @staticmethod
    def get_contact_time(df, threshold):
        # get all distances less than a threshold, order them by time and pick the earlier one
        cent_list = df.groupby('Centrosome').size().index
        if len(cent_list) <= 1:
            return 0, 0, 0
        elif len(cent_list) == 2:
            dsf = DataFrameFromImagej.compute_distance_velocity_acceleration_between_centrosomes(df)
            dsr = dsf[dsf['DistCentr'] < threshold]

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
    def compute_velocity_acceleration(df):
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
    def compute_distance_velocity_acceleration_between_centrosomes(df):
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
        if df.groupby(['Frame', 'Time', 'Centrosome']).size().max() > 1:
            # we accept just 1 value per (frame,centrosome)
            return df, df.isnull()

        u = df.set_index(['Time', 'Centrosome']).unstack('Centrosome')
        umask = u.isnull()  # true for interpolated values
        u = u.interpolate(limit=30, limit_direction='backward')

        return u.stack().reset_index(), umask.stack().reset_index()

    def join_tracks(self, df, cn, cm):
        u = df[df['Centrosome'].isin([cn, cm])]
        # search for sup{lny}
        supdn = u.groupby('Centrosome')['Time'].max()
        # get the time of the minimum of the two values
        tc = supdn[supdn == supdn.min()].values[0]

        s = u.set_index(['Time', 'Centrosome']).unstack('Centrosome')
        # s.index.get_level_values('Centrosome').unique() # centrosome values
        # where are the NaN's?
        nans_are_in = s['Frame'].transpose().isnull().any(axis=1)
        nans_are_in = list(nans_are_in.ix[nans_are_in].keys())[0]

        mask = s.isnull().stack().reset_index()
        # if (d[d['Time'] == tc]['DistCentr'] < self.d_threshold).bool():

        if nans_are_in == cm:
            g = s[s.index > tc].transpose().fillna(method='ffill').transpose()
        else:
            g = s[s.index > tc].transpose().fillna(method='bfill').transpose()
        s[s.index > tc] = g
        u = s.stack()

        return u.reset_index(), mask

    def add_stats(self, _ndf):
        if _ndf.groupby(['Frame', 'Time', 'Centrosome']).size().max() > 1:
            return

        # get time of contact
        time_contact, frame_contact, dist_contact = self.get_contact_time(_ndf, self.d_threshold)
        # hack: get two distances before dt_before_contact and store them in an independent dataframe
        if time_contact is not None:
            frame_before = frame_contact - self.dt_before_contact / self.t_per_frame
            if frame_before < 0:
                frame_before = 0
            dists_before_contact = list(
                _ndf[_ndf['Frame'] == frame_before]['Dist'])
            min_dist = max_dist = time_before = np.NaN
            if len(dists_before_contact) > 0:
                max_dist = max(dists_before_contact)
                min_dist = min(dists_before_contact)
                time_before = _ndf[_ndf['Frame'] == frame_before]['Time'].unique()[0]
        else:
            frame_before = time_before = min_dist = max_dist = np.NaN

        # pick the first item with both non NaN's in the series
        _dfi = _ndf.set_index(['Frame', 'Time', 'Centrosome'])
        _udfi = _dfi['Dist'].unstack()
        _udfi['valid'] = ~ _udfi[_udfi.columns[0]].isnull() & ~ _udfi[_udfi.columns[1]].isnull()
        ini_frame = _udfi[_udfi['valid']].index[0][0]
        ini_time = _udfi[_udfi['valid']].index[0][1]
        ini_dist_min = min(_ndf[_ndf['Frame'] == ini_frame]['Dist'])

        int_time = 100
        int_frame = _ndf[(_ndf['Time'] >= int_time) & (_ndf['Time'] < int_time + 5)]['Frame'].unique()[0]
        int_time = _ndf[_ndf['Frame'] == int_frame]['Time'].unique()[0]
        int_dist_min = min(_ndf[_ndf['Frame'] == int_frame]['Dist'])

        df_rowc = pd.DataFrame({'Tag'   : self.fname,
                                'Nuclei': _ndf['Nuclei'].unique()[0],
                                'Frame' : [frame_contact],
                                'Time'  : [time_contact],
                                'Stat'  : 'Contact',
                                'Type'  : 'Contact',
                                'Dist'  : [dist_contact]})
        df_row1 = pd.DataFrame({'Tag'   : self.fname,
                                'Nuclei': _ndf['Nuclei'].unique()[0],
                                'Frame' : [frame_before],
                                'Time'  : [time_before],
                                'Stat'  : 'Contact',
                                'Type'  : 'Away',
                                'Dist'  : [max_dist]})
        df_row2 = pd.DataFrame({'Tag'   : self.fname,
                                'Nuclei': _ndf['Nuclei'].unique()[0],
                                'Frame' : [frame_before],
                                'Time'  : [time_before],
                                'Stat'  : 'Contact',
                                'Type'  : 'Close',
                                'Dist'  : [min_dist]})
        df_row_ini = pd.DataFrame({'Tag'   : self.fname,
                                   'Nuclei': _ndf['Nuclei'].unique()[0],
                                   'Frame' : [ini_frame],
                                   'Time'  : [ini_time],
                                   'Stat'  : 'Snapshot',
                                   'Type'  : 'Initial',
                                   'Dist'  : [ini_dist_min]})
        df_row_int = pd.DataFrame({'Tag'   : self.fname,
                                   'Nuclei': _ndf['Nuclei'].unique()[0],
                                   'Frame' : [int_time],
                                   'Time'  : [int_frame],
                                   'Stat'  : 'Snapshot',
                                   'Type'  : '100min',
                                   'Dist'  : [int_dist_min]})

        self.stats = self.stats.append(df_row1, ignore_index=True)
        self.stats = self.stats.append(df_row2, ignore_index=True)
        self.stats = self.stats.append(df_rowc, ignore_index=True)
        self.stats = self.stats.append(df_row_ini, ignore_index=True)
        self.stats = self.stats.append(df_row_int, ignore_index=True)

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
        time_contact, frame_contact, dist_contact = self.get_contact_time(nuclei_df, self.d_threshold)

        # plot distance between centrosomes
        dsf = DataFrameFromImagej.compute_distance_velocity_acceleration_between_centrosomes(nuclei_df)
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
                ax1.axvline(x=time_contact - self.dt_before_contact, color='lightgray', linestyle='--')
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

    # TODO: move to mplwidget
    @staticmethod
    def plot_distance_to_nucleus(df, ax, mask=None, filename=None):
        # TODO: missing global parameter use
        nucleus_id = df['Nuclei'].min()

        # re-scale time
        # df['Time'] /= 60.0

        # get time of contact
        time_contact, frame_contact, dist_contact = DataFrameFromImagej.get_contact_time(df, 1.0)  # TODO: param here

        for [(lblCentr), _df], k in zip(df.groupby(['Centrosome']), range(len(df.groupby(['Centrosome'])))):
            track = _df.set_index('Time').sort_index()

            # tmask = mask[mask['Centrosome'] == lblCentr].set_index(['Time'])['Frame']
            tmask = []

            color = sns.color_palette()[k]
            track['Dist'].plot(ax=ax, label='N%d-C%d' % (nucleus_id, lblCentr), marker=None, sharex=True, c=color)

            if len(tmask) > 0:
                if len(track['Dist'][tmask]) > 0:
                    track['Dist'][tmask].plot(ax=ax, label='Original', marker='o', linewidth=0, c=color)
                if len(track['Dist'][~tmask]) > 0:
                    track['Dist'][~tmask].plot(ax=ax, label='Gen', marker='<', linewidth=0, c=color)
            else:
                # track['Dist'].plot(ax=ax, label='Original', marker='o', linewidth=0, c=color)
                track['Dist'].plot(ax=ax, marker='o', linewidth=0, c=color)

            # plot time of contact
            if time_contact is not None:
                ax.axvline(x=time_contact, color='dimgray', linestyle='--')
                ax.axvline(x=time_contact - 30, color='lightgray', linestyle='--')  # TODO: param here

        ax.legend()
        ax.set_ylabel('Dist to Nuclei $[\mu m]$')
        ax.set_xlabel('Time $[min]$')

        if filename is not None:
            plt.axes(ax).savefig(filename, format='svg')

    @staticmethod
    def merge_tracks(df, cn, cm):
        # makes the operation df(Cn) <- df(Cm)
        # that is, to replace all Cn with Cm
        cmnuclei = df.ix[df['Centrosome'] == cm, 'Nuclei'].unique()[0]
        cnnuclei = df.ix[df['Centrosome'] == cn, 'Nuclei'].unique()[0]

        print 'joining %d with %d on nuclei %d' % (cm, cn, cmnuclei)

        df.loc[(df['Nuclei'] == cnnuclei) & (df['Centrosome'] == cm), 'Nuclei'] = df.loc[
            (df['Nuclei'] == cmnuclei) & (df['Centrosome'] == cn), 'Nuclei'].unique()[0]

        df.loc[df['Centrosome'] == cn, 'Centrosome'] = cm

        return df

    @staticmethod
    def process_dataframe(df, experimentTag,
                          nuclei_list=None,
                          centrosome_exclusion_dict=None,
                          centrosome_inclusion_dict=None,
                          centrosome_equivalence_dict=None,
                          joined_tracks=None,
                          max_time_dict=None):

        df = DataFrameFromImagej.compute_velocity_acceleration(df)
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

        df_filtered_nucs = pd.DataFrame()
        # nuclei_data = list()
        for (nucleusID), filtered_nuc_df in df.groupby(['Nuclei']):
            if nucleusID in nuclei_list:
                if (max_time_dict is not None) and (nucleusID in max_time_dict):
                    filtered_nuc_df = filtered_nuc_df[filtered_nuc_df['Time'] <= max_time_dict[nucleusID]]

                # get centrosome list before manipulation
                centrosome_list = sorted(filtered_nuc_df['Centrosome'].unique())

                if centrosome_equivalence_dict is not None:
                    if nucleusID in centrosome_equivalence_dict.keys():
                        centr_repl = list()
                        for cneq in centrosome_equivalence_dict[nucleusID]:
                            min_cm = min(cneq)
                            ceq = dict()
                            for cn in cneq:
                                if cn != min_cm:
                                    filtered_nuc_df = DataFrameFromImagej.merge_tracks(filtered_nuc_df, cn, min_cm)
                                    ceq[cn] = min_cm
                            centr_repl.append(ceq.copy())
                            # DataFrameFromImagej.centrosome_replacements[nucleusID] = centr_repl

                c_tags = ''
                if nucleusID in centrosome_equivalence_dict:
                    for equivs in centrosome_equivalence_dict[nucleusID]:
                        min_c = min(equivs)
                        rest_c = set(equivs) - set([min_c])
                        c_tags += ' C%d was merged with (%s),' % (min_c, ''.join('C%d,' % c for c in rest_c)[0:-1])
                print c_tags[0:-1] + '.'

                # nuc_item = {'filename'               : self.path_csv,
                #             'exp_id'                 : exp_id,
                #             'nuclei_id'              : '%d ' % nucleusID,
                #             'nuclei_centrosomes_tags': ''.join(
                #                 ['C%d, ' % cnId for cnId in centrosome_list])[:-2] + '. ' + c_tags,
                #             'centrosomes_img'        : 'img/%s-nuc_%d.svg' % (exp_id, nucleusID)}
                # nuclei_data.append(nuc_item)

                filtered_nuc_df, imask = DataFrameFromImagej.interpolate_data(filtered_nuc_df)

                # reset mask variables in each loop
                jmask = None
                # join tracks if asked
                if joined_tracks is not None:
                    if nucleusID in joined_tracks.keys():
                        for centId in joined_tracks[nucleusID]:
                            filtered_nuc_df, jmask = DataFrameFromImagej.join_tracks(filtered_nuc_df, centId[0],
                                                                                     centId[1])

                # compute mas as logical AND of joined track mask and interpolated data mask
                im = imask.set_index(['Time', 'Centrosome'])
                try:
                    jm = jmask.set_index(['Time', 'Centrosome'])
                    mask = (~im & ~jm).reset_index()
                except (NameError, AttributeError)as e:  # jm not defined
                    mask = (~im).reset_index()

                # compute velocity again with interpolated data
                filtered_nuc_df = DataFrameFromImagej.compute_velocity_acceleration(filtered_nuc_df)
                # filtered_nuc_df['gap'] = mask
                df_filtered_nucs = df_filtered_nucs.append(filtered_nuc_df)
        return df_filtered_nucs

    def html_centrosomes_report(self, nuclei_list=None,
                                centrosome_exclusion_dict=None,
                                centrosome_inclusion_dict=None,
                                centrosome_equivalence_dict=None,
                                joined_tracks=None,
                                nuclei_equivalence_dict=None,
                                max_time_dict=None):
        htmlout = '<h3>Filename: %s.avi</h3>' % self.fname

        self.filtered_df = self.process_dataframe(centrosome_exclusion_dict, centrosome_inclusion_dict)
        df = self.merged_df
        print self.centrosome_replacements

        for (nucleusID), filtered_nuc_df in self.filtered_df.groupby(['Nuclei']):
            if nucleusID in nuclei_list:
                mask, nuc_item = None, None
                self.plot_nucleus_dataframe(filtered_nuc_df, mask, 'out/%s' % nuc_item['centrosomes_img'])
                self.add_stats(filtered_nuc_df)

        template = """
                {% for nuc_item in nuclei_list %}
                <div class="container">
                    <!--<h3>Filename: {{ nuc_item['filename']}}</h3>-->
                    <ul>
                        <li>Nucleus ID: {{ nuc_item['nuclei_id'] }} ({{ nuc_item['filename']}})</li>
                        <li>Centrosome Tags: {{ nuc_item['nuclei_centrosomes_tags'] }}</li>
                    </ul>
                    <!--<p style="page-break-before: always" >{{ nuc_item['centrosomes_speed_stats'] }}</p>-->
                    <img src="{{ nuc_item['centrosomes_img'] }}">
                </div>
                <div style="page-break-after: always"></div>
                {% endfor %}
            """
        nuclei_data = ''
        templ = j2.Template(template)
        htmlout += templ.render({'nuclei_list': nuclei_data})
        return htmlout, self.filtered_df

import os
import re
import pandas as pd
import numpy as np
import codecs
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import jinja2 as j2
import pdfkit
import time

sns.set_style("white")
print(plt.style.available)


class DataFrameFromImagej(object):
    def __init__(self, filename, stats_df):
        self.path_csv = filename
        self.df_csv = pd.read_csv(self.path_csv)
        self.fname = re.search('lab/(.+?)-table.csv', self.path_csv).group(1)
        self.path_csv_nuclei = re.search('(.*lab/)', self.path_csv).group(1) + self.fname + '-nuclei.csv'

        self.df_nuclei_csv = pd.read_csv(self.path_csv_nuclei)
        self.stats = stats_df

        self.dt_before_contact = 30
        self.t_per_frame = 5
        self.d_threshold = 3.5  # um before 1 frame of contact
        self.centrosome_replacements = dict()

        if not os.path.exists('out/img'):
            os.makedirs('out/img')

    @staticmethod
    def get_contact_time(df, threshold):
        # get all distances less than a threshold, order them by time and pick the earlier one
        cent_list = df.groupby('Centrosome').size().index
        if len(cent_list) <= 1:
            return 0, 0, 0
        elif len(cent_list) == 2:
            dsf = DataFrameFromImagej.compute_distance_between_centrosomes(df)
            dsr = dsf[dsf['DistCentr'] == 0]

            if dsr.size > 0:
                dsr = dsr.set_index('DistCentr').sort_index()
                frame = list(dsr['Frame'])[0]
                time = list(dsr['Time'])[0]
                dist = list(dsf[dsf['Frame'] == frame]['DistCentr'])[0]

                return time, frame, dist

        return None, None, None

    @staticmethod
    def compute_velocity(df):
        df = df.set_index('Frame').sort_index()
        df['CNx'] = df['NuclX'] - df['CentX']
        df['CNy'] = df['NuclY'] - df['CentY']
        for c_i in df.groupby('Centrosome').groups.keys():
            dc = df[df['Centrosome'] == c_i]
            d = dc[['CNx', 'CNy', 'Time']].diff()

            d['Dist'] = np.sqrt(d.CNx ** 2 + d.CNy ** 2)  # relative to nuclei centroid

            df.loc[df['Centrosome'] == c_i, 'Dist'] = np.sqrt(dc.CNx ** 2 + dc.CNy ** 2)  # relative to nuclei centroid
            df.loc[df['Centrosome'] == c_i, 'Speed'] = d.Dist / d.Time

        return df.reset_index()

    @staticmethod
    def compute_distance_between_centrosomes(df):
        cent_list = df.groupby('Centrosome').size().index
        if (len(cent_list) != 2) | (df.groupby(['Frame', 'Time', 'Centrosome']).size().max() > 1):
            # we accept just 1 value per (frame,centrosome)
            ds = pd.DataFrame()
            ds['Frame'] = np.NaN
            ds['Time'] = np.NaN
            ds['DistCentr'] = np.NaN
            return ds

        dc = df.set_index(['Frame', 'Time', 'Centrosome']).unstack()
        ddx = dc['CNx'][cent_list[0]] - dc['CNx'][cent_list[1]]
        ddy = dc['CNy'][cent_list[0]] - dc['CNy'][cent_list[1]]
        ds = pd.DataFrame()
        ds['DistCentr'] = np.sqrt(ddx ** 2 + ddy ** 2)

        return ds.reset_index()

    @staticmethod
    def interpolate_data(df):
        if df.groupby(['Frame', 'Time', 'Centrosome']).size().max() > 1:
            # we accept just 1 value per (frame,centrosome)
            return df, df.isnull()
        else:
            u = df.set_index(['Time', 'Centrosome']).unstack('Centrosome')
            umask = u.isnull()  # true for interpolated values
            u = u.interpolate()
            # u[umask][['CNx', 'CNy', 'Dist', 'Speed']].plot(marker='x')  # interpolated
            # u[~umask][['CNx', 'CNy', 'Dist', 'Speed']].plot(marker='o')
            # u[['CNx', 'CNy', 'Dist', 'Speed']].plot(marker='o')

            return u.stack().reset_index(), umask.stack().reset_index()

    def join_tracks(self, df, cn, cm):
        u = df[df['Centrosome'].isin([cn, cm])]
        # search for sup{lny}
        supdn = u.groupby('Centrosome')['Time'].max()
        # get the time of the minimum of the two values
        tc = supdn[supdn == supdn.min()].values[0]
        d = DataFrameFromImagej.compute_distance_between_centrosomes(u)

        s = u.set_index(['Time', 'Centrosome']).unstack('Centrosome')

        mask = s.isnull().stack().reset_index()
        if (d[d['Time'] == tc]['DistCentr'] < self.d_threshold).bool():
            s[s.index > tc] = s[s.index > tc].transpose().fillna(method='ffill').transpose()
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

        df_rowc = pd.DataFrame({'Tag': self.fname,
                                'Nuclei': _ndf['Nuclei'].unique()[0],
                                'Frame': [frame_contact],
                                'Time': [time_contact],
                                'Stat': 'Contact',
                                'Type': 'Contact',
                                'Dist': [dist_contact]})
        df_row1 = pd.DataFrame({'Tag': self.fname,
                                'Nuclei': _ndf['Nuclei'].unique()[0],
                                'Frame': [frame_before],
                                'Time': [time_before],
                                'Stat': 'Contact',
                                'Type': 'Away',
                                'Dist': [max_dist]})
        df_row2 = pd.DataFrame({'Tag': self.fname,
                                'Nuclei': _ndf['Nuclei'].unique()[0],
                                'Frame': [frame_before],
                                'Time': [time_before],
                                'Stat': 'Contact',
                                'Type': 'Close',
                                'Dist': [min_dist]})
        df_row_ini = pd.DataFrame({'Tag': self.fname,
                                   'Nuclei': _ndf['Nuclei'].unique()[0],
                                   'Frame': [ini_frame],
                                   'Time': [ini_time],
                                   'Stat': 'Snapshot',
                                   'Type': 'Initial',
                                   'Dist': [ini_dist_min]})
        df_row_int = pd.DataFrame({'Tag': self.fname,
                                   'Nuclei': _ndf['Nuclei'].unique()[0],
                                   'Frame': [int_time],
                                   'Time': [int_frame],
                                   'Stat': 'Snapshot',
                                   'Type': '100min',
                                   'Dist': [int_dist_min]})

        self.stats = self.stats.append(df_row1, ignore_index=True)
        self.stats = self.stats.append(df_row2, ignore_index=True)
        self.stats = self.stats.append(df_rowc, ignore_index=True)
        self.stats = self.stats.append(df_row_ini, ignore_index=True)
        self.stats = self.stats.append(df_row_int, ignore_index=True)

    def plot_nucleus_dataframe(self, nuclei_df, mask, filename=None):
        nucleus_id = nuclei_df['Nuclei'].min()

        plt.figure(5)
        plt.clf()
        gs = matplotlib.gridspec.GridSpec(4, 1)
        ax1 = plt.subplot(gs[0:2, 0])
        ax2 = plt.subplot(gs[2, 0])
        ax3 = plt.subplot(gs[3, 0])

        # get time of contact
        time_contact, frame_contact, dist_contact = self.get_contact_time(nuclei_df, self.d_threshold)

        # plot distance between centrosomes
        dsf = DataFrameFromImagej.compute_distance_between_centrosomes(nuclei_df)
        dsf = dsf.set_index('Time').sort_index()
        try:
            color = sns.color_palette()[-1]
            tmask = mask.set_index(['Time', 'Centrosome'])['CNx'].unstack().transpose().all()
            dsf['DistCentr'].plot(ax=ax3, label='Dist N%d' % (nucleus_id), marker=None, sharex=True, c=color)
            dsf[tmask]['DistCentr'].plot(ax=ax3, label='Original', marker='o', linewidth=0, sharex=True,
                                         c=color)
            dsf[~tmask]['DistCentr'].plot(ax=ax3, label='Gen', marker='<', linewidth=0, sharex=True,
                                          c=color)
            ax3.axvline(x=time_contact, color='dimgray', linestyle='--')
        except:
            pass

        for [(lblCentr), _df], k in zip(nuclei_df.groupby(['Centrosome']),
                                        range(len(nuclei_df.groupby(['Centrosome'])))):
            track = _df.set_index('Time').sort_index()

            tmask = mask[mask['Centrosome'] == lblCentr].set_index(['Time'])['CNx']

            color = sns.color_palette()[k]
            track['Dist'].plot(ax=ax1, label='N%d-C%d' % (nucleus_id, lblCentr), marker=None, sharex=True, c=color)
            track['Speed'].plot(ax=ax2, label='N%d-C%d' % (nucleus_id, lblCentr), sharex=True, c=color)

            try:
                track['Dist'][tmask].plot(ax=ax1, label='Original', marker='o', linewidth=0, sharex=True, c=color)
                track['Dist'][~tmask].plot(ax=ax1, label='Gen', marker='<', linewidth=0, sharex=True, c=color)
                track['Speed'][~tmask].plot(ax=ax2, label='Gen', marker='<', linewidth=0, sharex=True, c=color)
            except:
                pass

            # plot time of contact
            if time_contact is not None:
                ax1.axvline(x=time_contact, color='dimgray', linestyle='--')
                ax1.axvline(x=time_contact - self.dt_before_contact, color='lightgray', linestyle='--')

        ax1.legend()
        ax2.legend()
        ax3.legend()

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, format='svg')
        plt.close(5)

    @staticmethod
    def merge_tracks(df, cn, cm):
        # makes the operation df(Cn) <- df(Cm)
        # that is, to replace all Cn with Cm
        cmnuclei = df.ix[df['Centrosome'] == cm, 'Nuclei'].unique()[0]
        cnnuclei = df.ix[df['Centrosome'] == cn, 'Nuclei'].unique()[0]

        print 'joining %d with %d on nuclei %d' % (cm, cn, cmnuclei)

        df.loc[(df['Nuclei'] == cnnuclei) & (df['Centrosome'] == cm), 'Nuclei'] = df.loc[
            (df['Nuclei'] == cmnuclei) & (df['Centrosome'] == cn), 'Nuclei'].unique()[0]

        df.ix[df['Centrosome'] == cn, 'Centrosome'] = cm

        return df

    def html_centrosomes_report(self, nuclei_list=None,
                                centrosome_exclusion_dict=None,
                                centrosome_inclusion_dict=None,
                                centrosome_equivalence_dict=None,
                                joined_tracks=None,
                                nuclei_equivalence_dict=None,
                                max_time_dict=None):
        htmlout = '<h3>Filename: %s.avi</h3>' % self.fname

        # apply filters
        df = self.df_csv
        # drop things we won't need
        df = df.drop(['InsideNuclei', 'ValidCentroid'], axis=1)

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

        if centrosome_equivalence_dict is not None:
            for nuId in centrosome_equivalence_dict.keys():
                centr_repl = list()
                for cneq in centrosome_equivalence_dict[nuId]:
                    min_cm = min(cneq)
                    ceq = dict()
                    for cn in cneq:
                        if cn != min_cm:
                            df = self.merge_tracks(df, cn, min_cm)
                            ceq[cn] = min_cm
                    centr_repl.append(ceq.copy())
                self.centrosome_replacements[nuId] = centr_repl

        # merge with nuclei data
        df = df.merge(self.df_nuclei_csv)

        print self.centrosome_replacements

        # re-scale time
        df['Time'] /= 60.0

        nuclei_data = list()
        for (nucleusID), filtered_nuc_df in df.groupby(['Nuclei']):
            if nucleusID in nuclei_list:
                if (max_time_dict is not None) and (nucleusID in max_time_dict):
                    filtered_nuc_df = filtered_nuc_df[filtered_nuc_df['Time'] <= max_time_dict[nucleusID]]
                exp_id = re.search('centr-(.+?)-table.csv', self.path_csv).group(1)

                c_tags = ''
                if nucleusID in self.centrosome_replacements.keys():
                    for equivs in self.centrosome_replacements[nucleusID]:
                        centrs = set(equivs.keys())
                        for ct in centrs:
                            c_tags += 'C%d was merged with (' % ct
                            for set_r in self.centrosome_replacements[nucleusID]:
                                for r in set_r:
                                    c_tags += 'C%d,' % r
                                c_tags = c_tags[:-1] + ') '
                        print c_tags

                nuc_item = {'filename': self.path_csv,
                            'exp_id': exp_id,
                            'nuclei_id': '%d ' % nucleusID,
                            'nuclei_centrosomes_tags': ''.join(
                                ['C%d, ' % cnId for cnId in sorted(filtered_nuc_df['Centrosome'].unique())])[:-2]
                                                       + '. ' + c_tags,
                            'centrosomes_img': 'img/%s-nuc_%d.svg' % (exp_id, nucleusID)}
                nuclei_data.append(nuc_item)

                filtered_nuc_df = self.compute_velocity(filtered_nuc_df)

                # join tracks if asked
                if joined_tracks is not None:
                    if nucleusID in joined_tracks.keys():
                        for centId in joined_tracks[nucleusID]:
                            filtered_nuc_df, jmask = self.join_tracks(filtered_nuc_df, centId[0], centId[1])

                filtered_nuc_df, imask = self.interpolate_data(filtered_nuc_df)

                # compute mas as logical AND of joined track mask and interpolated data mask
                im = imask.set_index(['Time', 'Centrosome'])
                try:
                    jm = jmask.set_index(['Time', 'Centrosome'])
                    mask = (~im & ~jm).reset_index()
                except NameError:  # jm not defined
                    mask = (~im).reset_index()

                # compute velocity again with interpolated data
                filtered_nuc_df = self.compute_velocity(filtered_nuc_df)
                filtered_nuc_df = filtered_nuc_df.drop(['NuclX', 'NuclY', 'CentX', 'CentY'], axis=1)

                self.plot_nucleus_dataframe(filtered_nuc_df, mask, 'out/%s' % nuc_item['centrosomes_img'])
                self.add_stats(filtered_nuc_df)

        template = """
                {% for nuc_item in nuclei_list %}
                <div class="container">
                    <!--<h3>Filename: {{ nuc_item['filename']}}</h3>-->
                    <ul>
                        <li>Nucleus ID: {{ nuc_item['nuclei_id'] }}</li>
                        <li>Centrosome Tags: {{ nuc_item['nuclei_centrosomes_tags'] }}</li>
                    </ul>
                    <!--<p style="page-break-before: always" >{{ nuc_item['centrosomes_speed_stats'] }}</p>-->
                    <img src="{{ nuc_item['centrosomes_img'] }}">
                </div>
                <div style="page-break-after: always"></div>
                {% endfor %}
            """
        templ = j2.Template(template)
        htmlout += templ.render({'nuclei_list': nuclei_data})
        return htmlout


def box_beeswarm_plot(data, filename=None, ylim=None):
    if len(data) > 0:
        plt.figure(10)
        data['Dist'] = data.Dist.astype(np.float64)  # fixes a bug of seaborn
        sns.boxplot(data=data, y='Dist', x='Type')
        ax = sns.swarmplot(data=data, y='Dist', x='Type')
        for i, artist in enumerate(ax.artists):
            artist.set_facecolor('None')

        if ylim is not None:
            _axis = ax.axis()
            ax.axis([_axis[0], _axis[1], ylim[0], ylim[1]])

        # render text
        cat = data['Type'].unique()
        for c, x in zip(cat, range(len(cat) + 1)):
            d = data[data['Type'] == c]['Dist']
            if len(d) > 0:
                _max_y = ax.axis()[3]
                count = d.count()
                mean = d.mean()
                ax.text(x, _max_y - 2.0, '$\mu=%0.3f$' % mean, ha='center')
                ax.text(x, _max_y - 4.0, '$n=%d$' % count, ha='center')

        if filename is not None:
            plt.savefig('out/img/%s.svg' % filename, format='svg')
        plt.close(10)


if __name__ == '__main__':
    pc_to_process = {'path': '/Users/Fabio/lab/PC/data/',
                     'files': [{
                         'name': 'centr-pc-0-table.csv',
                         'nuclei_list': [2, 3],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {2: [[200, 201]]}
                     }, {
                         'name': 'centr-pc-1-table.csv',
                         'nuclei_list': [1, 2, 4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {4: [500]},
                         'centrosome_exclusion_dict': {4: [402]},
                         'centrosome_equivalence_dict': {1: [[100, 102, 103]]},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-3-table.csv',
                         'nuclei_list': [5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-4-table.csv',
                         # 'nuclei_list': [7],
                         'nuclei_list': [],
                         'max_time_dict': {7: 115},
                         'centrosome_inclusion_dict': {7: [0]},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                         # }, {
                         #     'name': 'centr-pc-5-table.csv',
                         #     'nuclei_list': [],
                         #     'max_time_dict': {},
                         #     'centrosome_inclusion_dict': {},
                         #     'centrosome_exclusion_dict': {},
                         #     'centrosome_equivalence_dict': {}
                         # 'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-10-table.csv',
                         'nuclei_list': [3],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-12-table.csv',
                         'nuclei_list': [1],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {1: [102]},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-14-table.csv',
                         'nuclei_list': [2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {2: [205]},
                         'centrosome_equivalence_dict': {2: [[200, 203], [201, 202, 204]]},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-17-table.csv',
                         'nuclei_list': [1, 2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {1: [104, 105]},
                         'centrosome_equivalence_dict': {1: [[100, 102, 103]]},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-18-table.csv',
                         'nuclei_list': [3, 4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {4: [402, 403]},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-200-table.csv',
                         'nuclei_list': [1, 5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-201-table.csv',
                         'nuclei_list': [10],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {10: [[1000, 1002]]},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-202-table.csv',
                         'nuclei_list': [1, 5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {5: [[500, 502]]},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-203-table.csv',
                         'nuclei_list': [6],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-204-table.csv',
                         'nuclei_list': [7],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-205-table.csv',
                         'nuclei_list': [4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-207-table.csv',
                         'nuclei_list': [7],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-209-table.csv',
                         'nuclei_list': [5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-210-table.csv',
                         'nuclei_list': [3, 6],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-211-table.csv',
                         'nuclei_list': [3],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-212-table.csv',
                         'nuclei_list': [4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {4: [[400, 402]]},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-213-table.csv',
                         'nuclei_list': [1, 2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-214-table.csv',
                         'nuclei_list': [5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {5: [600]},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {5: [[500, 502]]},
                         'joined_tracks': {2: [[200, 201]]}
                         # }, {
                         #     'name': 'centr-pc-216-table.csv',
                         #     'nuclei_list': [],
                         #     'max_time_dict': {},
                         #     'centrosome_inclusion_dict': {},
                         #     'centrosome_exclusion_dict': {},
                         # 'centrosome_equivalence_dict': {}
                         # 'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-218-table.csv',
                         'nuclei_list': [4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {4: [[400, 402]]},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-219-table.csv',
                         'nuclei_list': [4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {4: [[400, 402], [401, 403]]},
                         'joined_tracks': {}
                         # }, {
                         #     'name': 'centr-pc-220-table.csv',
                         #     'nuclei_list': [],
                         #     'max_time_dict': {},
                         #     'centrosome_inclusion_dict': {},
                         #     'centrosome_exclusion_dict': {},
                         # 'centrosome_equivalence_dict': {}
                         # 'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-221-table.csv',
                         'nuclei_list': [2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {2: [[201, 202]]},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-222-table.csv',
                         'nuclei_list': [2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {2: [[201, 202, 203]]},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-223-table.csv',
                         'nuclei_list': [5, 6, 7],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {5: [[500, 502]], 7: [[701, 702, 703]]},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-224-table.csv',
                         'nuclei_list': [4, 6],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }
                     ]}

    _yl = [-5, 35]  # limits in y axis for boxplots
    html = '<h3></h3>'
    stats = pd.DataFrame()
    cond = pc_to_process
    for f in cond['files'][0:]:
        print f['name']
        dfij = DataFrameFromImagej(cond['path'] + f['name'], stats_df=stats)
        html += dfij.html_centrosomes_report(nuclei_list=f['nuclei_list'], max_time_dict=f['max_time_dict'],
                                             centrosome_inclusion_dict=f['centrosome_inclusion_dict'],
                                             centrosome_exclusion_dict=f['centrosome_exclusion_dict'],
                                             centrosome_equivalence_dict=f['centrosome_equivalence_dict'],
                                             joined_tracks=f['joined_tracks'])

        stats = dfij.stats
    html_pc = html

    sdata = stats[(stats['Stat'] == 'Snapshot') & (stats['Dist'].notnull())][['Dist', 'Type']]
    box_beeswarm_plot(sdata, filename='beeswarm_boxplot_pc_snapshot', ylim=_yl)

    sdata = stats[(stats['Stat'] == 'Contact') & (stats['Dist'].notnull())][['Dist', 'Type']]
    box_beeswarm_plot(sdata, filename='beeswarm_boxplot_pc_contact', ylim=_yl)

    dyndic1_to_process = {'path': '/Users/Fabio/lab/Dyn/data/',
                          'files': [{
                              'name': 'centr-dyn-101-table.csv',
                              # 'nuclei_list': [4],
                              'nuclei_list': [],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {4: [0, 1]},
                              'centrosome_exclusion_dict': {4: [400]},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {}
                              # }, {
                              #     'name': 'centr-dyn-102-table.csv',
                              #     'nuclei_list': [3],
                              #     'max_time_dict': {},
                              #     'centrosome_inclusion_dict': {},
                              #     'centrosome_exclusion_dict': {},
                              #     'centrosome_equivalence_dict': {3:[]}
                              # 'joined_tracks': {}
                              # }, {
                              #     'name': 'centr-dyn-103-table.csv',
                              #     'nuclei_list': [],
                              #     'max_time_dict': {},
                              #     'centrosome_inclusion_dict': {},
                              #     'centrosome_exclusion_dict': {},
                              #     'centrosome_equivalence_dict': {}
                              # 'joined_tracks': {}
                              # }, {
                              #     'name': 'centr-dyn-104-table.csv',
                              #     'nuclei_list': [],
                              #     'max_time_dict': {},
                              #     'centrosome_inclusion_dict': {},
                              #     'centrosome_exclusion_dict': {},
                              #     'centrosome_equivalence_dict': {}
                              # 'joined_tracks': {}
                          }, {
                              'name': 'centr-dyn-105-table.csv',
                              'nuclei_list': [4],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {4: [402, 403]},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {}
                          }, {
                              'name': 'centr-dyn-107-table.csv',
                              'nuclei_list': [8],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {8: [0, 3]},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {}
                          }, {
                              'name': 'centr-dyn-109-table.csv',
                              'nuclei_list': [4, 5],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {4: [200]},
                              'centrosome_exclusion_dict': {4: [401], 5: [502]},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {}
                              # }, {
                              #     'name': 'centr-dyn-110-table.csv',
                              #     'nuclei_list': [],
                              #     'max_time_dict': {},
                              #     'centrosome_inclusion_dict': {},
                              #     'centrosome_exclusion_dict': {},
                              #     'centrosome_equivalence_dict': {}
                              # 'joined_tracks': {}
                          }, {
                              'name': 'centr-dyn-112-table.csv',
                              'nuclei_list': [3],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {3: [200]},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {}
                          }, {
                              'name': 'centr-dyn-203-table.csv',
                              'nuclei_list': [2],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {2: [200, 203, 204, 205, 206, 207]},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {}
                          }, {
                              'name': 'centr-dyn-204-table.csv',
                              'nuclei_list': [3],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {3: [101]},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {}
                          }, {
                              'name': 'centr-dyn-205-table.csv',
                              # 'nuclei_list': [4],
                              'nuclei_list': [],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {4: [0, 301]},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {}
                          }, {
                              'name': 'centr-dyn-207-table.csv',
                              # 'nuclei_list': [2, 3],
                              'nuclei_list': [3],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {2: [0, 301]},
                              'centrosome_exclusion_dict': {2: [200]},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {}
                          }, {
                              'name': 'centr-dyn-208-table.csv',
                              'nuclei_list': [1],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {}
                          }, {
                              'name': 'centr-dyn-209-table.csv',
                              'nuclei_list': [3],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {}
                          }, {
                              'name': 'centr-dyn-210-table.csv',
                              'nuclei_list': [2, 3, 4],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {4: [402]},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {}
                          }, {
                              'name': 'centr-dyn-213-table.csv',
                              'nuclei_list': [4],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {}
                          },
                          ]}

    dyncdk1as_to_process = {'path': '/Users/Fabio/lab/DynCDK1as/data/',
                            'files': [{
                                'name': 'centr-dyncdk1as-002-table.csv',
                                'nuclei_list': [1, 2, 3, 6, 9],
                                'max_time_dict': {},
                                'centrosome_inclusion_dict': {},
                                'centrosome_exclusion_dict': {6: [602, 603], 9: [903]},
                                'centrosome_equivalence_dict': {1: [[100, 102]], 6: [[600, 604, 605]],
                                                                9: [[901, 902]]},
                                'joined_tracks': {}
                            }, {
                                'name': 'centr-dyncdk1as-003-table.csv',
                                'nuclei_list': [],
                                'max_time_dict': {},
                                'centrosome_inclusion_dict': {},
                                'centrosome_exclusion_dict': {},
                                'centrosome_equivalence_dict': {},
                                'joined_tracks': {}
                            }, {
                                'name': 'centr-dyncdk1as-005-table.csv',
                                'nuclei_list': [2],
                                'max_time_dict': {},
                                'centrosome_inclusion_dict': {},
                                'centrosome_exclusion_dict': {},
                                'centrosome_equivalence_dict': {},
                                'joined_tracks': {}
                            }, {
                                'name': 'centr-dyncdk1as-007-table.csv',
                                'nuclei_list': [6],
                                'max_time_dict': {},
                                'centrosome_inclusion_dict': {},
                                'centrosome_exclusion_dict': {},
                                'centrosome_equivalence_dict': {},
                                'joined_tracks': {}
                            }, {
                                'name': 'centr-dyncdk1as-008-table.csv',
                                'nuclei_list': [6, 7],
                                'max_time_dict': {},
                                'centrosome_inclusion_dict': {},
                                'centrosome_exclusion_dict': {},
                                'centrosome_equivalence_dict': {},
                                'joined_tracks': {}
                            }, {
                                'name': 'centr-dyncdk1as-011-table.csv',
                                'nuclei_list': [4, 5],
                                'max_time_dict': {},
                                'centrosome_inclusion_dict': {},
                                'centrosome_exclusion_dict': {},
                                'centrosome_equivalence_dict': {4: [[400, 401]]},
                                'joined_tracks': {}
                            },
                            ]}

    html = '<h3></h3>'
    stats = pd.DataFrame()
    merged_conditions = [dyndic1_to_process, dyncdk1as_to_process]
    for cond in merged_conditions:
        for f in cond['files'][0:]:
            print f['name']
            dfij = DataFrameFromImagej(cond['path'] + f['name'], stats_df=stats)
            html += dfij.html_centrosomes_report(nuclei_list=f['nuclei_list'], max_time_dict=f['max_time_dict'],
                                                 centrosome_inclusion_dict=f['centrosome_inclusion_dict'],
                                                 centrosome_exclusion_dict=f['centrosome_exclusion_dict'],
                                                 centrosome_equivalence_dict=f['centrosome_equivalence_dict'],
                                                 joined_tracks=f['joined_tracks'])
            stats = dfij.stats
    html_dyndic1 = html

    sdata = stats[(stats['Stat'] == 'Snapshot') & (stats['Dist'].notnull())][['Dist', 'Type']]
    box_beeswarm_plot(sdata, filename='beeswarm_boxplot_dyndic1_snapshot', ylim=_yl)

    sdata = stats[(stats['Stat'] == 'Contact') & (stats['Dist'].notnull())][['Dist', 'Type']]
    box_beeswarm_plot(sdata, filename='beeswarm_boxplot_dyndic1_contact', ylim=_yl)

    master_template = """<!DOCTYPE html>
            <html>
            <head lang="en">
                <meta charset="UTF-8">
                <title>{{ title }}</title>
            </head>
            <body>
                <h1>Centrosome Data Report - {{report_date}}</h1>

                <h2>Condition: Positive Control ({{pc_n}} tracks)</h2>
                <h2>Brief</h2>
                <div class="container">
                <h3>Distance from nuclei center at time of contact</h3>
                    <img src="img/beeswarm_boxplot_pc_contact.svg">
                <h3>Distance from nuclei center at initial time and 100 mins after</h3>
                    <img src="img/beeswarm_boxplot_pc_snapshot.svg">
                </div>

                <h2>Condition: DynH1, DIC1 & DynCDK1as ({{dyndic1_n}} tracks)</h2>
                <h2>Brief</h2>
                <div class="container">
                <h3>Distance from nuclei center at time of contact</h3>
                    <img src="img/beeswarm_boxplot_dyndic1_contact.svg">
                <h3>Distance from nuclei center at initial time and 100 mins after</h3>
                    <img src="img/beeswarm_boxplot_dyndic1_snapshot.svg">
                </div>

                </br>
                <h2>Condition: Positive Control ({{pc_n}} tracks)</h2>
                <h2>Track Detail</h2>
                {{ nuclei_data_pc_html }}


                <h2>Condition: DynH1, DIC1 & DynCDK1as ({{dyndic1_n}} tracks)</h2>
                <h2>Track Detail</h2>
                {{ nuclei_data_dyndic1_html }}
            </body>
            </html>
            """
    templ = j2.Template(master_template)
    pc_tracks = len(np.concatenate([np.array(n['nuclei_list']) for n in pc_to_process['files']]))
    dyn_tracks = len(np.concatenate([np.array(n['nuclei_list']) for n in dyndic1_to_process['files']]))
    dyn_tracks += len(np.concatenate([np.array(n['nuclei_list']) for n in dyncdk1as_to_process['files']]))
    htmlout = templ.render(
        {'title': 'Centrosomes report',
         'report_date': time.strftime("%d/%m/%Y"), 'nuclei_data_pc_html': html_pc,
         'pc_n': pc_tracks,
         'nuclei_data_dyndic1_html': html_dyndic1,
         'dyndic1_n': dyn_tracks})

    with codecs.open('out/index.html', "w", "utf-8") as text_file:
        text_file.write(htmlout)

        # pdfkit.from_file('out/index.html', 'out/report.pdf')

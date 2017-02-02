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
        self.d_threshold = 1.0  # um before 1 frame of contact
        self.centrosome_replacements = dict()

    @staticmethod
    def get_contact_time(df, threshold):
        # get all distances less than a threshold, order them by time and pick the earlier one
        cent_list = df.groupby('Centrosome').size().index
        if len(cent_list) <= 1:
            return 0, 0, 0
        elif len(cent_list) == 2:
            dsf = DataFrameFromImagej.compute_distance_between_centrosomes(df)
            dsr = dsf[dsf['DistCentr'] < threshold * 10]

            if dsr.size > 0:
                dsr = dsr.set_index('DistCentr').sort_index()
                frame = list(dsr['Frame'])[0] + 1  # take next frame
                time = list(df[df['Frame'] == frame]['Time'])[0]
                dist = list(dsf[dsf['Frame'] == frame]['DistCentr'])[0]

                if not np.isnan(dist):
                    frame -= 1
                    time = list(df[df['Frame'] == frame]['Time'])[0]
                    dist = list(dsf[dsf['Frame'] == frame]['DistCentr'])[0]

                    if dist > threshold:
                        frame = time = dist = None

                return time, frame, dist

        return None, None, None

    @staticmethod
    def compute_velocity(df):
        df = df.set_index('Frame').sort_index()
        df['CNx'] = df.NuclX - df.CentX
        df['CNy'] = df.NuclY - df.CentY
        df = df.drop(['NuclX', 'NuclY', 'CentX', 'CentY'], axis=1)
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
        if len(cent_list) != 2:
            ds = pd.DataFrame()
            ds['Frame'] = np.NaN
            ds['Time'] = np.NaN
            ds['DistCentr'] = np.NaN
            return ds
        # we accept just 1 value per (frame,centrosome)
        if df.groupby(['Frame', 'Time', 'Centrosome']).size().max() > 1:
            ds = pd.DataFrame()
            ds['Frame'] = np.NaN
            ds['Time'] = np.NaN
            ds['DistCentr'] = np.NaN
            return ds

        dc = df.set_index(['Frame', 'Time', 'Centrosome']).unstack()
        ddx = dc.CNx[cent_list[0]] - dc.CNx[cent_list[1]]
        ddy = dc.CNy[cent_list[0]] - dc.CNy[cent_list[1]]
        ds = pd.DataFrame()
        ds['DistCentr'] = np.sqrt(ddx ** 2 + ddy ** 2)

        return ds.reset_index()

    def add_stats(self, _ndf):
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

        ini_frame = _ndf.set_index('Frame').sort_index().index[0]
        ini_time = _ndf[_ndf['Frame'] == ini_frame]['Time'].unique()[0]
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

    def plot_nucleus_dataframe(self, nuclei_df, filename=None):
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
            dsf.DistCentr.plot(ax=ax3, label='Dist N%d' % (nucleus_id), marker='o', sharex=True)
            ax3.axvline(x=time_contact, color='dimgray', linestyle='--')
        except:
            pass

        for [(lblCentr), _df], k in zip(nuclei_df.groupby(['Centrosome']),
                                        range(len(nuclei_df.groupby(['Centrosome'])))):
            track = _df.set_index('Time').sort_index()

            track.Dist.plot(ax=ax1, label='N%d-C%d' % (nucleus_id, lblCentr), marker='o', sharex=True)
            track.Speed.plot(ax=ax2, label='N%d-C%d' % (nucleus_id, lblCentr), sharex=True)

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
    def join_tracks(df, cn, cm):
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
                                nuclei_equivalence_dict=None,
                                max_time_dict=None):
        htmlout = '<h3>Filename: %s.avi</h3>' % self.fname

        # apply filters
        df = self.df_csv
        # drop thing we won't need
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
                            df = self.join_tracks(df, cn, min_cm)
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
                self.plot_nucleus_dataframe(filtered_nuc_df, 'out/%s' % nuc_item['centrosomes_img'])
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


if __name__ == '__main__':

    pc_to_process = {'path': '/Users/Fabio/lab/PC/data/',
                     'files': [{
                         'name': 'centr-pc-0-table.csv',
                         'nuclei_list': [2, 3],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-1-table.csv',
                         'nuclei_list': [1, 2, 4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {4: [500]},
                         'centrosome_exclusion_dict': {4: [402]},
                         'centrosome_equivalence_dict': {1: [[100, 102, 103]]}
                     }, {
                         'name': 'centr-pc-3-table.csv',
                         'nuclei_list': [5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-4-table.csv',
                         # 'nuclei_list': [7],
                         'nuclei_list': [],
                         'max_time_dict': {7: 115},
                         'centrosome_inclusion_dict': {7: [0]},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                         # }, {
                         #     'name': 'centr-pc-5-table.csv',
                         #     'nuclei_list': [],
                         #     'max_time_dict': {},
                         #     'centrosome_inclusion_dict': {},
                         #     'centrosome_exclusion_dict': {},
                         #     'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-10-table.csv',
                         'nuclei_list': [3],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-12-table.csv',
                         'nuclei_list': [1],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {1: [102]},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-14-table.csv',
                         'nuclei_list': [2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {2: [205]},
                         'centrosome_equivalence_dict': {2: [[200, 203], [201, 202, 204]]}
                     }, {
                         'name': 'centr-pc-17-table.csv',
                         'nuclei_list': [1, 2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {1: [104, 105]},
                         'centrosome_equivalence_dict': {1: [[100, 102, 103]]}
                     }, {
                         'name': 'centr-pc-18-table.csv',
                         'nuclei_list': [3, 4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {4: [402, 403]},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-200-table.csv',
                         'nuclei_list': [1, 5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-201-table.csv',
                         'nuclei_list': [10],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {10: [[1000, 1002]]}
                     }, {
                         'name': 'centr-pc-202-table.csv',
                         'nuclei_list': [1, 5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {5: [[500, 502]]}
                     }, {
                         'name': 'centr-pc-203-table.csv',
                         'nuclei_list': [6],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-204-table.csv',
                         'nuclei_list': [7],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-205-table.csv',
                         'nuclei_list': [4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-207-table.csv',
                         'nuclei_list': [7],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-209-table.csv',
                         'nuclei_list': [5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-210-table.csv',
                         'nuclei_list': [3, 6],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-211-table.csv',
                         'nuclei_list': [3],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-212-table.csv',
                         'nuclei_list': [4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {4: [[400, 402]]}
                     }, {
                         'name': 'centr-pc-213-table.csv',
                         'nuclei_list': [1, 2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-214-table.csv',
                         'nuclei_list': [5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {5: [600]},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {5: [[500, 502]]}
                         # }, {
                         #     'name': 'centr-pc-216-table.csv',
                         #     'nuclei_list': [],
                         #     'max_time_dict': {},
                         #     'centrosome_inclusion_dict': {},
                         #     'centrosome_exclusion_dict': {},
                         # 'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-218-table.csv',
                         'nuclei_list': [4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {4: [[400, 402]]}
                     }, {
                         'name': 'centr-pc-219-table.csv',
                         'nuclei_list': [4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {4: [[400, 402], [401, 403]]}
                         # }, {
                         #     'name': 'centr-pc-220-table.csv',
                         #     'nuclei_list': [],
                         #     'max_time_dict': {},
                         #     'centrosome_inclusion_dict': {},
                         #     'centrosome_exclusion_dict': {},
                         # 'centrosome_equivalence_dict': {}
                     }, {
                         'name': 'centr-pc-221-table.csv',
                         'nuclei_list': [2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {2: [[201, 202]]}
                     }, {
                         'name': 'centr-pc-222-table.csv',
                         'nuclei_list': [2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {2: [[201, 202, 203]]}
                     }, {
                         'name': 'centr-pc-223-table.csv',
                         'nuclei_list': [5, 6, 7],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {5: [[500, 502]], 7: [[701, 702, 703]]}
                     }, {
                         'name': 'centr-pc-224-table.csv',
                         'nuclei_list': [4, 6],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {}
                     }
                     ]}

    html = '<h3></h3>'
    stats = pd.DataFrame()
    cond = pc_to_process
    for f in cond['files'][0:]:
        print f['name']
        dfij = DataFrameFromImagej(cond['path'] + f['name'], stats_df=stats)
        html += dfij.html_centrosomes_report(nuclei_list=f['nuclei_list'], max_time_dict=f['max_time_dict'],
                                             centrosome_inclusion_dict=f['centrosome_inclusion_dict'],
                                             centrosome_exclusion_dict=f['centrosome_exclusion_dict'],
                                             centrosome_equivalence_dict=f['centrosome_equivalence_dict'])

        stats = dfij.stats
    html_pc = html

    plt.figure(10)
    sdata = stats[(stats['Stat'] == 'Snapshot') & (stats['Dist'].notnull())][['Dist', 'Type']]
    sdata['Dist'] = sdata.Dist.astype(np.float64)  # fixes a bug of seaborn
    sns.boxplot(data=sdata, y='Dist', x='Type', whis=np.inf)
    ax = sns.swarmplot(data=sdata, y='Dist', x='Type')
    for i, artist in enumerate(ax.artists):
        artist.set_facecolor('None')
    plt.savefig('out/img/beeswarm_boxplot_pc_snapshot.svg', format='svg')
    plt.close(10)

    plt.figure(11)
    sdata = stats[(stats['Stat'] == 'Contact') & (stats['Dist'].notnull())][['Dist', 'Type']]
    sdata['Dist'] = sdata.Dist.astype(np.float64)  # fixes a bug of seaborn
    sns.boxplot(data=sdata, y='Dist', x='Type', whis=np.inf)
    ax = sns.swarmplot(data=sdata, y='Dist', x='Type')
    for i, artist in enumerate(ax.artists):
        artist.set_facecolor('None')
    plt.savefig('out/img/beeswarm_boxplot_pc_contact.svg', format='svg')
    plt.close(11)

    dyndic1_to_process = {'path': '/Users/Fabio/lab/Dyn/data/',
                          'files': [{
                              'name': 'centr-dyn-101-table.csv',
                              # 'nuclei_list': [4],
                              'nuclei_list': [],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {4: [0, 1]},
                              'centrosome_exclusion_dict': {4: [400]},
                              'centrosome_equivalence_dict': {}
                              # }, {
                              #     'name': 'centr-dyn-102-table.csv',
                              #     'nuclei_list': [3],
                              #     'max_time_dict': {},
                              #     'centrosome_inclusion_dict': {},
                              #     'centrosome_exclusion_dict': {},
                              #     'centrosome_equivalence_dict': {3:[]}
                              # }, {
                              #     'name': 'centr-dyn-103-table.csv',
                              #     'nuclei_list': [],
                              #     'max_time_dict': {},
                              #     'centrosome_inclusion_dict': {},
                              #     'centrosome_exclusion_dict': {},
                              #     'centrosome_equivalence_dict': {}
                              # }, {
                              #     'name': 'centr-dyn-104-table.csv',
                              #     'nuclei_list': [],
                              #     'max_time_dict': {},
                              #     'centrosome_inclusion_dict': {},
                              #     'centrosome_exclusion_dict': {},
                              #     'centrosome_equivalence_dict': {}
                          }, {
                              'name': 'centr-dyn-105-table.csv',
                              'nuclei_list': [4],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {4: [402, 403]},
                              'centrosome_equivalence_dict': {}
                          }, {
                              'name': 'centr-dyn-107-table.csv',
                              'nuclei_list': [8],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {8: [0, 3]},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {}
                          }, {
                              'name': 'centr-dyn-109-table.csv',
                              'nuclei_list': [4, 5],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {4: [200]},
                              'centrosome_exclusion_dict': {4: [401], 5: [502]},
                              'centrosome_equivalence_dict': {}
                              # }, {
                              #     'name': 'centr-dyn-110-table.csv',
                              #     'nuclei_list': [],
                              #     'max_time_dict': {},
                              #     'centrosome_inclusion_dict': {},
                              #     'centrosome_exclusion_dict': {},
                              #     'centrosome_equivalence_dict': {}
                          }, {
                              'name': 'centr-dyn-112-table.csv',
                              'nuclei_list': [3],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {3: [200]},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {}
                          }, {
                              'name': 'centr-dyn-203-table.csv',
                              'nuclei_list': [2],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {2: [200, 203, 204, 205, 206, 207]},
                              'centrosome_equivalence_dict': {}
                          }, {
                              'name': 'centr-dyn-204-table.csv',
                              'nuclei_list': [3],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {3: [101]},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {}
                          }, {
                              'name': 'centr-dyn-205-table.csv',
                              # 'nuclei_list': [4],
                              'nuclei_list': [],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {4: [0, 301]},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {}
                          }, {
                              'name': 'centr-dyn-207-table.csv',
                              # 'nuclei_list': [2, 3],
                              'nuclei_list': [3],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {2: [0, 301]},
                              'centrosome_exclusion_dict': {2: [200]},
                              'centrosome_equivalence_dict': {}
                          }, {
                              'name': 'centr-dyn-208-table.csv',
                              'nuclei_list': [1],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {}
                          }, {
                              'name': 'centr-dyn-209-table.csv',
                              'nuclei_list': [3],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {}
                          }, {
                              'name': 'centr-dyn-210-table.csv',
                              'nuclei_list': [2, 3, 4],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {4: [402]},
                              'centrosome_equivalence_dict': {}
                          }, {
                              'name': 'centr-dyn-213-table.csv',
                              'nuclei_list': [4],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {}
                          },
                          ]}

    html = '<h3></h3>'
    stats = pd.DataFrame()
    cond = dyndic1_to_process
    for f in cond['files'][0:]:
        print f['name']
        dfij = DataFrameFromImagej(cond['path'] + f['name'], stats_df=stats)
        html += dfij.html_centrosomes_report(nuclei_list=f['nuclei_list'], max_time_dict=f['max_time_dict'],
                                             centrosome_inclusion_dict=f['centrosome_inclusion_dict'],
                                             centrosome_exclusion_dict=f['centrosome_exclusion_dict'],
                                             centrosome_equivalence_dict=f['centrosome_equivalence_dict'])

        stats = dfij.stats
    html_dyndic1 = html

    plt.figure(10)
    sdata = stats[(stats['Stat'] == 'Snapshot') & (stats['Dist'].notnull())][['Dist', 'Type']]
    sdata['Dist'] = sdata.Dist.astype(np.float64)
    sns.boxplot(data=sdata, y='Dist', x='Type', whis=np.inf)
    ax = sns.swarmplot(data=sdata, y='Dist', x='Type')
    for i, artist in enumerate(ax.artists):
        artist.set_facecolor('None')
    plt.savefig('out/img/beeswarm_boxplot_dyndic1_snapshot.svg', format='svg')
    plt.close(10)

    plt.figure(11)
    sdata = stats[(stats['Stat'] == 'Contact') & (stats['Dist'].notnull())][['Dist', 'Type']]
    sdata['Dist'] = sdata.Dist.astype(np.float64)
    sns.boxplot(data=sdata, y='Dist', x='Type', whis=np.inf)
    ax = sns.swarmplot(data=sdata, y='Dist', x='Type')
    for i, artist in enumerate(ax.artists):
        artist.set_facecolor('None')
    plt.savefig('out/img/beeswarm_boxplot_dyndic1_contact.svg', format='svg')
    plt.close(11)

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
                    <h3>Distance from nuclei center at initial time and 100 mins after</h3>
                        <img src="img/beeswarm_boxplot_pc_snapshot.svg">
                    <h3>Distance from nuclei center at time of contact</h3>
                        <img src="img/beeswarm_boxplot_pc_contact.svg">
                    </div>
                    {{ nuclei_data_pc_html }}

                    <h2>Condition: DynH1 & DIC1 ({{dyndic1_n}} tracks)</h2>
                    <h2>Brief</h2>
                    <div class="container">
                    <h3>Distance from nuclei center at initial time and 100 mins after</h3>
                        <img src="img/beeswarm_boxplot_dyndic1_snapshot.svg">
                    <h3>Distance from nuclei center at time of contact</h3>
                        <img src="img/beeswarm_boxplot_dyndic1_contact.svg">
                    </div>
                    {{ nuclei_data_dyndic1_html }}
                </body>
                </html>
                """
    templ = j2.Template(master_template)
    htmlout = templ.render(
        {'title': 'Centrosomes report',
         'report_date': time.strftime("%d/%m/%Y"), 'nuclei_data_pc_html': html_pc,
         'pc_n': len(np.concatenate([np.array(n['nuclei_list']) for n in pc_to_process['files']])),
         'nuclei_data_dyndic1_html': html_dyndic1,
         'dyndic1_n': len(np.concatenate([np.array(n['nuclei_list']) for n in dyndic1_to_process['files']]))})

    with codecs.open('out/index.html', "w", "utf-8") as text_file:
        text_file.write(htmlout)

        # pdfkit.from_file('out/index.html', 'out/report.pdf')

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
# sns.set_style("dark")
# plt.style.use('default')
# plt.style.use('seaborn-white')
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
        self.d_threshold = 1
        self.centrosome_replacements = dict()

    @staticmethod
    def get_contact_time(df, threshold):
        # get all distances less than a threshold, order them by time and pick the earlier one
        cent_list = df.groupby('Centrosome').size().index
        if len(cent_list) <= 1:
            return 0, 0, 0
        elif len(cent_list) == 2:
            ds = df.copy()
            cl_bool = ds['Centrosome'] == cent_list[1]

            ds.loc[cl_bool, ['CNx', 'CNy']] *= -1
            dsf = ds.groupby(['Frame'])[['CNx', 'CNy']].sum().apply(
                lambda x: x ** 2)  # dist between centrosomes squared
            dsf['Dist'] = (dsf.CNx + dsf.CNy).apply(lambda x: np.sqrt(x))
            dsr = dsf[dsf['Dist'] < threshold]

            if dsr.size > 0:
                dsr = dsr['Dist'].sort_index()
                frame = list(dsr.index)[0]
                time = list(ds[ds['Frame'] == frame]['Time'])[0]
                dist = list(ds[ds['Frame'] == frame]['Dist'])[0]

            else:  # just get the time for the min distance
                dsr = dsf['Dist'].sort_values()
                frame = list(dsr.index)[0]
                time = list(ds[ds['Frame'] == frame]['Time'])[0]
                dist = list(ds[ds['Frame'] == frame]['Dist'])[0]

            return time, frame, dist
        else:
            return None, None, None

    @staticmethod
    def compute_velocity(df):
        df = df.set_index('Frame').sort_index()
        for c_i in df.groupby('Centrosome').groups.keys():
            dc = df[df['Centrosome'] == c_i]
            d = dc[['CentX', 'CentY', 'NuclX', 'NuclY', 'CNx', 'CNy', 'Time']].diff()

            d['Dist'] = np.sqrt(d.CNx ** 2 + d.CNy ** 2)  # relative to nuclei centroid
            #  d['Dist'] = np.sqrt(d.CentX ** 2 + d.CentX ** 2)  # dist  between centrosomes

            df.loc[df['Centrosome'] == c_i, 'Dist'] = np.sqrt(dc.CNx ** 2 + dc.CNy ** 2)  # relative to nuclei centroid
            df.loc[df['Centrosome'] == c_i, 'Speed'] = d.Dist / d.Time

        return df.reset_index()

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
            return

        df_rowc = pd.DataFrame({'Tag': self.fname,
                                'Nuclei': _ndf['Nuclei'].unique()[0],
                                'Frame': [frame_contact],
                                'Time': [time_contact],
                                'Stat': 'Distance',
                                'Type': 'Contact',
                                'Dist': [dist_contact]})
        df_row1 = pd.DataFrame({'Tag': self.fname,
                                'Nuclei': _ndf['Nuclei'].unique()[0],
                                'Frame': [frame_before],
                                'Time': [time_before],
                                'Stat': 'Distance',
                                'Type': 'Away',
                                'Dist': [max_dist]})
        df_row2 = pd.DataFrame({'Tag': self.fname,
                                'Nuclei': _ndf['Nuclei'].unique()[0],
                                'Frame': [frame_before],
                                'Time': [time_before],
                                'Stat': 'Distance',
                                'Type': 'Close',
                                'Dist': [min_dist]})
        self.stats = self.stats.append(df_row1, ignore_index=True)
        self.stats = self.stats.append(df_row2, ignore_index=True)
        self.stats = self.stats.append(df_rowc, ignore_index=True)

    def plot_nucleus_dataframe(self, nuclei_df, filename=None):
        nucleus_id = nuclei_df['Nuclei'].min()
        gs = matplotlib.gridspec.GridSpec(4, 1)
        ax1 = plt.subplot(gs[0:2, 0])
        ax2 = plt.subplot(gs[2, 0])
        ax3 = plt.subplot(gs[3, 0])

        # get time of contact
        time_contact, frame_contact, dist_contact = self.get_contact_time(nuclei_df, self.d_threshold)
        # hack: get two distances before dt_before_contact and store them in an independent dataframe
        if time_contact is not None:
            frame_before = frame_contact - self.dt_before_contact / self.t_per_frame
            if frame_before < 0:
                frame_before = 0
            dists_before_contact = list(
                nuclei_df[nuclei_df['Frame'] == frame_before]['Dist'])

        in_yticks = list()
        in_yticks_lbl = list()
        for [(lblCentr), _df], k in zip(nuclei_df.groupby(['Centrosome']),
                                        range(len(nuclei_df.groupby(['Centrosome'])))):
            track = _df.set_index('Time').sort_index()

            track.Dist.plot(ax=ax1, label='N%d-C%d' % (nucleus_id, lblCentr), marker='o', sharex=True)
            track.Speed.plot(ax=ax2, label='N%d-C%d' % (nucleus_id, lblCentr), sharex=True)

            # plot time of contact
            if time_contact is not None:
                ax1.axvline(x=time_contact, color='dimgray', linestyle='--')
                ax1.axvline(x=time_contact - self.dt_before_contact, color='lightgray', linestyle='--')

                i_n = track.InsideNuclei + 2 * k
                in_yticks.append(2 * k)
                in_yticks.append(2 * k + 1)
                in_yticks_lbl.append('Out')
                in_yticks_lbl.append('Inside')
                i_n.plot(ax=ax3, label='N%d-C%d' % (nucleus_id, lblCentr), marker='o', ylim=[-0.5, 2 * k + 1.5],
                         sharex=True)
            else:
                track.InsideNuclei.plot(ax=ax3, label='N%d-C%d' % (nucleus_id, lblCentr), color=(0, 0, 0, 0),
                                        sharex=True)

        ax1.legend()
        ax2.legend()
        ax3.set_yticks(in_yticks)
        ax3.set_yticklabels(in_yticks_lbl)

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, format='svg')

    @staticmethod
    def join_tracks(df, cn, cm):
        # makes the operation df(Cn) <- df(Cm)
        # that is, to replace all Cn with Cm
        df.ix[df['Centrosome'] == cn, 'Centrosome'] = cm
        # df.ix[df['InsideNuclei'] == cn, 'InsideNuclei'] = df.ix[df['InsideNuclei'] == cm, 'InsideNuclei']
        cmnuclei = df.ix[df['Centrosome'] == cm, 'Nuclei'].unique()
        cnnuclei = df.ix[df['Centrosome'] == cm, 'Nuclei'].unique()
        df.ix[(df['Nuclei'] == cnnuclei) & (df['Centrosome'] == cm), 'Nuclei'] = df.ix[
            (df['Nuclei'] == cmnuclei) & (df['Centrosome'] == cn), 'Nuclei']
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
        # filter non wanted centrosomes
        if centrosome_exclusion_dict is not None:
            for nuId in centrosome_exclusion_dict.keys():
                for centId in centrosome_exclusion_dict[nuId]:
                    df[(df['Nuclei'] == nuId) & (df['Centrosome'] == centId)] = np.NaN
        # include wanted centrosomes
        if centrosome_inclusion_dict is not None:
            for nuId in centrosome_inclusion_dict.keys():
                for centId in centrosome_inclusion_dict[nuId]:
                    df.ix[df['Centrosome'] == centId, 'Nuclei'] = nuId
                    # print df.ix[df['Centrosome'] == centId]

        if centrosome_equivalence_dict is not None:
            for nuId in centrosome_equivalence_dict.keys():
                min_cm = min(centrosome_equivalence_dict[nuId])
                ceq = dict()
                for cn in centrosome_equivalence_dict[nuId]:
                    if cn != min_cm:
                        df = self.join_tracks(df, cn, min_cm)
                        ceq[cn] = min_cm
                self.centrosome_replacements[nuId] = ceq

        # filter dataframe with wanted data
        df = df.ix[(df.ValidCentroid == 1)]
        # merge with nuclei data
        df = df.merge(self.df_nuclei_csv)

        # df = self.join_tracks_algorithm(df)
        print self.centrosome_replacements

        # compute characteristics
        df['CNx'] = df.NuclX - df.CentX
        df['CNy'] = df.NuclY - df.CentY
        df['Time'] /= 60.0

        nuclei_data = list()
        for (nucleusID), filtered_nuc_df in df.groupby(['Nuclei']):
            if nucleusID in nuclei_list:
                if (max_time_dict is not None) and (nucleusID in max_time_dict):
                    filtered_nuc_df = filtered_nuc_df[filtered_nuc_df['Time'] <= max_time_dict[nucleusID]]
                exp_id = re.search('centr-(.+?)-table.csv', self.path_csv).group(1)

                c_tags = ''
                if nucleusID in self.centrosome_replacements.keys():
                    centrs = set(self.centrosome_replacements[nucleusID].values())
                    for ct in centrs:
                        c_tags += 'C%d was merged with (' % ct
                        for r in self.centrosome_replacements[nucleusID]:
                            if self.centrosome_replacements[nucleusID][r] == ct:
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
    stats_pc_dataframe = pd.DataFrame()
    html_pc = '<h3></h3>'
    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-0-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[1, 2], max_time_dict={1: 130})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-1-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[0, 2, 4], max_time_dict={2: 110, 4: 110})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-3-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[8], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-4-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[5], centrosome_exclusion_dict={5: [26, 27]},
                                            max_time_dict={5: 120})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-10-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-12-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[1], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-14-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-17-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[3], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-18-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[2, 3], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-200-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[1, 4], centrosome_exclusion_dict={1: [13], 4: [12]},
                                            max_time_dict={4: 170})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-201-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[13], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-202-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[9], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-203-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    # dfij= html_centrosomes_report("/Users/Fabio/lab/PC/data/centr-pc-204-table.csv", stats_df=stats_dataframe).html_centrosomes_report(nuclei_list=[], max_time_dict={})
    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-205-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    # dfij= html_centrosomes_report("/Users/Fabio/lab/PC/data/centr-pc-207-table.csv", stats_df=stats_dataframe).html_centrosomes_report(nuclei_list=[], max_time_dict={})
    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-209-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-210-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-211-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-212-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-213-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[4, 7], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-214-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[4, 5], centrosome_inclusion_dict={5: [4]}, max_time_dict={})
    stats_pc_dataframe = dfij.stats

    # dfij= html_centrosomes_report("/Users/Fabio/lab/PC/data/centr-pc-216-table.csv").html_centrosomes_report(nuclei_list=[], max_time_dict={})
    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-218-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-219-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[8], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    # dfij= html_centrosomes_report("/Users/Fabio/lab/PC/data/centr-pc-220-table.csv").html_centrosomes_report(nuclei_list=[], max_time_dict={})
    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-221-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[3], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-222-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[1], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    # dfij= html_centrosomes_report("/Users/Fabio/lab/PC/data/centr-pc-223-table.csv").html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    dfij = DataFrameFromImagej("/Users/Fabio/lab/PC/data/centr-pc-224-table.csv", stats_df=stats_pc_dataframe)
    html_pc += dfij.html_centrosomes_report(nuclei_list=[6], max_time_dict={})
    stats_pc_dataframe = dfij.stats

    plt.figure(10)
    ax = sns.swarmplot(data=stats_pc_dataframe, y='Dist', x='Type')
    plt.savefig('out/img/beeswarm_pc.svg', format='svg')
    plt.figure(11)
    sns.boxplot(data=stats_pc_dataframe, y='Dist', x='Type', whis=np.inf)
    ax = sns.swarmplot(data=stats_pc_dataframe[stats_pc_dataframe['Stat'] == 'Distance'], y='Dist', x='Type')
    for i, artist in enumerate(ax.artists):
        artist.set_facecolor('None')
    plt.savefig('out/img/beeswarm_boxplot_pc.svg', format='svg')
    # print stats_dataframe[stats_dataframe['Type'] == 'Contact']
    # print stats_dataframe




    stats_dyndic1_dataframe = pd.DataFrame()
    html_dyndic1 = ''
    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-101-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[4], centrosome_inclusion_dict={4: [1, 2]},
                                                 centrosome_exclusion_dict={})
    stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-102-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[0], centrosome_exclusion_dict={0: [0]},
                                                 centrosome_equivalence_dict={})
    stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-103-table.csv",
                               stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[], centrosome_inclusion_dict={},
                                                 centrosome_exclusion_dict={})
    stats_dyndic1_dataframe = dfij.stats

    # dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-104-table.csv", stats_df=stats_dyndic1_dataframe)
    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-105-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[5, 10])
    stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-107-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[0, 7], centrosome_exclusion_dict={0: [0], 7: [701]})
    stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-109-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[0, 4], centrosome_exclusion_dict={0: [2, 3]})
    stats_dyndic1_dataframe = dfij.stats

    # dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-110-table.csv", stats_df=stats_dyndic1_dataframe)
    # html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[0, 2, 3])
    # stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-112-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[2])
    stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-203-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[2],
                                                 centrosome_exclusion_dict={2: [202, 203, 204, 205, 206]})
    stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-204-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[3], centrosome_inclusion_dict={3: [202]})
    stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-205-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[0, 1], centrosome_inclusion_dict={1: [301]},
                                                 centrosome_exclusion_dict={0: [2], 1: [100, 102, 103, 104]})
    stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-207-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[1, 3], centrosome_inclusion_dict={3: [102, 103]},
                                                 centrosome_exclusion_dict={1: [102, 103]})
    stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-208-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[0])
    stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-209-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[2], centrosome_exclusion_dict={2: [202, 203]})
    stats_dyndic1_dataframe = dfij.stats

    # dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-210-table.csv", stats_df=stats_dyndic1_dataframe)
    # html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[2, 3, 5], centrosome_inclusion_dict={2: [300, 302]})
    # stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/Dyn/data/centr-dyn-213-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[1])
    stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/DIC1/data/centr-dic1-203-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[0, 2])
    stats_dyndic1_dataframe = dfij.stats

    dfij = DataFrameFromImagej("/Users/Fabio/lab/DIC1/data/centr-dic1-204-table.csv", stats_df=stats_dyndic1_dataframe)
    html_dyndic1 += dfij.html_centrosomes_report(nuclei_list=[2])
    stats_dyndic1_dataframe = dfij.stats

    plt.figure(10)
    ax = sns.swarmplot(data=stats_dyndic1_dataframe, y='Dist', x='Type')
    plt.savefig('out/img/beeswarm_dyndic1.svg', format='svg')
    plt.figure(11)
    sns.boxplot(data=stats_dyndic1_dataframe, y='Dist', x='Type', whis=np.inf)
    ax = sns.swarmplot(data=stats_dyndic1_dataframe[stats_dyndic1_dataframe['Stat'] == 'Distance'], y='Dist', x='Type')
    for i, artist in enumerate(ax.artists):
        artist.set_facecolor('None')
    plt.savefig('out/img/beeswarm_boxplot_dyndic1.svg', format='svg')

    master_template = """<!DOCTYPE html>
        <html>
        <head lang="en">
            <meta charset="UTF-8">
            <title>{{ title }}</title>
        </head>
        <body>
            <h1>Centrosome Data Report - {{report_date}}</h1>

            <h2>Condition: Positive Control</h2>
            <h3>Brief</h3>
            <div class="container">
                <img src="img/beeswarm_pc.svg">
                <img src="img/beeswarm_boxplot_pc.svg">
            </div>
            {{ nuclei_data_pc_html }}

            <h2>Condition: DynH1 & DIC1</h2>
            <h3>Brief</h3>
            <div class="container">
                <img src="img/beeswarm_dyndic1.svg">
                <img src="img/beeswarm_boxplot_dyndic1.svg">
            </div>
            {{ nuclei_data_dyndic1_html }}
        </body>
        </html>
        """
    templ = j2.Template(master_template)
    htmlout = templ.render(
        {'title': 'Centrosomes report',
         'report_date': time.strftime("%d/%m/%Y"), 'nuclei_data_pc_html': html_pc, 'nuclei_data_dyndic1_html': html_dyndic1})

    with codecs.open('out/index.html', "w", "utf-8") as text_file:
        text_file.write(htmlout)

        # pdfkit.from_file('out/index.html', 'out/report.pdf')

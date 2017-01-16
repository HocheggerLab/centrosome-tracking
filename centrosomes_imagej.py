import re
import pandas as pd
import numpy as np
import codecs
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import jinja2 as j2
import pdfkit

matplotlib.style.use('ggplot')


class dataframe_from_imagej():
    def __init__(self, filename, stats_df):
        self.path_csv = filename
        self.df_csv = pd.read_csv(self.path_csv)
        self.fname = re.search('lab/(.+?)-table.csv', self.path_csv).group(1)
        self.stats = stats_df

        self.dt_before_contact = 30
        self.t_per_frame = 5
        self.d_threshold = 1

    @staticmethod
    def get_contact_time(df, threshold):
        # get all distances less than a threshold, order them by time and pick the earlier one
        # ds = df[df['Dist'] < threshold].set_index('Time').sort_index()
        cent_list = df.groupby('Centrosome').size().index
        ds = df.copy()
        cl_bool = ds['Centrosome'] == cent_list[1]
        ds.loc[cl_bool, ['CNx', 'CNy']] *= -1
        dsf = ds.groupby(['Frame'])[['CNx', 'CNy']].sum().apply(lambda x: x ** 2)
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

    def plot_nucleus_dataframe(self, nuclei_df, filename=None):
        nucleus_id = nuclei_df['Nuclei'].min()
        gs = matplotlib.gridspec.GridSpec(4, 1)
        ax1 = plt.subplot(gs[0:2, 0])
        ax2 = plt.subplot(gs[2, 0])
        ax3 = plt.subplot(gs[3, 0])

        # compute distance & speed
        nuclei_df = nuclei_df.set_index('Time').sort_index().reset_index()
        d = nuclei_df.diff()
        d['Dist'] = np.sqrt((d.CNx) ** 2 + (d.CNy) ** 2)
        d['Time'] = nuclei_df.Time.diff()

        nuclei_df['Dist'] = np.sqrt((nuclei_df.CNx) ** 2 + (nuclei_df.CNy) ** 2)
        nuclei_df['Speed'] = d.Dist / d.Time
        spd = pd.rolling_mean(nuclei_df.Speed.abs(), window=3)

        # get time of contact
        time_contact, frame_contact, dist_contact = self.get_contact_time(nuclei_df, self.d_threshold)
        # hack: get two distances before dt_before_contact and store them in an independent dataframe
        # FIXME: move hack to a method
        frame_before = frame_contact - self.dt_before_contact / self.t_per_frame
        if frame_before < 0:
            frame_before = 0
        dists_before_contact = list(
            nuclei_df[nuclei_df['Frame'] == frame_before]['Dist'])
        df_rowc = pd.DataFrame({'Tag': self.fname,
                                'Nuclei': nuclei_df['Nuclei'].unique()[0],
                                'Frame': [frame_contact],
                                'Time': [time_contact],
                                'Type': 'Contact',
                                'Dist': [dist_contact]})
        df_row1 = pd.DataFrame({'Tag': self.fname,
                                'Nuclei': nuclei_df['Nuclei'].unique()[0],
                                'Frame': [frame_before],
                                'Time': [nuclei_df[nuclei_df['Frame'] == frame_before]['Time'].unique()[0]],
                                'Type': 'Away',
                                'Dist': [max(dists_before_contact)]})
        df_row2 = pd.DataFrame({'Tag': self.fname,
                                'Nuclei': nuclei_df['Nuclei'].unique()[0],
                                'Frame': [frame_before],
                                'Time': [nuclei_df[nuclei_df['Frame'] == frame_before]['Time'].unique()[0]],
                                'Type': 'Close',
                                'Dist': [min(dists_before_contact)]})
        self.stats = self.stats.append(df_row1, ignore_index=True)
        self.stats = self.stats.append(df_row2, ignore_index=True)
        self.stats = self.stats.append(df_rowc, ignore_index=True)

        in_yticks = list()
        in_yticks_lbl = list()
        for [(lblCentr), _df], k in zip(nuclei_df.groupby(['Centrosome']), range(len(nuclei_df.groupby(['Centrosome'])))):
            track = _df.set_index('Time').sort_index()
            track = track.reset_index()

            track = track.set_index('Time').sort_index()
            # spd = track.Speed
            track.Dist.plot(ax=ax1, label='N%d-C%d' % (nucleus_id, lblCentr), marker='o', sharex=True)
            spd.plot(ax=ax2, label='N%d-C%d' % (nucleus_id, lblCentr), sharex=True)

            # plot time of contact
            ax1.axvline(x=time_contact, color='dimgray', linestyle='--')
            ax1.axvline(x=time_contact - self.dt_before_contact, color='lightgray', linestyle='--')

            iN = track.InsideNuclei + 2 * k
            in_yticks.append(2 * k)
            in_yticks.append(2 * k + 1)
            in_yticks_lbl.append('Out')
            in_yticks_lbl.append('Inside')
            iN.plot(ax=ax3, label='N%d-C%d' % (nucleus_id, lblCentr), marker='o', ylim=[-0.5, 2 * k + 1.5], sharex=True)
        ax1.legend()
        ax2.legend()
        ax3.set_yticks(in_yticks)
        ax3.set_yticklabels(in_yticks_lbl)

        if filename == None:
            plt.show()
        else:
            plt.savefig(filename, format='svg')

    @staticmethod
    def join_tracks(df, Cn, Cm):
        # makes the operation df(Cn) <- df(Cm)
        df.ix[df['Centrosome'] == Cn, 'Centrosome'] = Cm
        df.ix[df['CentX'] == Cn, 'CentX'] = df.ix[df['CentX'] == Cm, 'CentX']
        df.ix[df['CentY'] == Cn, 'CentY'] = df.ix[df['CentY'] == Cm, 'CentY']
        df.ix[df['NuclX'] == Cn, 'NuclX'] = df.ix[df['NuclX'] == Cm, 'NuclX']
        df.ix[df['NuclY'] == Cn, 'NuclY'] = df.ix[df['NuclY'] == Cm, 'NuclY']
        return df

    @staticmethod
    def fill_gaps_in_track(df, row, frame, nucleusId, h, c):
        # fills a point in centrosome track h using a point from centrosome track c
        idata = {'Frame': [frame],
                 'Time': [row.loc['Time', c[0]]],
                 'Nuclei': [nucleusId],
                 'ValidCentroid': [True],
                 'Centrosome': [h[0]],
                 'CentX': [row.loc['CentX', c[0]]],
                 'CentY': [row.loc['CentY', c[0]]],
                 'NuclX': [row.loc['NuclX', c[0]]],
                 'NuclY': [row.loc['NuclY', c[0]]]}
        idf = pd.DataFrame(idata)
        df = df.append(idf, ignore_index=True)
        return df

    def html_centrosomes_report(self, nuclei_list=None,
                                centrosome_exclusion_dict=None,
                                centrosome_inclusion_dict=None,
                                max_time_dict=None):
        htmlout = '<h3>Filename: %s.avi</h3>' % (self.fname)

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
                    # fstCentr = df[df['Nuclei']==nuId]['Centrosome'].unique()[0]
                    nucRpl = df.ix[df['Nuclei'] == nuId, ['Nuclei', 'NuclX', 'NuclY', 'Time']].drop_duplicates()
                    for t in nucRpl.Time:
                        df.ix[(df['Centrosome'] == centId) & (df['Time'] == t), 'NuclX'] = \
                            list(nucRpl[nucRpl['Time'] == t].NuclX)[0]
                        df.ix[(df['Centrosome'] == centId) & (df['Time'] == t), 'NuclY'] = \
                            list(nucRpl[nucRpl['Time'] == t].NuclY)[0]

                    df.ix[df['Centrosome'] == centId, 'Nuclei'] = nuId
                    # print df.ix[df['Centrosome'] == centId]

        # filter dataframe wit wanted data
        df = df.ix[(df['Nuclei'].isin(nuclei_list)) & (df.ValidCentroid == 1)].copy()

        # ----------------------------------
        # Track joining algorithm
        # ----------------------------------
        # df = df.set_index(['Frame', 'Centrosome']).reindex(['Time']).reset_index()
        # check if there's a maximun of 2 centrosomes at all times
        for nID in df['Nuclei'].unique():
            # FIXME: correct that ugly filter in the if, nuc==7 and fname=='centr-pc-213'
            if (self.fname == 'centr-pc-213' and nID == 7):
                df = self.join_tracks(df, 4, 6)

            elif ((df[df['Nuclei'] == nID].groupby('Frame')['Centrosome'].count().max() <= 2) & (
                    not (self.fname == 'centr-pc-213' and nID == 7))):
                # print df[df['Nuclei'] == nID].set_index(['Frame', 'Centrosome'])[['CentX', 'CentY', 'Nuclei', 'NuclX', 'NuclY']].unstack()
                d = df[df['Nuclei'] == nID].set_index(['Frame', 'Centrosome']).sort_index().unstack()

                last_ccouple, last_row, hid_centr = (None, None, None)
                idx, f = d.iterrows().next()
                orig_couple = f['CentX'][f['CentX'].notnull()].index.get_values()
                all_centr = f['CentX'].index.get_values()
                centrosomes_processed = list()
                centr_equivalence = dict()
                is_in_merge, just_ini = False, True
                for index, row in d.iterrows():
                    curr_ccouple = row['CentX'][row['CentX'].notnull()].index.get_values()

                    if ((len(orig_couple) == 1) & (len(curr_ccouple) == 2) & just_ini):  # change on original condition
                        just_ini = False
                        orig_couple = curr_ccouple

                    if ((len(curr_ccouple) == 1) & (last_ccouple is not None) & (not is_in_merge) & (
                                len(orig_couple) == 2)):  # merge
                        is_in_merge = True
                        just_ini = False
                        hid_centr = list(set(last_ccouple).difference(curr_ccouple))
                        hid_orig = hid_centr
                        while (hid_centr in centrosomes_processed):
                            hid_centr = [centr_equivalence[hid_centr[0]]]

                    if ((len(curr_ccouple) == 1) & (hid_centr is not None) & (is_in_merge)):  # inside a merge
                        visible_centr = list(set(all_centr).intersection(curr_ccouple))
                        df = self.fill_gaps_in_track(df, row, index, nID, hid_centr, visible_centr)

                    if ((len(curr_ccouple) == 2) & (hid_centr is not None) & (is_in_merge)):  # split (Cn = Cm)
                        is_in_merge = False
                        new_centr = list(set(curr_ccouple) - set(orig_couple))
                        if (len(new_centr) == 1 & (new_centr not in centrosomes_processed)):
                            df = self.join_tracks(df, new_centr, hid_centr)

                            centrosomes_processed.append(new_centr)
                            centr_equivalence[new_centr[0]] = hid_centr[0]
                        hid_centr = None

                    last_ccouple = curr_ccouple
                    # last_row = row

        # compute characteristics
        df.set_index('Frame').sort_index().reset_index()
        df['CNx'] = df.NuclX - df.CentX
        df['CNy'] = df.NuclY - df.CentY
        df['Time'] = df.Time / 60.0
        # print df[df.Speed.isnull()]

        # print df[df['Nuclei'] == 0].set_index(['Frame', 'Centrosome']).sort_index().unstack()

        nuclei_data = list()
        for (nucleusID), filtered_nuc_df in df.groupby(['Nuclei']):
            if (nucleusID in nuclei_list):
                if ((max_time_dict is not None) and (nucleusID in max_time_dict)):
                    filtered_nuc_df = filtered_nuc_df[filtered_nuc_df['Time'] <= max_time_dict[nucleusID]]
                nuc_item = {}
                nuc_item['filename'] = self.path_csv
                nuc_item['exp_id'] = re.search('centr-(.+?)-table.csv', self.path_csv).group(1)
                nuc_item['nuclei_id'] = '%d ' % nucleusID
                nuc_item['nuclei_centrosomes_tags'] = ''.join(
                    ['C%d, ' % cnId for cnId in sorted(filtered_nuc_df['Centrosome'].unique())])[:-2]
                nuc_item['centrosomes_img'] = 'img/%s-nuc_%d.svg' % (nuc_item['exp_id'], nucleusID)
                # nuc_item['centrosomes_speed_stats'] = filtered_nuc_df.groupby(['Centrosome']).Speed.describe()

                self.plot_nucleus_dataframe(filtered_nuc_df, 'out/%s' % nuc_item['centrosomes_img'])

                nuclei_data.append(nuc_item)

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
    stats_dataframe = pd.DataFrame()
    html = '<h3></h3>'
    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-0-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[1, 2], max_time_dict={1: 130})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-1-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[0, 2, 4], max_time_dict={2: 110, 4: 110})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-3-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[8], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-4-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[5], centrosome_exclusion_dict={5: [26, 27]},
                                         max_time_dict={5: 120})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-10-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-12-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[1], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-14-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-17-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[3], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-18-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[2, 3], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-200-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[1, 4], centrosome_exclusion_dict={1: [13], 4: [12]},
                                         max_time_dict={4: 170})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-201-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[13], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-202-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[9], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-203-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    stats_dataframe = dfij.stats

    # dfij= html_centrosomes_report("/Users/Fabio/lab/centr-pc-204-table.csv", stats_df=stats_dataframe).html_centrosomes_report(nuclei_list=[], max_time_dict={})
    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-205-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    stats_dataframe = dfij.stats

    # dfij= html_centrosomes_report("/Users/Fabio/lab/centr-pc-207-table.csv", stats_df=stats_dataframe).html_centrosomes_report(nuclei_list=[], max_time_dict={})
    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-209-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-210-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-211-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-212-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-213-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[4, 7], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-214-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[4, 5], centrosome_inclusion_dict={5: [4]}, max_time_dict={})
    stats_dataframe = dfij.stats

    # dfij= html_centrosomes_report("/Users/Fabio/lab/centr-pc-216-table.csv").html_centrosomes_report(nuclei_list=[], max_time_dict={})
    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-218-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-219-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[8], max_time_dict={})
    stats_dataframe = dfij.stats

    # dfij= html_centrosomes_report("/Users/Fabio/lab/centr-pc-220-table.csv").html_centrosomes_report(nuclei_list=[], max_time_dict={})
    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-221-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[3], max_time_dict={})
    stats_dataframe = dfij.stats

    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-222-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[1], max_time_dict={})
    stats_dataframe = dfij.stats

    # dfij= html_centrosomes_report("/Users/Fabio/lab/centr-pc-223-table.csv").html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    dfij = dataframe_from_imagej("/Users/Fabio/lab/centr-pc-224-table.csv", stats_df=stats_dataframe)
    html += dfij.html_centrosomes_report(nuclei_list=[6], max_time_dict={})
    stats_dataframe = dfij.stats

    plt.figure(10)
    ax = sns.swarmplot(data=stats_dataframe, y='Dist', x='Type')
    plt.savefig('out/img/beeswarm.svg', format='svg')
    plt.figure(11)
    sns.boxplot(data=stats_dataframe, y='Dist', x='Type', whis=np.inf)
    ax = sns.swarmplot(data=stats_dataframe, y='Dist', x='Type')
    for i, artist in enumerate(ax.artists):
        artist.set_facecolor('None')
    plt.savefig('out/img/beeswarm_boxplot.svg', format='svg')
    # print stats_dataframe[stats_dataframe['Type'] == 'Contact']
    print stats_dataframe

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
                <img src="img/beeswarm.svg">
                <img src="img/beeswarm_boxplot.svg">
            </div>
            {{ nuclei_data_html }}
        </body>
        </html>
        """
    templ = j2.Template(master_template)
    htmlout = templ.render({'title': 'Centrosomes report', 'nuclei_data_html': html})

    with codecs.open('out/index.html', "w", "utf-8") as text_file:
        text_file.write(htmlout)

    pdfkit.from_file('out/index.html', 'out/report.pdf')

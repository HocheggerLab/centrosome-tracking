import re
import pandas as pd
import numpy as np
import codecs
import matplotlib
import matplotlib.pyplot as plt
import jinja2 as j2
import pdfkit

matplotlib.style.use('ggplot')


class dataframe_from_imagej():
    def __init__(self, filename):
        self.path_csv = filename
        self.df_csv = pd.read_csv(self.path_csv)
        self.fname = re.search('lab/(.+?)-table.csv', self.path_csv).group(1)

    def plot_nucleus_dataframe(self, nucleiDf, filename=None):
        nucleusID = nucleiDf['Nuclei'].min()
        gs = matplotlib.gridspec.GridSpec(4, 1)
        ax1 = plt.subplot(gs[0:2, 0])
        ax2 = plt.subplot(gs[2, 0])
        ax3 = plt.subplot(gs[3, 0])

        in_yticks = list()
        in_yticks_lbl = list()
        for [(lblCentr), _df], k in zip(nucleiDf.groupby(['Centrosome']), range(len(nucleiDf.groupby(['Centrosome'])))):
            track = _df.set_index('Time').sort_index()
            track = track.reset_index()

            # compute distance & speed
            d = track.diff()
            d['Dist'] = np.sqrt((d.CNx) ** 2 + (d.CNy) ** 2)
            d['Time'] = track.Time.diff()

            track['Dist'] = np.sqrt((track.CNx) ** 2 + (track.CNy) ** 2)
            track['Speed'] = d.Dist / d.Time

            spd = pd.rolling_mean(track.Speed.abs(), window=3)

            track = track.set_index('Time').sort_index()
            # spd = track.Speed
            track.Dist.plot(ax=ax1, label='N%d-C%d' % (nucleusID, lblCentr), marker='o', sharex=True)
            spd.plot(ax=ax2, label='N%d-C%d' % (nucleusID, lblCentr), sharex=True)

            iN = track.InsideNuclei + 2 * k
            in_yticks.append(2 * k)
            in_yticks.append(2 * k + 1)
            in_yticks_lbl.append('Out')
            in_yticks_lbl.append('Inside')
            iN.plot(ax=ax3, label='N%d-C%d' % (nucleusID, lblCentr), marker='o', ylim=[-0.5, 2 * k + 1.5], sharex=True)
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
        # print idf
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
                df = self.join_tracks(df,4,6)

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
                        df = self.fill_gaps_in_track(df,row,index,nID,hid_centr,visible_centr)

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
    html = '<h3></h3>'
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-0-table.csv") \
        .html_centrosomes_report(nuclei_list=[1, 2], max_time_dict={1: 130})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-1-table.csv") \
        .html_centrosomes_report(nuclei_list=[0, 2, 4], max_time_dict={2: 110, 4: 110})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-3-table.csv") \
        .html_centrosomes_report(nuclei_list=[8], max_time_dict={})

    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-4-table.csv") \
        .html_centrosomes_report(nuclei_list=[5], centrosome_exclusion_dict={5: [26, 27]}, max_time_dict={5: 120})

    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-10-table.csv") \
        .html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-12-table.csv") \
        .html_centrosomes_report(nuclei_list=[1], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-14-table.csv") \
        .html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-17-table.csv") \
        .html_centrosomes_report(nuclei_list=[3], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-18-table.csv") \
        .html_centrosomes_report(nuclei_list=[2, 3], max_time_dict={})

    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-200-table.csv") \
        .html_centrosomes_report(nuclei_list=[1, 4], centrosome_exclusion_dict={1: [13], 4: [12]},
                                 max_time_dict={4: 170})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-201-table.csv") \
        .html_centrosomes_report(nuclei_list=[13], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-202-table.csv") \
        .html_centrosomes_report(nuclei_list=[9], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-203-table.csv") \
        .html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    # html += html_centrosomes_report("/Users/Fabio/lab/centr-pc-204-table.csv").html_centrosomes_report(nuclei_list=[], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-205-table.csv") \
        .html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    # html += html_centrosomes_report("/Users/Fabio/lab/centr-pc-207-table.csv").html_centrosomes_report(nuclei_list=[], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-209-table.csv") \
        .html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-210-table.csv") \
        .html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-211-table.csv") \
        .html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-212-table.csv") \
        .html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-213-table.csv") \
        .html_centrosomes_report(nuclei_list=[4,7], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-214-table.csv") \
        .html_centrosomes_report(nuclei_list=[4, 5], centrosome_inclusion_dict={5: [4]}, max_time_dict={})
    # html += html_centrosomes_report("/Users/Fabio/lab/centr-pc-216-table.csv").html_centrosomes_report(nuclei_list=[], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-218-table.csv") \
        .html_centrosomes_report(nuclei_list=[5], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-219-table.csv") \
        .html_centrosomes_report(nuclei_list=[8], max_time_dict={})
    # html += html_centrosomes_report("/Users/Fabio/lab/centr-pc-220-table.csv").html_centrosomes_report(nuclei_list=[], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-221-table.csv") \
        .html_centrosomes_report(nuclei_list=[3], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-222-table.csv") \
        .html_centrosomes_report(nuclei_list=[1], max_time_dict={})
    # html += html_centrosomes_report("/Users/Fabio/lab/centr-pc-223-table.csv").html_centrosomes_report(nuclei_list=[4], max_time_dict={})
    html += dataframe_from_imagej("/Users/Fabio/lab/centr-pc-224-table.csv") \
        .html_centrosomes_report(nuclei_list=[6], max_time_dict={})

    master_template = """<!DOCTYPE html>
    <html>
    <head lang="en">
        <meta charset="UTF-8">
        <title>{{ title }}</title>
    </head>
    <body>
        <h1>Centrosome Data Report - {{report_date}}</h1>
        <h2>Condition: Positive Control</h2>
        {{ nuclei_data_html }}
    </body>
    </html>
    """
    templ = j2.Template(master_template)
    htmlout = templ.render({'title': 'Centrosomes report', 'nuclei_data_html': html})

    with codecs.open('out/index.html', "w", "utf-8") as text_file:
        text_file.write(htmlout)

    pdfkit.from_file('out/index.html', 'out/report.pdf')

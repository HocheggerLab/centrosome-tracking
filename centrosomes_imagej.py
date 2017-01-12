import re
import pandas as pd
import numpy as np
import codecs
import matplotlib
import matplotlib.pyplot as plt
import jinja2 as j2
import pdfkit
matplotlib.style.use('ggplot')


def plot_nucleus_dataframe(nucleiDf, filename=None):
    nucleusID = nucleiDf['Nuclei'].min()
    gs = matplotlib.gridspec.GridSpec(4, 1)
    ax1 = plt.subplot(gs[0:2,0])
    ax2 = plt.subplot(gs[2,0])
    ax3 = plt.subplot(gs[3,0])

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

        iN = track.InsideNuclei + 2*k
        in_yticks.append(2*k)
        in_yticks.append(2*k+1)
        in_yticks_lbl.append('Out')
        in_yticks_lbl.append('Inside')
        iN.plot(ax=ax3, label='N%d-C%d' % (nucleusID, lblCentr), marker='o', ylim=[-0.5, 2*k+1.5], sharex=True)
    ax1.legend()
    ax2.legend()
    ax3.set_yticks(in_yticks)
    ax3.set_yticklabels(in_yticks_lbl)

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, format='svg')




def html_centrosomes_csv(filename, nuclei_list=None,
                         centrosome_exclusion_dict=None,
                         centrosome_inclusion_dict=None,
                         max_time_dict=None):
    df = pd.read_csv(filename)
    fname = re.search('lab/(.+?)-table.csv', filename).group(1)
    htmlout = '<h3>Filename: %s.avi</h3>'%(fname)

    # apply filters
    # filter non wanted centrosomes
    if centrosome_exclusion_dict is not None:
        for nuId in centrosome_exclusion_dict.keys():
            for centId in centrosome_exclusion_dict[nuId]:
                df[(df['Nuclei']==nuId) & (df['Centrosome']==centId)] = np.NaN
    # include wanted centrosomes
    if centrosome_inclusion_dict is not None:
        for nuId in centrosome_inclusion_dict.keys():
            for centId in centrosome_inclusion_dict[nuId]:
                # fstCentr = df[df['Nuclei']==nuId]['Centrosome'].unique()[0]
                nucRpl = df.ix[df['Nuclei']==nuId, ['Nuclei','NuclX','NuclY','Time']].drop_duplicates()
                for t in nucRpl.Time:
                    df.ix[(df['Centrosome'] == centId) & (df['Time'] == t), 'NuclX'] = list(nucRpl[nucRpl['Time'] == t].NuclX)[0]
                    df.ix[(df['Centrosome'] == centId) & (df['Time'] == t), 'NuclY'] = list(nucRpl[nucRpl['Time'] == t].NuclY)[0]

                df.ix[df['Centrosome'] == centId, 'Nuclei'] = nuId
                # print df.ix[df['Centrosome'] == centId]

    # filter dataframe wit wanted data
    df = df.ix[(df['Nuclei'].isin(nuclei_list)) & (df.ValidCentroid==1)]

    # ----------------------------------
    # Track joining algorithm
    # ----------------------------------
    # df = df.set_index(['Frame', 'Centrosome']).reindex(['Time']).reset_index()
    # check if there's a maximun of 2 centrosomes at all times
    for nID in df['Nuclei'].unique():
        if (df[df['Nuclei']==nID].groupby('Frame')['Centrosome'].count().max() <= 2):
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

                if ((len(orig_couple)==1) & (len(curr_ccouple)==2) & just_ini): # change on original condition
                    just_ini = False
                    orig_couple = curr_ccouple

                if ((len(curr_ccouple)==1) & (last_ccouple is not None) & (not is_in_merge) & (len(orig_couple)==2)): # merge
                    is_in_merge = True
                    just_ini =  False
                    hid_centr = list(set(last_ccouple).difference(curr_ccouple))
                    hid_orig = hid_centr
                    while (hid_centr in centrosomes_processed):
                        hid_centr = [centr_equivalence[hid_centr[0]]]

                if ((len(curr_ccouple)==1) & (hid_centr is not None) & (is_in_merge)): # inside a merge
                    visible_centr = list(set(all_centr).intersection(curr_ccouple))
                    # if(visible_centr in centrosomes_processed):
                    #     visible_centr = [centr_equivalence[visible_centr[0]]]
                    # df.ix[(df['Frame']==index) & (df['Centrosome']==hcentr_orig[0]), ['CentX']] = row.loc['CentX', visible_centr[0]]
                    # df.ix[(df['Frame']==index) & (df['Centrosome']==hcentr_orig[0]), ['CentY']] = row.loc['CentY', visible_centr[0]]
                    # df.ix[(df['Frame']==index) & (df['Centrosome']==hcentr_orig[0]), ['NuclX']] = row.loc['NuclX', visible_centr[0]]
                    # df.ix[(df['Frame']==index) & (df['Centrosome']==hcentr_orig[0]), ['NuclY']] = row.loc['NuclY', visible_centr[0]]
                    idata = {'Frame':[index],
                             'Time':[row.loc['Time', visible_centr[0]]],
                             'Nuclei':[nID],
                             'ValidCentroid':[True],
                             'Centrosome':[hid_centr[0]],
                             'CentX':[row.loc['CentX', visible_centr[0]]],
                             'CentY':[row.loc['CentY', visible_centr[0]]],
                             'NuclX':[row.loc['NuclX', visible_centr[0]]],
                             'NuclY':[row.loc['NuclY', visible_centr[0]]]}
                    idf = pd.DataFrame(idata)
                    # print idf
                    df = df.append(idf, ignore_index=True)

                if ((len(curr_ccouple)==2) & (hid_centr is not None) & (is_in_merge)): # split (Cn = Cm)
                    is_in_merge =  False
                    new_centr = list(set(curr_ccouple)-set(orig_couple))
                    if (len(new_centr)==1 & (new_centr not in centrosomes_processed)):
                        df.ix[df['Centrosome'] == new_centr, 'Centrosome'] = hid_centr
                        df.ix[df['CentX'] == new_centr, 'CentX'] = df.ix[df['CentX'] == hid_centr, 'CentX']
                        df.ix[df['CentY'] == new_centr, 'CentY'] = df.ix[df['CentY'] == hid_centr, 'CentY']
                        df.ix[df['NuclX'] == new_centr, 'NuclX'] = df.ix[df['NuclX'] == hid_centr, 'NuclX']
                        df.ix[df['NuclY'] == new_centr, 'NuclY'] = df.ix[df['NuclY'] == hid_centr, 'NuclY']

                        centrosomes_processed.append(new_centr)
                        centr_equivalence[new_centr[0]] = hid_centr[0]
                    hid_centr = None

                last_ccouple = curr_ccouple
                # last_row = row



    # compute characteristics
    df.set_index('Frame').sort_index()
    df['CNx'] = df.NuclX - df.CentX
    df['CNy'] = df.NuclY - df.CentY
    df['Time'] = df.Time / 60.0
    df.reset_index()
    # print df[df.Speed.isnull()]

    # print df[df['Nuclei'] == 0].set_index(['Frame', 'Centrosome']).sort_index().unstack()

    nuclei_data = list()
    for (nucleusID), filtered_nuc_df in df.groupby(['Nuclei']):
        if(nucleusID in nuclei_list):
            if((max_time_dict is not None) and (nucleusID in max_time_dict)):
                filtered_nuc_df = filtered_nuc_df[filtered_nuc_df['Time']<=max_time_dict[nucleusID]]
            nuc_item={}
            nuc_item['filename'] = filename
            nuc_item['exp_id'] = re.search('centr-(.+?)-table.csv', filename).group(1)
            nuc_item['nuclei_id'] = '%d '%nucleusID
            nuc_item['nuclei_centrosomes_tags'] = ''.join( ['C%d, '%cnId for cnId in sorted( filtered_nuc_df['Centrosome'].unique() )] )[:-2]
            nuc_item['centrosomes_img'] = 'img/%s-nuc_%d.svg'%(nuc_item['exp_id'],nucleusID)
            # nuc_item['centrosomes_speed_stats'] = filtered_nuc_df.groupby(['Centrosome']).Speed.describe()

            plot_nucleus_dataframe(filtered_nuc_df, 'out/%s'%nuc_item['centrosomes_img'])
            nuclei_data.append(nuc_item)


    template ="""
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
    htmlout += templ.render({'nuclei_list':nuclei_data})
    return htmlout


html = '<h3></h3>'
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-0-table.csv", nuclei_list=[1,2], max_time_dict={1:130})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-1-table.csv", nuclei_list=[0,2,4], max_time_dict={2:110,4:110})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-3-table.csv", nuclei_list=[8], max_time_dict={})

html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-4-table.csv", nuclei_list=[5], max_time_dict={5:120})

html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-10-table.csv", nuclei_list=[4], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-12-table.csv", nuclei_list=[1], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-14-table.csv", nuclei_list=[4], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-17-table.csv", nuclei_list=[3], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-18-table.csv", nuclei_list=[2,3], max_time_dict={})


html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-200-table.csv", nuclei_list=[1,4],
                             centrosome_exclusion_dict={1:[13], 4:[12]}, max_time_dict={4:170})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-201-table.csv", nuclei_list=[13], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-202-table.csv", nuclei_list=[9], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-203-table.csv", nuclei_list=[5], max_time_dict={})
# html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-204-table.csv", nuclei_list=[], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-205-table.csv", nuclei_list=[5], max_time_dict={})
# html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-207-table.csv", nuclei_list=[], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-209-table.csv", nuclei_list=[5], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-210-table.csv", nuclei_list=[5], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-211-table.csv", nuclei_list=[4], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-212-table.csv", nuclei_list=[4], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-213-table.csv", nuclei_list=[7], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-214-table.csv", nuclei_list=[4,5],
                             centrosome_inclusion_dict={5:[4]}, max_time_dict={})
# html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-216-table.csv", nuclei_list=[], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-218-table.csv", nuclei_list=[5], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-219-table.csv", nuclei_list=[8], max_time_dict={})
# html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-220-table.csv", nuclei_list=[], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-221-table.csv", nuclei_list=[3], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-222-table.csv", nuclei_list=[1], max_time_dict={})
# html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-223-table.csv", nuclei_list=[4], max_time_dict={})
html += html_centrosomes_csv("/Users/Fabio/lab/centr-pc-224-table.csv", nuclei_list=[6], max_time_dict={})




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
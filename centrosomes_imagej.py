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
        track = _df[_df.Centrosome == lblCentr].set_index('Time')

        track.Dist.plot(ax=ax1, label='N%d-C%d' % (nucleusID, lblCentr), marker='o', sharex=True)
        spd = pd.rolling_mean(track.Speed.abs(), window=3)
        # spd = track.Speed
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




def html_centrosomes_csv(filename, nuclei_ban_list=None, max_time_dict=None):
    df = pd.read_csv(filename)

    df = df[df.ValidCentroid==1]
    df['CNx'] = df.NuclX - df.CentX
    df['CNy'] = df.NuclY - df.CentY
    df['Time'] = df.Time / 60.0

    d = df.diff()
    d['Dist'] = np.sqrt((d.CNx) ** 2 + (d.CNy) ** 2)
    d['Time'] = df.groupby(['Centrosome']).Time.apply(lambda x: x.diff())
    # d = d.fillna(0.)

    df['Dist'] = np.sqrt((df.CNx) ** 2 + (df.CNy) ** 2)
    df['Speed'] = d.Dist / d.Time
    df[np.isinf(df.Speed)] = float('NaN')


    nuclei_list = list()
    for (nucleusID), filtered_nuc_df in df.groupby(['Nuclei']):
        nuc_item={}
        nuc_item['filename'] = filename
        nuc_item['nuclei_id'] = '%d '%nucleusID
        nuc_item['nuclei_centrosomes_tags'] = ''.join( ['C%d, '%cnId for cnId in sorted( filtered_nuc_df['Centrosome'].unique() )] )[:-2]
        nuc_item['centrosomes_img'] = 'img/nuc_%d.svg'%nucleusID
        nuc_item['centrosomes_speed_stats'] = filtered_nuc_df.groupby(['Centrosome']).Speed.describe()

        plot_nucleus_dataframe(filtered_nuc_df, 'out/%s'%nuc_item['centrosomes_img'])
        nuclei_list.append(nuc_item)


    template ="""
        {% for nuc_item in nuclei_list %}
        <div class="container">
            <h3>Filename: {{ nuc_item['filename']}}</h3>
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
    htmlout = templ.render({'nuclei_list':nuclei_list})
    return htmlout



html = '<h2>Condition: Positive Control</h2>' + html_centrosomes_csv("/Users/Fabio/lab/centr-pc-0-table.csv")

master_template = """<!DOCTYPE html>
<html>
<head lang="en">
    <meta charset="UTF-8">
    <title>{{ title }}</title>
</head>
<body>
    <h1>Centrosome Data Report - {{report_date}}</h1>
    {{ nuclei_data_html }}
</body>
</html>
"""
templ = j2.Template(master_template)
htmlout = templ.render({'title': 'Centrosomes report', 'nuclei_data_html': html})

with codecs.open('out/index.html', "w", "utf-8") as text_file:
    text_file.write(htmlout)

pdfkit.from_file('out/index.html', 'out/report.pdf')
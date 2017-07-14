import codecs
import os
import time

import jinja2 as j2
import matplotlib
import matplotlib.gridspec
import matplotlib.pyplot as plt
import pandas as pd
import pdfkit

import special_plots as sp


def plots_for_individual(df):
    if df['Nuclei'].unique().size > 1:
        raise IndexError('Just one nuclei per plot.')
    condition = df['condition'].unique()[0]
    run = df['run'].unique()[0]
    nuclei = df['Nuclei'].unique()[0]

    fig = plt.figure(5, figsize=[13, 7])
    plt.clf()
    gs = matplotlib.gridspec.GridSpec(7, 1)
    ax1 = plt.subplot(gs[0:2, 0])
    ax2 = plt.subplot(gs[2, 0])
    ax3 = plt.subplot(gs[3, 0])
    ax4 = plt.subplot(gs[4, 0])
    ax5 = plt.subplot(gs[5, 0])
    ax6 = plt.subplot(gs[6, 0])

    between_df = df[df['CentrLabel'] == 'A']
    sp.plot_distance_to_nucleus(df, ax1)
    sp.plot_speed_to_nucleus(df, ax2)
    sp.plot_acceleration_to_nucleus(df, ax3)
    sp.plot_distance_between_centrosomes(between_df, ax4)
    sp.plot_speed_between_centrosomes(between_df, ax5)
    sp.plot_acceleration_between_centrosomes(between_df, ax6)
    for _ax in [ax2, ax3, ax4, ax5, ax6]:
        _ax.set_ylabel(_ax.get_ylabel(), rotation='horizontal', ha='right', fontsize=9, weight='ultralight')

    _fname = '%s-%s-n%02d.svg' % (condition, run, nuclei)
    fig.savefig('_html/img/' + _fname, format='svg')
    return _fname


def html_centrosomes_condition_run_subreport(df):
    if df['condition'].unique().size > 1 or df['run'].unique().size > 1:
        raise IndexError('Just one condition-run per subreport.')

    condition = df['condition'].unique()[0]
    run = df['run'].unique()[0]

    htmlout = '<h3>Experiment Condition Group: %s run: %s</h3>' % (condition, run)

    for nuclei_id, filtered_nuc_df in df.groupby(['Nuclei']):
        nuclei_template = """
                <div class="nuclei_plot">
                    <h4>Nucleus ID: {{ nuclei_id }} ({{ condition_id }}-{{ run_id }})</h4>
                    <img src="img/{{ plot_filename }}" style="width: 20cm;">
                </div>
                <div style="page-break-after: always"></div>
            """
        plot_fname = plots_for_individual(filtered_nuc_df)

        templ = j2.Template(nuclei_template)
        htmlout += templ.render({
            'nuclei_id': nuclei_id,
            'condition_id': condition,
            'run_id': run,
            'plot_filename': plot_fname
        })

    return htmlout


if __name__ == '__main__':
    # make output directory
    if not os.path.isdir('_html/img'):
        os.makedirs('_html/img')
    _f = os.path.abspath('_html')

    df_disk = pd.read_pickle('/Users/Fabio/centrosomes.pandas')
    html_cond = ''
    for cond_id, df_cond in df_disk.groupby(['condition', 'run']):
        html_cond += html_centrosomes_condition_run_subreport(df_cond)

    master_template = """<!DOCTYPE html>
            <html>
            <head lang="en">
                <meta charset="UTF-8">
                <title>Centrosomes report</title>
            </head>
            <body>
                <h1>Centrosome Data Report - {{report_date}}</h1>
                <h2>Track Detail</h2>
                <div class="container">
                {{ html_render }}
                </div>
            </body>
            </html>
            """
    templj2_master = j2.Template(master_template)
    html_final = templj2_master.render(
        {'report_date': time.strftime('%d/%m/%Y'),
         'html_render': html_cond
         })

    with codecs.open('_html/index.html', "w", "utf-8") as text_file:
        text_file.write(html_final)

    pdfkit.from_file('_html/index.html', '/Users/Fabio/%s-report.pdf' % time.strftime('%Y%m%d'))
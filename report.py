import matplotlib
import numpy as np
import pandas as pd
import jinja2 as j2
import matplotlib.pyplot as plt
import imagej_pandas as ijdf
import codecs
import time


def subreport(conditions):
    html = '<h3></h3>'
    data = pd.DataFrame()
    stats = pd.DataFrame()
    for cond in conditions:
        for f in cond['files'][0:]:
            print f['name']
            dfij = ijdf.ImagejPandas(cond['path'] + f['name'], stats_df=stats)
            _html, _data = dfij.html_centrosomes_report(nuclei_list=f['nuclei_list'], max_time_dict=f['max_time_dict'],
                                                        centrosome_inclusion_dict=f['centrosome_inclusion_dict'],
                                                        centrosome_exclusion_dict=f['centrosome_exclusion_dict'],
                                                        centrosome_equivalence_dict=f['centrosome_equivalence_dict'],
                                                        joined_tracks=f['joined_tracks'])
            html += _html
            data = data.append(_data)
            stats = dfij.stats
    return html, stats, data


def plot_distance(data, filename=None):
    data.loc[data.index, 'Time'] = data.loc[data.index, 'Time'].round()
    # data = data.set_index('Time').sort_index()
    data = data.set_index('Frame').sort_index()
    data.index *= 5  # hack: compute mean using frame data, but plot it converting x axis into time

    plt.figure(20)
    plt.clf()
    gs = matplotlib.gridspec.GridSpec(4, 1)
    ax1 = plt.subplot(gs[0:2, 0])
    ax2 = plt.subplot(gs[2, 0])
    ax3 = plt.subplot(gs[3, 0])

    data.reset_index().plot(ax=ax1, kind='scatter', x='Frame', y='Dist', sharex=True, c='k', alpha=0.4)
    ax1.set_xlim(0, data.index.max())
    ax1.set_ylabel('both $[\mu m]$')
    ylim_gbl = ax1.get_ylim()

    axi = ax2
    d = data[data['Far']]
    t = d.index.unique()
    mean = d['Dist'].mean(level='Frame')
    stddev = d['Dist'].std(level='Frame')
    d.reset_index().plot(ax=axi, kind='scatter', x='Frame', y='Dist', sharex=True, c='k', alpha=0.4, marker='.')
    mean.plot(ax=axi, c='k')
    # axi.fill_between(t, mean + 1.5 * stddev, mean - 1.5 * stddev, alpha=0.2, color='k')
    axi.fill_between(t, mean + 1.0 * stddev, mean - 1.0 * stddev, alpha=0.2, color='k')
    axi.set_xlim(0, data.index.max())
    axi.set_ylim(ylim_gbl)
    axi.set_ylabel('far $[\mu m]$')

    axi = ax3
    nearmsk = data['Far'].map(lambda x: not x)
    d = data[nearmsk]
    t = d.index.unique()
    mean = d['Dist'].mean(level='Frame')
    stddev = d['Dist'].std(level='Frame')
    d.reset_index().plot(ax=axi, kind='scatter', x='Frame', y='Dist', sharex=True, c='k', alpha=0.4, marker='.')
    mean.plot(ax=axi, c='k')
    # axi.fill_between(t, mean + 1.5 * stddev, mean - 1.5 * stddev, alpha=0.2, color='k')
    axi.fill_between(t, mean + 1.0 * stddev, mean - 1.0 * stddev, alpha=0.2, color='k')
    axi.set_xlim(0, data.index.max())
    axi.set_ylim(ylim_gbl)
    axi.set_ylabel('close $[\mu m]$')

    # hack: compute mean using frame data, but plot it converting x axis into time
    axi.set_xlabel('Time $[min]$')

    if filename is not None:
        plt.savefig('out/img/%s.svg' % filename, format='svg')
    plt.close(20)


if __name__ == '__main__':
    html_pc = html_dyndic1 = ''

    pc_to_process = {'path': '/Users/Fabio/lab/PC/data/',
                     'files': [{
                         'name': 'centr-pc-0-table.csv',
                         'nuclei_list': [2, 3],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {2: [[200, 201]], 3: [[300, 301]]}
                     }, {
                         'name': 'centr-pc-1-table.csv',
                         'nuclei_list': [1, 2, 4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {4: [500]},
                         'centrosome_exclusion_dict': {4: [402]},
                         'centrosome_equivalence_dict': {1: [[100, 102, 103]]},
                         'joined_tracks': {2: [[200, 201]], 4: [[400, 401]]}
                     }, {
                         'name': 'centr-pc-3-table.csv',
                         'nuclei_list': [5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {5: [[500, 501]]}
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
                         'joined_tracks': {3: [[300, 301]]}
                     }, {
                         'name': 'centr-pc-12-table.csv',
                         'nuclei_list': [1],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {1: [102]},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {1: [[100, 101]]}
                     }, {
                         'name': 'centr-pc-14-table.csv',
                         'nuclei_list': [2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {2: [202, 203, 204, 205]},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-17-table.csv',
                         'nuclei_list': [1, 2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {1: [104, 105]},
                         'centrosome_equivalence_dict': {1: [[100, 102, 103]]},
                         'joined_tracks': {2: [[200, 201]]}
                     }, {
                         'name': 'centr-pc-18-table.csv',
                         'nuclei_list': [3, 4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {4: [402, 403]},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {3: [[300, 301]], 4: [[400, 401]]}
                     }, {
                         'name': 'centr-pc-200-table.csv',
                         'nuclei_list': [1, 5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {5: [500]},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {1: [[100, 101]], 5: [[501, 502]]}
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
                         'joined_tracks': {5: [[500, 501]]}
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
                         'joined_tracks': {7: [[700, 701]]}
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
                         'centrosome_exclusion_dict': {7: [700]},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {7: [[701, 702]]}
                     }, {
                         'name': 'centr-pc-209-table.csv',
                         'nuclei_list': [5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {5: [[500, 501]]}
                     }, {
                         'name': 'centr-pc-210-table.csv',
                         'nuclei_list': [3, 6],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {3: [[300, 301]], 6: [[600, 601]]}
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
                         'centrosome_equivalence_dict': {4: [[401, 402]]},
                         'joined_tracks': {4: [[400, 401]]}
                     }, {
                         'name': 'centr-pc-213-table.csv',
                         'nuclei_list': [1, 2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {2: [[200, 201]]}
                     }, {
                         'name': 'centr-pc-214-table.csv',
                         'nuclei_list': [5],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {5: [[500, 501]]}
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
                         'centrosome_equivalence_dict': {4: [[401, 402]]},
                         'joined_tracks': {4: [[400, 401]]}
                     }, {
                         'name': 'centr-pc-219-table.csv',
                         'nuclei_list': [4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {4: [[400, 402], [401, 403]]},
                         'joined_tracks': {4: [[400, 401]]}
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
                         'centrosome_equivalence_dict': {2: [[200, 202]]},
                         'joined_tracks': {2: [[200, 201]]}
                     }, {
                         'name': 'centr-pc-222-table.csv',
                         'nuclei_list': [2],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {2: [[200, 202, 203]]},
                         'joined_tracks': {}
                     }, {
                         'name': 'centr-pc-223-table.csv',
                         'nuclei_list': [5, 6, 7],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {50: [502]},
                         'centrosome_equivalence_dict': {5: [[500, 502]], 7: [[701, 702, 703]]},
                         'joined_tracks': {50: [[500, 501]], 6: [[600, 601]], 7: [[700, 701]]}
                     }, {
                         'name': 'centr-pc-224-table.csv',
                         'nuclei_list': [4],
                         'max_time_dict': {},
                         'centrosome_inclusion_dict': {},
                         'centrosome_exclusion_dict': {},
                         'centrosome_equivalence_dict': {},
                         'joined_tracks': {4: [[400, 401]]}
                     }
                     ]}
    df_d_to_nuclei_centr = pd.DataFrame([
        ['manual', 0, 0, 0, 'Contact', 'Cell', 2.9],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 5.8],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 3.3],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 8.9],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 5.1],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 3.3],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 5.1],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 1.8],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 6.9],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 8.7],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 6.2],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 2.0],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 7.1],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 7.8],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 1.3],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 10.4],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 5.6],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 7.1],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 9.8],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 8.7],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 6.4],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 7.1],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 4.9],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 5.0],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 0.7],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 15.1],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 8.2],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 4.0],
        ['manual', 0, 0, 0, 'Contact', 'Cell', 3.3],
    ], columns=['Tag', 'Nuclei', 'Frame', 'Time', 'Stat', 'Type', 'Dist'])

    html_pc, stats, data = subreport([pc_to_process])
    plot_distance(data, filename='distance_all_pc')
    data.to_pickle('out/dataframe_pc.pandas')
    stats = stats.append(df_d_to_nuclei_centr)
    stats.to_pickle('out/stats_pc.pandas')

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
                              'joined_tracks': {4: [[400, 401]]}
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
                              'joined_tracks': {3: [[300, 301]]}
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
                              'centrosome_inclusion_dict': {3: [103]},
                              'centrosome_exclusion_dict': {},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {3: [[103, 300]]}
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
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {3: [300]},
                              'centrosome_equivalence_dict': {},
                              'joined_tracks': {3: [[301, 302]]}
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
                              'nuclei_list': [2, 4],
                              'max_time_dict': {},
                              'centrosome_inclusion_dict': {},
                              'centrosome_exclusion_dict': {4: [401]},
                              'centrosome_equivalence_dict': {2: [[201, 202]]},
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
                                'nuclei_list': [1, 2, 3, 6, 90],
                                'max_time_dict': {},
                                'centrosome_inclusion_dict': {},
                                'centrosome_exclusion_dict': {6: [602, 603], 9: [903]},
                                'centrosome_equivalence_dict': {1: [[100, 102]], 6: [[601, 604, 605]],
                                                                90: [[900, 902]]},
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
                                'nuclei_list': [4, 10],
                                'max_time_dict': {},
                                'centrosome_inclusion_dict': {},
                                'centrosome_exclusion_dict': {10: [1002]},
                                'centrosome_equivalence_dict': {4: [[401, 402]]},
                                'joined_tracks': {}
                            },
                            ]}

    html_dyndic1, stats, data = subreport([dyndic1_to_process, dyncdk1as_to_process])
    plot_distance(data, filename='distance_all_dyndic1')
    data.to_pickle('out/dataframe_dyn.pandas')
    stats.to_pickle('out/stats_dyn.pandas')

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
                    <img src="img/beeswarm_vel_inout_NE.svg">
                    <img src="img/beeswarm_vel_tcon_NE.svg">
                    <img src="img/beeswarm_vel_farnear.svg">
                </div>

                <div class="container">
                <h3>Distance from nuclei center at time of contact</h3>
                    <img src="img/beeswarm_boxplot_pc_contact.svg">
                <h3>Distance from nuclei center at initial time and 100 mins after</h3>
                    <img src="img/beeswarm_boxplot_pc_snapshot.svg">
                </div>
                <h3>Distance of centrosomes from nuclei center over time</h3>
                    <img src="img/distance_all_pc.svg">
                </div>

                <h2>Condition: DynH1, DIC1 & DynCDK1as ({{dyndic1_n}} tracks)</h2>
                <h2>Brief</h2>
                <div class="container">
                <h3>Distance from nuclei center at time of contact</h3>
                    <img src="img/beeswarm_boxplot_dyndic1_contact.svg">
                <h3>Distance from nuclei center at initial time and 100 mins after</h3>
                    <img src="img/beeswarm_boxplot_dyndic1_snapshot.svg">
                </div>
                <h3>Distance of centrosomes from nuclei center over time</h3>
                    <img src="img/distance_all_dyndic1.svg">
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

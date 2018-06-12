import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import parameters as p
import stats

plt.style.use('ggplot')
sns.set(font_scale=0.9, context='paper', style='whitegrid')
pd.set_option('display.width', 320)

colors = ['windows blue', 'amber', 'greyish', 'faded green', 'dusty purple']
palette = sns.xkcd_palette(colors)

# print df['datetime'].apply(lambda t: t.strftime('%R'))

new_dist_name = 'Distance relative\nto nuclei center $[\mu m]$'
new_speed_name = 'Speed relative\nto nuclei center $\\left[\\frac{\mu m}{min} \\right]$'
new_distcntr_name = 'Distance between\ncentrosomes $[\mu m]$'
new_speedcntr_name = 'Speed between\ncentrosomes $\\left[\\frac{\mu m}{min} \\right]$'

# distribution of speed and tracks for speed filtered dataframe
if not os.path.isfile(p.data_dir + 'filt_tracks.pandas'):
    print 'computing'
    dfcentr = pd.read_pickle(p.data_dir + 'merge_centered.pandas')
    # filter original dataframe to get just data between centrosomes
    dfcentr.loc[:, 'indv'] = dfcentr['condition'] + '-' + dfcentr['run'] + '-' + dfcentr['Nuclei'].map(int).map(str) + \
                             '-' + dfcentr['CentrLabel']
    dfcentr.drop(['CentrLabel', 'Centrosome', 'NuclBound', 'CNx', 'CNy', 'CentX', 'CentY', 'NuclX', 'NuclY',
                  'CellBound', 'CellX', 'CellY', 'AccCentr', 'DistCell', 'SpdCell', 'AccCell'],
                 axis=1, inplace=True)

    sdf = dfcentr.replace([np.inf, -np.inf], np.nan)
    spd_df = trk_df = pd.DataFrame()
    for spd in np.arange(0.1, 1.0, 0.1):
        sdf.loc[sdf['SpeedCentr'] < spd, 'DistCentr'] = np.nan
        trklbldf = stats.extract_consecutive_timepoints(sdf)
        ctp = trklbldf.groupby(['condition', 'indv', 'timepoint_cluster_id']).size() \
            .reset_index(name='consecutive_time_points')
        ctp['filter_speed'] = spd
        spd_df = spd_df.append(ctp)
        trklbldf['filter_speed'] = spd
        trk_df = trk_df.append(trklbldf)

    trk_df.drop(['Dist', 'Speed', 'Acc'], axis=1, inplace=True)
    trk_df.to_pickle(p.data_dir + 'filt_tracks.pandas')
    spd_df.to_pickle(p.data_dir + 'filt_speedpoints.pandas')
    print 'computed'
else:
    trk_df = pd.read_pickle(p.data_dir + 'filt_tracks.pandas')
    spd_df = pd.read_pickle(p.data_dir + 'filt_speedpoints.pandas')
    print 'loaded'

# names = ['1_N.C.', '1_P.C.', '1_DIC', '1_Dynei', '1_CENPF', '1_BICD2', '2_Kines1', '2_CDK1_DK', '2_CDK1_DC']
# names = ['1_P.C.', '1_DIC', '1_Dynei', '1_CENPF']
# names = ['1_P.C.']
# spd_df = spd_df.loc[spd_df['condition'].isin(names)]
# trk_df = trk_df.loc[trk_df['condition'].isin(names)]


g = sns.FacetGrid(trk_df, row='condition', col='filter_speed', hue='timepoint_cluster_id')
g.map(plt.plot, 'Time', 'DistCentr')
g.map(plt.scatter, 'Time', 'DistCentr')
g.fig.suptitle('Centrosome pair speed distribution for speeds greater than filter_speed $[\mu m/min]$')
plt.savefig(p.data_dir + 'out/sfilt_cond_dist.pdf', format='pdf')

# plt.figure()
# sns.swarmplot(x='filter_speed', y='consecutive_time_points', hue='condition', data=spd_df)
# plt.savefig(p.data_dir + 'out/sfilt_cum.pdf', format='pdf')

plt.figure()
sns.barplot(x='filter_speed', y='consecutive_time_points', hue='condition', data=spd_df)
plt.savefig(p.data_dir + 'out/sfilt_bar.pdf', format='pdf')

# g = sns.FacetGrid(trk_df, row='indv', col='filter_speed', hue='timepoint_cluster_id')
# g.map(plt.plot, 'DistCentr')
# g.map(plt.scatter, 'DistCentr')
# g.fig.suptitle('Centrosome pair speed distribution for speeds greater than filter_speed $[\mu m/min]$')
# plt.savefig(p.data_dir + 'out/sfilt_dist.pdf', format='pdf')

# g = sns.FacetGrid(sdf, col='condition', col_wrap=4)
# g.map(sns.distplot, 'SpeedCentr', hist=False, rug=True)
# g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle('Centrosome pair speed distribution for speeds greater than %0.2f $[\mu m/min]$' % spd)
# plt.savefig(p.data_dir + 'out/sfilt_speed_%d.pdf' % (spd * 10), format='pdf')

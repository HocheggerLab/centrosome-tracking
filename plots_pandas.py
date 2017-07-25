import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import special_plots as sp

plt.style.use('ggplot')
sns.set(font_scale=0.9, context='paper', style='whitegrid')
pd.set_option('display.width', 320)

# flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
palette = sns.xkcd_palette(colors)

df = pd.read_pickle('/Users/Fabio/merge.pandas')
_df = pd.read_pickle('/Users/Fabio/merge_centered.pandas')

# congression plots
plt.figure(120)
df['indv'] = df['condition'] + '-' + df['run'] + '-' + df['Nuclei'].map(int).map(str)
sp.congression(df)
plt.savefig('/Users/Fabio/congression.svg', format='svg')

new_dist_name = 'Distance relative\nto nuclei center $[\mu m]$'
new_speed_name = 'Speed relative\nto nuclei center $\\left[\\frac{\mu m}{min} \\right]$'
new_distcntr_name = 'Distance between \ncentrosomes $[\mu m]$'
new_speedcntr_name = 'Speed between \ncentrosomes $\\left[\\frac{\mu m}{min} \\right]$'
df.rename(columns={'Dist': new_dist_name}, inplace=True)
df.rename(columns={'Speed': new_speed_name}, inplace=True)
df.rename(columns={'DistCentr': new_distcntr_name}, inplace=True)
df.rename(columns={'SpeedCentr': new_speedcntr_name}, inplace=True)
_df.rename(columns={'Dist': new_dist_name}, inplace=True)
_df.rename(columns={'Speed': new_speed_name}, inplace=True)
_df.rename(columns={'DistCentr': new_distcntr_name}, inplace=True)
_df.rename(columns={'SpeedCentr': new_speedcntr_name}, inplace=True)

# filter original dataframe to get just data between centrosomes
dfcentr = _df[_df['CentrLabel'] == 'A']
dfcentr['indv'] = dfcentr['condition'] + '-' + dfcentr['run'] + '-' + dfcentr['Nuclei'].map(int).map(str)
dfcentr.drop(['CentrLabel', 'Centrosome', 'NuclBound', 'CNx', 'CNy', 'CentX', 'CentY', 'NuclX', 'NuclY'],
             axis=1, inplace=True)

# average speed boxplot
plt.figure(100)
dfc_cleaned = dfcentr.loc[dfcentr['Time'] <= 0, :]
mua = dfc_cleaned.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
mua.rename(columns={'SpeedCentr': new_speedcntr_name}, inplace=True)
sp.anotated_boxplot(mua, new_speedcntr_name, stats_rotation='vertical', size=2)
plt.subplots_adjust(left=0.125, right=0.9, bottom=0.2, top=0.9, wspace=0.2, hspace=0.2)
plt.savefig('/Users/Fabio/boxplot_avg_speed.svg', format='svg')

plt.figure(102)
ax = plt.gca()
sns.tsplot(data=dfcentr, time='Frame', value=new_distcntr_name, unit='indv', condition='condition',
           estimator=np.nanmean, ax=ax)
ax.set_xlim([-10, 5])
plt.savefig('/Users/Fabio/dist_tsplot.svg', format='svg')

plt.figure(103)
ax = plt.gca()
sns.tsplot(data=dfcentr, time='Frame', value=new_speedcntr_name, unit='indv', condition='condition',
           estimator=np.nanmean, ax=ax)
ax.set_xlim([-10, 5])
plt.savefig('/Users/Fabio/speed_tsplot.svg', format='svg')

# distribution of speed against distance near time of contact
plt.figure(104)
dftof = dfcentr[(dfcentr['Time'] >= -30) & (dfcentr['Time'] <= 0)]
dftof.rename(columns={'DistCentr': new_distcntr_name}, inplace=True)
dftof.rename(columns={'SpeedCentr': new_speedcntr_name}, inplace=True)
sns.set_palette('GnBu_d')
g = sns.FacetGrid(dftof, col='condition', col_wrap=2, size=5)
g.map(plt.scatter, new_distcntr_name, new_speedcntr_name, s=1)
# g.map(sns.kdeplot, new_distcntr_name, new_speedcntr_name, lw=3)
g.add_legend()
plt.savefig('/Users/Fabio/speed_vs_dist_timeofcontact.svg', format='svg')

# distribution of speed against distance
plt.figure(105)
g = sns.FacetGrid(df, col='condition', col_wrap=2, size=5)
g.map(plt.scatter, new_distcntr_name, new_speedcntr_name, s=1)
# g.map(sns.kdeplot, new_distcntr_name, new_speedcntr_name, lw=3)
g.add_legend()
plt.savefig('/Users/Fabio/speed_vs_dist.svg', format='svg')

# distribution of distance
plt.figure(106)
g = sns.FacetGrid(df, row='condition', size=1.7, aspect=4)
g.map(sns.distplot, new_dist_name, hist=False, rug=True)
plt.savefig('/Users/Fabio/dist_distr.svg', format='svg')

column_order = ['1_P.C.', 'pc', '1_N.C.', '1_ASUND', '1_BICD2', '1_Bleb', '1_CENPF', '1_Cili+', '1_CyDT', '1_DIC',
                '1_Dynei', '1_FAKI', '1_MCAK', '1_No10+', '1_No20+', '1_No50+', '1_chTOG', '2_CDK1_+', '2_CDK1_A',
                '2_CDK1_DA', '2_CDK1_DC', '2_CDK1_DK', '2_CDK1_K', '2_Kines1', '2_MCAK', '2_cdTOG', 'hset',
                'hset+kif25', 'kif25']
# plot of every distance between centrosome's, centered at time of contact
plt.figure(110)
sns.set_palette('Set2')
df_idx_grp = dfcentr.set_index('Time').sort_index().reset_index()
g = sns.FacetGrid(df_idx_grp, col='condition', hue='indv', col_wrap=2, col_order=column_order, size=5)
g.map(plt.plot, 'Time', new_distcntr_name, linewidth=1, alpha=0.5)
plt.savefig('/Users/Fabio/dist_centrosomes_all.svg', format='svg')

# plot of every speed between centrosome's, centered at time of contact
plt.figure(111)
sns.set_palette('Set2')
df_idx_grp = dfcentr.set_index('Time').sort_index().reset_index()
g = sns.FacetGrid(df_idx_grp, col='condition', hue='indv', col_wrap=2, col_order=column_order, size=5)
g.map(plt.plot, 'Time', new_speedcntr_name, linewidth=1, alpha=0.5)
plt.savefig('/Users/Fabio/speed_centrosomes_all.svg', format='svg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import special_plots as sp
import stats

sns.set_style('whitegrid')
sns.set_context('paper')
sns.set(font_scale=0.9)
pd.set_option('display.width', 320)

df = pd.read_pickle('/Users/Fabio/centrosomes.pandas')
msk = pd.read_pickle('/Users/Fabio/mask.pandas')

new_dist_name = 'Distance relative\nto nuclei center $[\mu m]$'
new_speed_name = 'Speed relative\nto nuclei center $\\left[\\frac{\mu m}{min} \\right]$'
new_distcntr_name = 'Distance between \ncentrosomes $[\mu m]$'
new_speedcntr_name = 'Speed between \ncentrosomes $\\left[\\frac{\mu m}{min} \\right]$'
df.rename(columns={'Dist': new_dist_name}, inplace=True)
df.rename(columns={'Speed': new_speed_name}, inplace=True)
df.rename(columns={'DistCentr': new_distcntr_name}, inplace=True)
df.rename(columns={'SpeedCentr': new_speedcntr_name}, inplace=True)

# df['datetime'] = df['Time'].apply(
#     lambda x: pd.to_datetime(datetime.datetime.fromtimestamp(time.mktime(time.gmtime(x * 60.0))))
# )
_df = stats.dataframe_centered_in_time_of_contact(df)
# _df['Time'] -= _df['Time'].min()
dfcentr = _df[_df['CentrLabel'] == 'A'].drop(['CentrLabel', 'Centrosome', 'NuclBound',
                                              'CNx', 'CNy', 'CentX', 'CentY', 'NuclX', 'NuclY'], axis=1)
dfcentr['indv'] = dfcentr['condition'] + '-' + dfcentr['run'] + '-' + dfcentr['Nuclei'].map(int).map(str)

# average speed boxplot
mua = dfcentr.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
mua.rename(columns={'SpeedCentr': new_speed_name}, inplace=True)
sp.anotated_boxplot(mua, new_speed_name)
plt.savefig('/Users/Fabio/boxplot_avg_speed.svg', format='svg')

# plot of every distance between centrosome's, centered at time of contact
plt.figure(101)
sns.set_palette('GnBu_d')
df_idx_grp = dfcentr.set_index('Time').sort_index().groupby(['condition', 'run', 'Nuclei'])
df_idx_grp['DistCentr'].plot(linewidth=1, alpha=0.5)
plt.savefig('/Users/Fabio/dist_centrosomes_all.svg', format='svg')

plt.figure(102)
ax = plt.gca()
# flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
palette = sns.xkcd_palette(colors)
sns.set_palette(sns.color_palette(palette))
sns.tsplot(data=dfcentr, time='Frame', value='DistCentr', unit='indv', condition='condition', ax=ax)
ax.set_xlim([-10, 5])
plt.savefig('/Users/Fabio/dist_tsplot.svg', format='svg')

plt.figure(103)
ax = plt.gca()
sns.tsplot(data=dfcentr, time='Frame', value='SpeedCentr', unit='indv', condition='condition', ax=ax)
ax.set_xlim([-10, 5])
plt.savefig('/Users/Fabio/speed_tsplot.svg', format='svg')

# distribution of speed against distance near time of contact
dftof = dfcentr[(dfcentr['Time'] >= -30) & (dfcentr['Time'] <= 0)]
dftof.rename(columns={'DistCentr': new_distcntr_name}, inplace=True)
dftof.rename(columns={'SpeedCentr': new_speedcntr_name}, inplace=True)
sns.set_palette('GnBu_d')
g = sns.FacetGrid(dftof, col='condition', col_wrap=2, size=5)
g.map(plt.scatter, new_distcntr_name, new_speedcntr_name, s=1)
g.map(sns.kdeplot, new_distcntr_name, new_speedcntr_name, lw=3)
g.add_legend()
plt.savefig('/Users/Fabio/speed_vs_dist_timeofcontact.svg', format='svg')

# distribution of speed against distance
g = sns.FacetGrid(df, col='condition', col_wrap=2, size=5)
g.map(plt.scatter, new_dist_name, new_speed_name, s=1)
g.map(sns.kdeplot, new_dist_name, new_speed_name, lw=3)
# g.set(yscale='log')
g.add_legend()
plt.savefig('/Users/Fabio/speed_vs_dist.svg', format='svg')

# distribution of distance
g = sns.FacetGrid(df, row='condition', size=1.7, aspect=4)
g.map(sns.distplot, new_dist_name, hist=False, rug=True)
plt.savefig('/Users/Fabio/dist_distr.svg', format='svg')

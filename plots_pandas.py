import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import special_plots as sp

sns.set_style('whitegrid')
sns.set_context('paper')
sns.set(font_scale=0.9)
df = pd.read_pickle('/Users/Fabio/centrosomes.pandas')
msk = pd.read_pickle('/Users/Fabio/mask.pandas')

dfcentr = df[df['CentrLabel'] == 'A'].drop(['CentrLabel', 'Centrosome', 'NuclBound',
                                            'CNx', 'CNy', 'CentX', 'CentY', 'NuclX', 'NuclY',
                                            'Dist', 'Speed', 'Acc'], axis=1)
dfcentr['indv'] = dfcentr['condition'] + '-' + dfcentr['run'] + '-' + dfcentr['Nuclei'].map(int).map(str)
dfcentr.loc[dfcentr['SpeedCentr'] == 0, 'SpeedCentr'] = np.NaN

# average speed boxplot
mua = dfcentr.groupby(['condition', 'run', 'Nuclei']).mean().reset_index()
with open('/Users/Fabio/speed_nuclei_desc.txt', 'w') as f:
    str = mua.describe()
    f.writelines(str.to_string())
new_spd_name = 'Speed average relative to \nnucleus center [$\mu m$]'
mua.rename(columns={'SpeedCentr': new_spd_name}, inplace=True)
sp.anotated_boxplot(mua, new_spd_name)
plt.savefig('/Users/Fabio/boxplot_avg_speed.svg', format='svg')

# plot of every distance between centrosome's track, all together
plt.figure(101)
sns.set_palette('GnBu_d')
dfcentr.set_index('Time').sort_index().groupby(['condition', 'run', 'Nuclei'])['DistCentr'].plot(linewidth=1, alpha=0.5)
plt.savefig('/Users/Fabio/dist_centrosomes_all.svg', format='svg')

# distribution of speed against distance
new_dist_name = 'Distance relative to \nnucleus center [$\mu m$]'
df.rename(columns={'Dist': new_dist_name}, inplace=True)
g = sns.FacetGrid(df, col='condition', col_wrap=2, size=5)
# g.map(plt.scatter, new_dist_name, 'SpeedCentr', s=50, alpha=.7, linewidth=.5, edgecolor='white')
g.map(plt.scatter, new_dist_name, 'SpeedCentr', s=1)
g.map(sns.kdeplot, new_dist_name, 'SpeedCentr', lw=3)
# g.set(yscale='log')
g.add_legend()
plt.savefig('/Users/Fabio/dist_vs_speed.svg', format='svg')

# distribution of speed
g = sns.FacetGrid(df, row='condition', size=1.7, aspect=4)
g.map(sns.distplot, new_dist_name, hist=False, rug=True)

plt.show()

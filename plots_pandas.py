import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import special_plots as sp

sns.set_style('white')
sns.set(font_scale=0.9)
df = pd.read_pickle('/Users/Fabio/centrosomes.pandas')

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
dfcentr.set_index('Time').sort_index().groupby(['condition', 'run', 'Nuclei'])['DistCentr'].plot(linewidth=1)
plt.savefig('/Users/Fabio/dist_centrosomes_all.svg', format='svg')

# plot of every distance between centrosome's track, individually
g = sns.FacetGrid(dfcentr, col='indv', col_wrap=6)
g.map(plt.plot, 'Time', 'DistCentr')
g.add_legend()
plt.savefig('/Users/Fabio/dist_centrosomes.png', format='png')
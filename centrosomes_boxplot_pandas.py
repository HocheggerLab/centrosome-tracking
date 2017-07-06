import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from special_plots import anotated_boxplot

sns.set_style('white')

df = pd.read_pickle('/Users/Fabio/centrosomes.pandas')
# run207 = df[df['run'] == 'run_207'].drop(['NuclBound'], axis=1)
dfcentr = df[df['CentrLabel'] == 'A'].drop(['CentrLabel', 'Centrosome', 'NuclBound',
                                            'CNx', 'CNy', 'CentX', 'CentY', 'NuclX', 'NuclY',
                                            'Dist', 'Speed', 'Acc'], axis=1)
dfcentr['unit'] = dfcentr['condition'] + '-' + dfcentr['run'] + '-' + dfcentr['Nuclei'].map(int).map(str)

mua = df[df['CentrLabel'] == 'A'].groupby(['condition', 'run', 'Nuclei']).mean().reset_index()

plt.figure(100)
anotated_boxplot(mua, 'SpeedCentr')
with open('/Users/Fabio/desc.txt', 'w') as f:
    str = mua.describe()
    f.writelines(str.to_string())

plt.figure(101)
# sns.tsplot(data=dfcentr, time='Time', unit='unit', condition='condition', value='DistCentr')
dfcentr.set_index('Time').sort_index().groupby(['condition', 'run', 'Nuclei'])['DistCentr'].plot(linewidth=1)
# ax.set_ylim(ax1.get_ylim())



g = sns.FacetGrid(dfcentr, col='unit', col_wrap=6)
g.map(plt.plot, 'DistCentr')
g.add_legend()
plt.savefig('/Users/Fabio/dist_centrosomes.png', format='png')

plt.show()

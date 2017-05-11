import os
import re
import pandas as pd
import numpy as np
import codecs
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import jinja2 as j2
import pdfkit
import time

sns.set_style("white")


def preprocess_df(df):
    p = df[df['ValidCentroid'] == 1]
    p.loc[p.index, 'WhereInNuclei'] = p.loc[p.index, 'WhereInNuclei'].map(
        lambda i: 'Outside' if i == 0 else 'Touching' if i == 1  else 'Inside')
    p.loc[p.index, 'Far'] = p.loc[p.index, 'Far'].map(lambda i: 'Far' if i else 'Close')
    p.loc[p.index, 'Speed'] *= -1

    return p


def sw(data):
    mud = data.groupby(['exp', 'Nuclei', 'WhereInNuclei', 'Far']).mean().reset_index()
    ax = sns.swarmplot(data=mud, y='Speed', x='InsideNuclei')
    cat = mud['InsideNuclei'].unique()
    for c, x in zip(cat, range(len(cat) + 1)):
        d = mud[mud['InsideNuclei'] == c]['Speed']
        _max_y = ax.axis()[3]
        count = d.count()
        mean = d.mean()
        ax.text(x, _max_y - 0.05, '$\mu=%0.3f$' % mean, ha='center')
        ax.text(x, _max_y - 0.08, '$n=%d$' % count, ha='center')


df_pc = preprocess_df(pd.read_pickle('out/dataframe_pc.pandas'))
df_dyn = preprocess_df(pd.read_pickle('out/dataframe_dyn.pandas'))

# plt.figure(10)
# sw(df_pc)
#
# plt.figure(20)
# sw(df_dyn)

df_pc['Condition'] = 'STLC+'
df_dyn['Condition'] = 'Dyn'
a = df_pc.append(df_dyn)
mua = a.groupby(['Condition', 'exp', 'Nuclei', 'WhereInNuclei', 'Far']).mean().reset_index()

ymetric = 'Acc'
plt.figure(100)
ax = sns.swarmplot(data=mua, y=ymetric, x='Condition', hue='WhereInNuclei', split=True, order=['STLC+', 'Dyn'])
plt.savefig('out/img/beeswarm_vel_inout_NE.svg', format='svg')
with open('out/img/beeswarm_vel_inout_NE.desc.txt', 'w') as f:
    str = mua.groupby(['Condition', 'WhereInNuclei'])['Speed'].describe()
    f.writelines(str.to_string())

# fig = plt.figure(200)
# fig.suptitle('STLC+')
# ax1 = sns.swarmplot(data=mua[mua['Condition'] == 'STLC+'], y=ymetric, x='WhereInNuclei', hue='Far', split=True)
# plt.savefig('out/img/beeswarm_pc_vel_inout_NE.svg', format='svg')
#
# fig = plt.figure(201)
# fig.suptitle('Dyn DIC1 CDK1asDyn')
# ax2 = sns.swarmplot(data=mua[mua['Condition'] == 'Dyn'], y=ymetric, x='WhereInNuclei', hue='Far', split=True)
# ax2.set_ylim(ax1.get_ylim())
# plt.savefig('out/img/beeswarm_dyn_vel_inout_NE.svg', format='svg')

fig = plt.figure(300)
g = sns.FacetGrid(mua, row='WhereInNuclei', col='Condition')
# g.map(sns.swarmplot, hue='Far', split=True)
g.map(plt.hist, ymetric)
g.add_legend();

plt.show()

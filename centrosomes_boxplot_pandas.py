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
        lambda i: 'Inside' if i == 0 else 'Touching' if i == 1  else 'Outside')
    return p


def sw(data):
    mud = data.groupby(['exp', 'Nuclei', 'InsideNuclei', 'Far']).mean().reset_index()
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
mua = a.groupby(['Condition', 'exp', 'Nuclei', 'InsideNuclei', 'Far']).mean().reset_index()

plt.figure(100)
ax = sns.swarmplot(data=mua, y='Speed', x='Condition', hue='InsideNuclei', split=True, order=['STLC+', 'Dyn'])
plt.savefig('out/img/beeswarm_vel_inout_NE.svg', format='svg')
with open('out/img/beeswarm_vel_inout_NE.desc.txt', 'w') as f:
    str = mua.groupby(['Condition', 'InsideNuclei'])['Speed'].describe()
    f.writelines(str.to_string())

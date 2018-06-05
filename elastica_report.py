import ConfigParser
import json

import matplotlib
import matplotlib.axes
import matplotlib.gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

import elastica as e
import parameters

print font_manager.OSXInstalledFonts()
print font_manager.OSXFontDirectories

matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('svg', fonttype='none')
sns.set(context='paper', style='whitegrid', font='Arial', font_scale=0.9)
pd.set_option('display.width', 320)
plt.style.use('ggplot')

_fig_size_A3 = (11.7, 16.5)
_err_kws = {'alpha': 0.3, 'lw': 1}
msd_ylim = [0, 420]


def fig_1(df):
    with PdfPages(parameters.data_dir + 'out/elastica.pdf') as pdf:
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(_fig_size_A3)
        gs = matplotlib.gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0:2, 0:2])
        # ax2 = plt.subplot(gs[0, 1])
        # ax3 = plt.subplot(gs[1, 0])
        # ax4 = plt.subplot(gs[1, 1])
        ax5 = plt.subplot(gs[2, 0])
        ax6 = plt.subplot(gs[2, 1])

        with open('/Users/Fabio/elastica.cfg.txt', 'r') as configfile:
            config = ConfigParser.ConfigParser()
            config.readfp(configfile)

        print 'sections found in file ', config.sections()

        section = config.sections()[1]
        yn = np.array(json.loads(config.get(section, 'measure')))
        inip = np.array(json.loads(config.get(section, 'comment')))

        num_points = 100

        # ---------------------
        # plot initial model
        # ---------------------
        L, a1, a2, E, F, gamma, x0, y0, theta = inip
        s = np.linspace(0, L, num_points)
        r = e.heavy_planar_bvp(s, F=F, E=E, gamma=gamma)
        pol = r.sol
        xo = pol(s)[0:2, :]
        ys = e.eval_heavy_planar(s, pol, a1, a2)[0:2, :]

        # deal with rotations and translations
        sinth, costh = np.sin(theta), np.cos(theta)
        M = np.array([[costh, -sinth], [sinth, costh]])
        ys = np.matmul(M, ys) + np.array([x0, y0]).reshape((2, 1))
        xo = np.matmul(M, xo) + np.array([x0, y0]).reshape((2, 1))

        _ax = ax1
        _ax.plot(xo[0], xo[1], lw=3, c='r', zorder=2)
        _ax.plot(ys[0], ys[1], lw=5, c='g', zorder=1)
        _ax.scatter(yn[0], yn[1], c='g', marker='<', zorder=3)

        # ---------------------
        # plot estimations
        # ---------------------
        othr = 20
        filter_df = df[df['objfn'] < othr]
        print 'filtered %d rows' % len(filter_df.index)
        for row in filter_df.iterrows():
            row = row[1]
            s = np.linspace(0, row['L'], num_points)
            r = e.heavy_planar_bvp(s, F=row['F'], E=row['E'], gamma=row['gamma'])
            pol = r.sol
            xo = pol(s)[0:2, :]
            xs, ys = e.eval_heavy_planar(s, pol, row['a1'], row['a2'])[0:2, :]
            ys = np.array([xs, ys])

            # deal with rotations and translations
            sinth, costh = np.sin(row['theta']), np.cos(row['theta'])
            M = np.array([[costh, -sinth], [sinth, costh]])
            ys = np.matmul(M, ys) + np.array([row['x0'], row['y0']]).reshape((2, 1))
            xo = np.matmul(M, xo) + np.array([row['x0'], row['y0']]).reshape((2, 1))

            _ax.plot(xo[0], xo[1], lw=1, c='b', label='%0.1f' % row['E'], alpha=0.2, zorder=4)

        sns.distplot(df['objfn'], bins=20, ax=ax5)
        sns.distplot(filter_df['objfn'], bins=20, ax=ax6)

        fig.suptitle('Solutions with an objective function lower than %0.1f' % othr)
        ax1.set_title('Solutions with an objective function lower than %0.1f' % othr)
        ax5.set_title('Objective function of %d solutions' % len(df.index))
        ax6.set_title('Objective function of %d filtered solutions' % len(filter_df.index))

        pdf.savefig()
        plt.close()

        fig.clf()
        fig = matplotlib.pyplot.gcf()
        fig.clf()
        fig.set_size_inches(_fig_size_A3)
        gs = matplotlib.gridspec.GridSpec(4, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])
        ax5 = plt.subplot(gs[2, 0])
        ax6 = plt.subplot(gs[2, 1])
        ax7 = plt.subplot(gs[3, 0])
        ax8 = plt.subplot(gs[3, 1])

        sns.distplot(filter_df['E'], bins=20, ax=ax1)
        sns.distplot(filter_df['F'], bins=20, ax=ax2)
        sns.distplot(filter_df['gamma'], bins=20, ax=ax3)
        sns.distplot(filter_df['L'], bins=20, ax=ax4)
        sns.distplot(filter_df['a1'], bins=20, ax=ax5)
        # sns.distplot(df['a2'], bins=20, ax=ax2)
        sns.distplot(filter_df['x0'], bins=20, ax=ax7)
        sns.distplot(filter_df['y0'], bins=20, ax=ax8)
        sns.distplot(filter_df['theta'], bins=20, ax=ax6)

        ax1.axvline(E, ls='--', c='g')
        ax2.axvline(F, ls='--', c='g')
        ax3.axvline(gamma, ls='--', c='g')
        ax4.axvline(L, ls='--', c='g')
        ax5.axvline(a1, ls='--', c='g')
        ax7.axvline(x0, ls='--', c='g')
        ax8.axvline(y0, ls='--', c='g')
        ax6.axvline(theta, ls='--', c='g')

        ax1.axvline(filter_df['E'].mean(), ls='--', c='r')
        ax2.axvline(filter_df['F'].mean(), ls='--', c='r')
        ax3.axvline(filter_df['gamma'].mean(), ls='--', c='r')
        ax4.axvline(filter_df['L'].mean(), ls='--', c='r')
        ax5.axvline(filter_df['a1'].mean(), ls='--', c='r')
        ax7.axvline(filter_df['x0'].mean(), ls='--', c='r')
        ax8.axvline(filter_df['y0'].mean(), ls='--', c='r')
        ax6.axvline(filter_df['theta'].mean(), ls='--', c='r')

        print filter_df['E'].describe()
        print filter_df['F'].describe()
        print filter_df['gamma'].describe()
        print np.degrees(filter_df['gamma']).describe()

        for _ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
            red_patch = mpatches.Patch(color='red', label='Estimated value')
            green_patch = mpatches.Patch(color='green', label='Known value')
            red_line = mlines.Line2D([], [], color='red', ls='--', label='Estimated value')
            green_line = mlines.Line2D([], [], color='green', ls='--', label='Known value')
            _ax.legend(handles=[red_line, green_line])

        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    df = pd.read_csv(parameters.data_dir + 'elastica.csv')

    fig_1(df)

import matplotlib
import matplotlib.axes
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

import elastica as e

print font_manager.OSXInstalledFonts()
print font_manager.OSXFontDirectories

matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('svg', fonttype='none')
sns.set(context='paper', style='whitegrid', font='Arial', font_scale=0.9)
pd.set_option('display.width', 320)
plt.style.use('bmh')

_fig_size_A3 = (11.7, 16.5)
_err_kws = {'alpha': 0.3, 'lw': 1}
msd_ylim = [0, 420]


def fig_1(df):
    with PdfPages('/Users/Fabio/elastica.pdf') as pdf:
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

        yn = [[206.6262028467798, 206.5373595808336, 206.44615827128368, 206.35286369125276, 206.25773932717482,
               206.1610456472161, 206.06303848101155, 205.96396752589752, 205.86407499148964, 205.76359439112116,
               205.66274948542042, 205.56175338024133, 205.4608077783395, 205.36010238165497, 205.25981443885755,
               205.16010843095393, 205.06113588624933, 204.96303531480172, 204.86593225168227, 204.76993939784262,
               204.6751568471576, 204.58167238822745, 204.4895618697538, 204.39888961870935, 204.30970890107142,
               204.222062415548, 204.13598281146227, 204.05149322274968, 203.9686078108368, 203.88733230999014,
               203.80766456953305, 203.72959508810868, 203.65310753591447, 203.57817926153103, 203.50478178061934,
               203.43288124435347, 203.3624388859957, 203.2934114445033, 203.22575156448573, 203.1594081722055,
               203.094326827644, 203.0304500529326, 202.9677176376886, 202.90606692199424, 202.84543305792312,
               202.78574925065246, 202.72694698030824, 202.6689562057752, 202.6117055517688, 202.55512248051608],
              [255.03579327562517, 254.95194291011805, 255.15942289606917, 255.00575874281034, 255.30166549809204,
               255.26741508895608, 255.15971606601153, 255.36838326124249, 255.17269629850338, 255.3116723340672,
               254.983969504522, 255.39790457438554, 255.15076860538457, 255.0757930732049, 255.39782662044553,
               255.3013101128558, 254.9224618979214, 255.03841939144618, 255.15390877287567, 254.98355729734322,
               254.90483957242367, 255.19405122775552, 254.83078970222394, 254.9011479196835, 254.7929706383408,
               254.85245147794103, 254.94365715396995, 254.7745874652467, 254.39712138564036, 254.65546760248642,
               254.4741919612822, 254.28792484148227, 254.44375766512078, 254.49531134758118, 254.1693529419822,
               254.02851727587816, 254.24679960402682, 254.0869567513644, 254.08250778242714, 253.94716416386402,
               254.037690042932, 253.83729472638643, 253.65863816106426, 253.46308976806395, 253.2806185107044,
               253.65290210412272, 253.48920945351998, 253.47782531045988, 252.93776362278192, 253.15046847958638]]
        inip = [10.0, 0.1, 0.6, 1.0, 0.1, 1.5707963267948966, 200, 250, 0.5235987755982988]

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

        ax = ax1
        ax.plot(xo[0], xo[1], lw=3, c='r', zorder=2)
        ax.plot(ys[0], ys[1], lw=5, c='g', zorder=1)
        ax.scatter(yn[0], yn[1], c='g', marker='<', zorder=3)

        # ---------------------
        # plot estimations
        # ---------------------
        filter_df = df[df['objfn'] < 10]
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

            ax.plot(xo[0], xo[1], lw=1, c='b', label='%0.1f' % row['E'], alpha=0.2, zorder=4)

        sns.distplot(df['objfn'], bins=20, ax=ax5)
        sns.distplot(filter_df['objfn'], bins=20, ax=ax6)

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

        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    df = pd.read_csv('elastica.csv')

    fig_1(df)

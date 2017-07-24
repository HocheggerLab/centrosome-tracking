import numpy as np
import pandas as pd

from imagej_pandas import ImagejPandas


def p_values(data, var, group_label, filename):
    from scipy.stats import ttest_ind

    cat = data[group_label].unique()
    pmat = list()
    for c1 in cat:
        d1 = data[data[group_label] == c1][var]
        for c2 in cat:
            d2 = data[data[group_label] == c2][var]
            s, p = ttest_ind(d1, d2)
            pmat.append(p)
    pmat = np.array(pmat).reshape(len(cat), len(cat))
    np.savetxt('out/img/%s-pvalues.txt' % filename, pmat, header=','.join(cat), fmt='%.4e')


def dataframe_centered_in_time_of_contact(df):
    df_tofc_center = pd.DataFrame()
    for _, fdf in df.groupby(['condition', 'run', 'Nuclei']):
        fdf.loc[:, 'Centrosome'] = fdf['CentrLabel']
        time_of_c, frame_of_c, dist_of_c = ImagejPandas.get_contact_time(fdf, ImagejPandas.DIST_THRESHOLD)

        if time_of_c is None:
            time_of_c = fdf['Time'].max()
            frame_of_c = fdf['Frame'].max()

        _df = fdf[fdf['CentrLabel'] == 'A']
        _df.loc[:, 'Time'] -= time_of_c
        _df.loc[:, 'Frame'] -= frame_of_c

        df_tofc_center = df_tofc_center.append(_df)
    print 'done.'
    return df_tofc_center
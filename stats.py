import numpy as np
import pandas as pd

from imagej_pandas import ImagejPandas


def p_values(data, var, group_label, filename=None):
    from scipy.stats import ttest_ind

    cat = data[group_label].unique()
    df_p = pd.DataFrame(np.zeros([len(cat), len(cat)]), index=cat, columns=cat)

    for c1 in cat:
        d1 = data[data[group_label] == c1][var]
        for c2 in cat:
            d2 = data[data[group_label] == c2][var]
            s, p = ttest_ind(d1, d2)
            df_p.loc[c1, c2] = p

    if filename is not None:
        writer = pd.ExcelWriter(filename)
        df_p.to_excel(writer, 'p-values')
        star_fn = lambda x: '****' if x <= 0.0001 else '***' if x <= 0.001 else '**' if x <= 0.01 else '*' if x <= 0.05 \
            else 'ns'
        star = df_p.applymap(star_fn)
        star.to_excel(writer, 'star system')
        writer.save()

    return df_p.as_matrix()


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

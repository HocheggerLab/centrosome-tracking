import datetime
import time

import numpy as np
import pandas as pd

from imagej_pandas import ImagejPandas


def star_system(p_value):
    return '****' if p_value <= 0.0001 else '***' if p_value <= 0.001 else '**' if p_value <= 0.01 else '*' if p_value <= 0.05 \
        else 'ns(p=%0.2f)' % p_value


def p_values(data, var, group_label, filename=None):
    from scipy.stats import ttest_ind

    cat = data[group_label].unique()
    df_p = pd.DataFrame(np.zeros([len(cat), len(cat)]), index=cat, columns=cat)

    for c1 in cat:
        d1 = data[data[group_label] == c1][var]
        for c2 in cat:
            d2 = data[data[group_label] == c2][var]
            # FIXME: This test assumes that the populations have identical variances by default.
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
    for id, fdf in df.groupby(['condition', 'run', 'Nuclei']):
        fdf.loc[:, 'Centrosome'] = fdf['CentrLabel']
        time_of_c, frame_of_c, dist_of_c = ImagejPandas.get_contact_time(fdf, ImagejPandas.DIST_THRESHOLD)
        # print id, time_of_c
        _df = fdf.loc[fdf['CentrLabel'].isin(['A', 'B'])]
        if time_of_c is None:
            fdfi = fdf.set_index('Time')
            time_a = fdfi.loc[fdfi['CentrLabel'] == 'A', 'Dist'].dropna().index.max()
            time_b = fdfi.loc[fdfi['CentrLabel'] == 'B', 'Dist'].dropna().index.max()
            # print time_a, time_b,
            time_of_c = min(time_a, time_b)
            frame_of_c = fdfi.iloc[-1]['Frame']
        # print time_of_c, frame_of_c

        _df.loc[:, 'Time'] -= time_of_c
        _df.loc[:, 'Frame'] -= frame_of_c
        if 'datetime' in _df:
            dt = pd.to_datetime(datetime.datetime.fromtimestamp(time.mktime(time.gmtime(time_of_c * 60.0))))
            _df.loc[:, 'datetime'] -= dt

        df_tofc_center = df_tofc_center.append(_df)
    return df_tofc_center


def baseround(x, base=5):
    return int(base * round(float(x) / base))


def reconstruct_time(df):
    # reconstruct time of tracks analyzed with Fiji plugin to match Matlab
    odf = pd.DataFrame()
    for _, _df in df.groupby(ImagejPandas.CENTROSOME_INDIV_INDEX):
        delta = baseround(_df['Time'][0:2].diff().iloc[1])
        _df.loc[:, 'Time'] = _df['Frame'] * delta
        odf = odf.append(_df)
    return odf


def extract_consecutive_timepoints(df):
    odf = pd.DataFrame()
    i = 0
    edge_rise = False
    for _ci, _df in df.groupby('indv'):
        for _ti, _tdf in _df.groupby('Frame'):
            if np.isnan(_tdf['DistCentr'].item()):
                if edge_rise:
                    edge_rise = False
            else:
                if not edge_rise:
                    edge_rise = True
                    i += 1
                _tdf['timepoint_cluster_id'] = i
                odf = odf.append(_tdf)
        i += 1
    return odf

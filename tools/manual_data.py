import logging
import numpy as np
import pandas as pd

from imagej_pandas import ImagejPandas

log = logging.getLogger(__name__)


def gen_dist_data(df):
    stats = pd.DataFrame()
    dt_before_contact = 30
    t_per_frame = 5
    d_thr = ImagejPandas.DIST_THRESHOLD
    for i, (id, idf) in enumerate(df.groupby(ImagejPandas.NUCLEI_INDIV_INDEX)):
        log.debug(id)

        time_of_c, frame_of_c, dist_of_c = ImagejPandas.get_contact_time(idf, d_thr)
        print([time_of_c, frame_of_c, dist_of_c])
        if time_of_c is not None:
            frame_before = frame_of_c - dt_before_contact / t_per_frame
            if frame_before < 0:
                frame_before = 0
            dists_before_contact = idf[idf['Frame'] == frame_before]['Dist'].values
            min_dist = max_dist = time_before = np.NaN
            if len(dists_before_contact) > 0:
                max_dist = max(dists_before_contact)
                min_dist = min(dists_before_contact)
                time_before = idf[idf['Frame'] == frame_before]['Time'].unique()[0]
        else:
            frame_before = time_before = min_dist = max_dist = np.NaN

        df_row1 = pd.DataFrame({'Tag': [id],
                                'Nuclei': idf['Nuclei'].unique()[0],
                                'Frame': [frame_before],
                                'Time': [time_before],
                                'Stat': '30Before',
                                'Type': 'C1 (Away)',
                                'Dist': [max_dist]})
        df_row2 = pd.DataFrame({'Tag': [id],
                                'Nuclei': idf['Nuclei'].unique()[0],
                                'Frame': [frame_before],
                                'Time': [time_before],
                                'Stat': '30Before',
                                'Type': 'C2 (Close)',
                                'Dist': [min_dist]})
        df_rown = pd.DataFrame({'Tag': [id],
                                'Nuclei': idf['Nuclei'].unique()[0],
                                'Frame': [frame_of_c],
                                'Time': [time_of_c],
                                'Stat': 'Contact',
                                'Type': 'Nucleus\nCentroid',
                                'Dist': [dist_of_c]})
        df_rowc = pd.DataFrame({'Tag': [id],
                                'Nuclei': idf['Nuclei'].unique()[0],
                                'Frame': [frame_of_c],
                                'Time': [time_of_c],
                                'Stat': 'Contact',
                                'Type': 'Cell\nCentroid',
                                'Dist': idf.loc[idf['Frame'] == frame_of_c, 'DistCell'].min()})
        stats = stats.append(df_row1, ignore_index=True)
        stats = stats.append(df_row2, ignore_index=True)
        stats = stats.append(df_rown, ignore_index=True)
        stats = stats.append(df_rowc, ignore_index=True)

    df_d_to_nuclei_centr = pd.DataFrame([
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 2.9],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 5.8],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 3.3],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 8.9],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 5.1],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 3.3],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 5.1],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 1.8],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 6.9],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 8.7],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 6.2],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 2.0],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 7.1],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 7.8],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 1.3],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 10.4],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 5.6],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 7.1],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 9.8],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 8.7],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 6.4],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 7.1],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 4.9],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 5.0],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 0.7],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 15.1],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 8.2],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 4.0],
        ['manual', 0, 0, 0, 'Contact', 'Cell\n(manual)', 3.3],
    ], columns=['Tag', 'Nuclei', 'Frame', 'Time', 'Stat', 'Type', 'Dist'])
    stats = stats.append(df_d_to_nuclei_centr, ignore_index=True)

    # sdata = stats[(stats['Stat'] == 'Contact') & (stats['Dist'].notnull())][['Dist', 'Type']]
    sdata = stats[stats['Dist'].notnull()]
    sdata['Dist'] = sdata.Dist.astype(np.float64)  # fixes a bug of seaborn

    log.debug('individuals for boxplot:\r\n%s' %
              sdata[(sdata['Stat'] == 'Contact') & (sdata['Type'] == 'Cell\nCentroid')])

    return sdata

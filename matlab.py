import numpy as np
import pandas as pd
import scipy.io as sio

import parameters
import stats
from imagej_pandas import ImagejPandas

pd.set_option('display.width', 320)

x = sio.loadmat(parameters.data_dir + 'finalData_2016_10_07.mat', squeeze_me=True)
exp_names = x['namesExperiment']
mtr_names = x['metricsName']
metrics = x['metrics']
tracks = x['tracks']

print len(exp_names), len(mtr_names), len(metrics), len(tracks)
df_matlab = pd.DataFrame()

for set_i in range(2):
    print tracks[set_i].shape, ',', 'conditions in the set.'
    for cond_i in range(len(tracks[set_i])):
        exp_name_cond_i = '%d_%s' % (set_i + 1, exp_names[set_i][cond_i].strip())
        print exp_name_cond_i, set_i + 1, cond_i + 1, tracks[set_i][cond_i].shape, ',',
        for run_j in range(len(tracks[set_i][cond_i])):
            print tracks[set_i][cond_i][run_j].shape, ',',
            for ind_k in range(len(tracks[set_i][cond_i][run_j])):
                track_ind_k = tracks[set_i][cond_i][run_j][ind_k]
                print track_ind_k.shape,

                if len(track_ind_k.shape) > 1:
                    c_a = {'Nuclei': ind_k,
                           'Centrosome': 1,
                           'CentrLabel': 'A',
                           'condition': exp_name_cond_i,
                           'run': 'mat_%d%02d' % (cond_i, run_j),
                           'Frame': range(track_ind_k.shape[0]),
                           'CentX': track_ind_k[:, 0],
                           'CentY': track_ind_k[:, 1],
                           'NuclX': np.zeros(track_ind_k.shape[0]),
                           'NuclY': np.zeros(track_ind_k.shape[0]),
                           }
                    c_b = {'Nuclei': ind_k,
                           'Centrosome': 2,
                           'CentrLabel': 'B',
                           'condition': exp_name_cond_i,
                           'run': 'mat_%d%02d' % (cond_i, run_j),
                           'Frame': range(track_ind_k.shape[0]),
                           'CentX': track_ind_k[:, 3],
                           'CentY': track_ind_k[:, 4],
                           'NuclX': np.zeros(track_ind_k.shape[0]),
                           'NuclY': np.zeros(track_ind_k.shape[0]),
                           }
                    _df = pd.DataFrame(c_a).append(pd.DataFrame(c_b))
                    df_matlab = df_matlab.append(_df)

        print

print df_matlab['condition'].unique(), len(df_matlab['condition'].unique())

# add time for every track
df_matlab.loc[:, 'Time'] = df_matlab.loc[:, 'Frame'] * 5.0

print 'Computing speed and acceleration for the dataset'
df_matlab = ImagejPandas.vel_acc_nuclei(df_matlab)
df_out = pd.DataFrame()
for _, _df in df_matlab.groupby(['condition', 'run', 'Nuclei']):
    _df = ImagejPandas.dist_vel_acc_centrosomes(_df)
    maxframe1 = _df.loc[_df['CentrLabel'] == 'A', 'Frame'].max()
    maxframe2 = _df.loc[_df['CentrLabel'] == 'B', 'Frame'].max()
    maxframedc = _df['Frame'].max()
    minframe1 = min(maxframe1, maxframedc)

    idx1 = (_df['CentrLabel'] == 'A') & (_df['Frame'] <= minframe1)
    _df.loc[idx1, 'SpeedCentr'] *= -1
    _df.loc[idx1, 'AccCentr'] *= -1
    df_out = df_out.append(_df)

ordered_columns = ['condition', 'run', 'Nuclei', 'Centrosome', 'CentrLabel',
                   'Frame', 'Time', 'CentX', 'CentY', 'NuclX', 'NuclY', 'CNx', 'CNy',
                   'Dist', 'Speed', 'Acc', 'DistCentr', 'SpeedCentr', 'AccCentr', 'NuclBound',
                   'CellX', 'CellY', 'DistCell', 'SpdCell', 'AccCell', 'CellBound']
df_out = df_out.assign(NuclBound=np.nan, CellX=np.nan, CellY=np.nan, DistCell=np.nan,
                       SpdCell=np.nan, AccCell=np.nan, CellBound=np.nan)
df_out = df_out[ordered_columns]

df_c = pd.read_pickle(parameters.data_dir + 'centrosomes.pandas')
df_c = df_c.append(df_out)
df_c = df_c[ordered_columns]
df_c.to_pickle(parameters.data_dir + 'merge.pandas')

print 'Re-centering timeseries around time of contact...'
df_c = stats.dataframe_centered_in_time_of_contact(df_c)
df_c.to_pickle(parameters.data_dir + 'merge_centered.pandas')

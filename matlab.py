import numpy as np
import pandas as pd
import scipy.io as sio

from imagej_pandas import ImagejPandas

pd.set_option('display.width', 320)

x = sio.loadmat('/Users/Fabio/finalData_2016_10_07.mat', squeeze_me=True)
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
        print exp_name_cond_i, tracks[set_i][cond_i].shape, ',',
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
_idx = df_matlab['condition'] == '2_MCAK'
df_matlab.loc[_idx, 'Time'] = df_matlab.loc[_idx, 'Frame'] * 1.0
df_matlab.loc[~_idx, 'Time'] = df_matlab.loc[~_idx, 'Frame'] * 5.0

# transform coordinates from pixels to um


# add speed and acceleration calculations
df_matlab = ImagejPandas.vel_acc_nuclei(df_matlab)
df_out = pd.DataFrame()
for _, _df in df_matlab.groupby(['condition', 'run', 'Nuclei']):
    dc = ImagejPandas.dist_vel_acc_centrosomes(_df)
    maxframe1 = _df.loc[_df['CentrLabel'] == 'A', 'Frame'].max()
    maxframe2 = _df.loc[_df['CentrLabel'] == 'B', 'Frame'].max()
    maxframedc = dc['Frame'].max()
    minframe1 = min(maxframe1, maxframedc)
    minframe2 = min(maxframe2, maxframedc)

    idx1 = (_df['CentrLabel'] == 'A') & (_df['Frame'] <= minframe1)
    idx2 = (_df['CentrLabel'] == 'B') & (_df['Frame'] <= minframe2)
    _df.loc[idx1, 'DistCentr'] = dc[dc['Frame'] <= minframe1]['DistCentr'].values
    _df.loc[idx1, 'SpeedCentr'] = -dc[dc['Frame'] <= minframe1]['SpeedCentr'].values
    _df.loc[idx1, 'AccCentr'] = -dc[dc['Frame'] <= minframe1]['AccCentr'].values
    _df.loc[idx2, 'DistCentr'] = -dc[dc['Frame'] <= minframe2]['DistCentr'].values
    _df.loc[idx2, 'SpeedCentr'] = dc[dc['Frame'] <= minframe2]['SpeedCentr'].values
    _df.loc[idx2, 'AccCentr'] = dc[dc['Frame'] <= minframe2]['AccCentr'].values
    df_out = df_out.append(_df)

df_out.to_pickle('/Users/Fabio/matlab.pandas')

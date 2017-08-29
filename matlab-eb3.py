import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

pd.set_option('display.width', 320)

fname = '/Users/Fabio/data/lab/eb3-control/data/Result of U2OS CDK1as EB3 +1NM on controls only.sld - Capture 1/TrackingPackage/tracks/Channel_1_tracking_result.mat'
x = sio.loadmat(fname, squeeze_me=True)
print x
tracks = x['tracksFinal']
tracked_particles = tracks['tracksCoordAmpCG']
trk_events = tracks['seqOfEvents']
num_tracks = tracked_particles.shape[0]
df_matlab = pd.DataFrame()

for ti, (_trk, _ev) in enumerate(zip(tracked_particles, trk_events)):
    trk = np.reshape(_trk, [len(_trk) / 8, 8])
    _df = pd.DataFrame(data=trk[:, 0:2], columns=['x', 'y'])
    _df['trk'] = ti

    df_matlab = df_matlab.append(_df)

df_matlab.groupby('trk').plot(x='x', y='y')
plt.show()
# print df_matlab['condition'].unique(), len(df_matlab['condition'].unique())

df_matlab.to_pickle('/Users/Fabio/eb3.pandas')

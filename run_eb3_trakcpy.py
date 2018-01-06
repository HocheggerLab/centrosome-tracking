import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
import trackpy as tp
import trackpy.predict


def trshow(tr, first_style='bo', last_style='gs', style='b.'):
    frames = list(tr.groupby('frame'))
    nframes = len(frames)
    for i, (fnum, pts) in enumerate(frames):
        if i == 0:
            sty = first_style
        elif i == nframes - 1:
            sty = last_style
        else:
            sty = style
            plt.plot(pts.x, pts.y, sty)
    trackpy.plot_traj(tr, colorby='frame', ax=plt.gca())
    plt.axis('equal');
    plt.ylim(ymin=-1.0, ymax=3.5)
    plt.xlabel('x')
    plt.ylabel('y')


file = '/Users/Fabio/data/lab/eb3/eb3-chtog/20171017/Result of U2OS CDK1as EB3 chTOG RNAi 3days +1NM on.sld - Capture 1.tif'
with tf.TiffFile(file, fastij=True) as tif:
    if tif.is_imagej is not None:
        dt = tif.pages[0].imagej_tags.finterval
        res = 'n/a'
        if tif.pages[0].resolution_unit == 'centimeter':
            # asuming square pixels
            xr = tif.pages[0].x_resolution
            res = float(xr[0]) / float(xr[1])  # pixels per cm
            res = res / 1e4  # pixels per um
        elif tif.pages[0].imagej_tags.unit == 'micron':
            # asuming square pixels
            xr = tif.pages[0].x_resolution
            res = float(xr[0]) / float(xr[1])  # pixels per um

frames = tif.pages[0].asarray()

diam = np.ceil(1 * res) // 2 * 2 + 1
f = tp.locate(frames[10], diam, invert=True)
print(f.head())

plt.figure()  # make a new figure
tp.annotate(f, frames[10])

fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)
# Optionally, label the axes.
ax.set(xlabel='mass', ylabel='count')

plt.figure()
f = tp.locate(frames[10], diam, invert=True, minmass=500)
tp.annotate(f, frames[10], plot_style={'markersize': diam})

f = tp.batch(frames[2:], diam, invert=True, minmass=200)
pred = trackpy.predict.NearestVelocityPredict()
t = pred.link_df(f, 5)
# t = tp.link_df(f, 5, memory=3)
trshow(t)

plt.figure()
tp.mass_size(t.groupby('particle').mean())
t2 = t[((t['mass'] > 650) & (t['size'] < 1.5) & (t['ecc'] < 0.26))]

print(t.head())
plt.figure()
tp.plot_traj(t2)

plt.show()

import argparse
import datetime
import os
import re
from subprocess import call

import h5py
import numpy as np
import pandas as pd
import tifffile as tf

from imagej_pandas import ImagejPandas as dfij


class LabHDF5NeXusFile():
    def __init__(self, filename='fabio_data_hochegger_lab.nexus.hdf5', fileflag='r'):
        self.filename = filename

        # open the HDF5 NeXus file
        if fileflag in ['w', 'a', 'r+']:
            with h5py.File(self.filename, fileflag) as f:
                # point to the default data to be plotted
                f.attrs['default'] = 'entry'
                # give the HDF5 root some more attributes
                f.attrs['file_name'] = self.filename
                timestamp = datetime.datetime.now().isoformat()
                f.attrs['file_time'] = timestamp
                f.attrs['creator'] = 'fabio@hochegger.lab'
                f.attrs['NeXus_version'] = '4.3.0'
                f.attrs['HDF5_Version'] = h5py.version.hdf5_version
                f.attrs['h5py_version'] = h5py.version.version

    def add_experiment(self, group, experiment_tag, timestamp=None, tags=None):
        with h5py.File(self.filename, 'a') as f:
            timestamp = timestamp if timestamp is not None else datetime.datetime.now().isoformat()

            gr = '%s/%s' % (group, experiment_tag)
            if gr in f: del f[gr]

            # create the NXentry experiment group
            nxentry = f.create_group(gr)
            nxentry.attrs['NX_class'] = 'NXentry'
            nxentry.attrs['datetime'] = timestamp
            nxentry.attrs['default'] = 'processed'

            nxentry_raw = nxentry.create_group('raw')
            nxentry_raw.attrs['NX_class'] = 'NXgroup'
            nxentry_meas = nxentry.create_group('measurements')
            nxentry_meas.attrs['NX_class'] = 'NXgroup'
            nxentry_sel = nxentry.create_group('selection')
            nxentry_sel.attrs['NX_class'] = 'NXgroup'
            nxentry_proc = nxentry.create_group('processed')
            nxentry_proc.attrs['NX_class'] = 'NXgroup'

    def add_tiff_sequence(self, tiffpath, experiment_tag, run):
        print 'adding tiff:', tiffpath

        # open the HDF5 NeXus file
        f = h5py.File(self.filename, 'a')

        # create the NXentry group
        nxdata = f['%s/%s/raw' % (experiment_tag, run)]
        nxdata.attrs['NX_class'] = 'NXdata'
        nxdata.attrs['signal'] = 'Y'  # Y axis of default plot
        nxdata.attrs['axes'] = 'X'  # X axis of default plot
        nxdata.attrs['units'] = 'um'  # default units

        with tf.TiffFile(tiffpath, fastij=True) as tif:
            if tif.is_imagej is not None:
                sizeT, channels = tif.pages[0].imagej_tags.frames, tif.pages[0].imagej_tags.channels
                sizeZ, sizeX, sizeY = 1, tif.pages[0].image_width, tif.pages[0].image_length

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

                if sizeT > 1:
                    p1a = tif.pages[0].asarray()
                    for i in range(sizeT):
                        outImg = np.zeros((1, channels, sizeX, sizeY), np.uint16)
                        outImg[0][0] = p1a[i][0]
                        # create a NXentry frame group
                        nxframe = nxdata.create_group('%03d' % i)
                        nxframe.attrs['units'] = 'um'
                        nxframe.attrs['resolution'] = res
                        nxframe.attrs['long_name'] = 'image um (micrometers)'  # suggested X axis plot label

                        # save XY data
                        ch1 = nxframe.create_dataset('channel-1', data=p1a[channels * i], dtype=np.uint16)
                        ch2 = nxframe.create_dataset('channel-2', data=p1a[channels * i + 1], dtype=np.uint16)
                        ch3 = nxframe.create_dataset('channel-3', data=p1a[channels * i + 2], dtype=np.uint16)
                        for ch in [ch1, ch2, ch3]:
                            ch.attrs['CLASS'] = np.string_('IMAGE')
                            ch.attrs['IMAGE_SUBCLASS'] = np.string_('IMAGE_GRAYSCALE')
                            ch.attrs['IMAGE_VERSION'] = np.string_('1.2')
        f.close()

    def add_measurements(self, csvpath, experiment_tag, run):
        dfc = dfij(csvpath)
        with h5py.File(self.filename, 'a') as f:
            nxmeas = f['%s/%s/measurements' % (experiment_tag, run)]

            dfnt = dfc.df_nuclei.set_index('Frame').sort_index()
            for (nuc_id), filt_nuc_df in dfnt.groupby('Nuclei'):
                nxnucl = nxmeas.create_group('nuclei/N%02d' % nuc_id)
                nxnucl.attrs['NX_class'] = 'NXdata'
                nxnucl.attrs['Nuclei'] = 'N%02d' % nuc_id
                nx = filt_nuc_df['NuclX']
                ny = filt_nuc_df['NuclY']
                fn = filt_nuc_df.reset_index()
                nxnucl.create_dataset('pos', data=fn[['Frame', 'NuclX', 'NuclY']], dtype=nx.dtype)
                nxnucl.create_dataset('frame', data=fn['Frame'], dtype=nx.dtype)
                nxnucl.create_dataset('sample_x', data=nx, dtype=nx.dtype)
                nxnucl.create_dataset('sample_y', data=ny, dtype=ny.dtype)

            dfct = dfc.df_centrosome.set_index('Frame').sort_index()

            for (centr_id), filt_centr_df in dfct.groupby('Centrosome'):
                nxcid = nxmeas.create_group('centrosomes/C%03d' % centr_id)
                nxcid.attrs['NX_class'] = 'NXdata'
                cx = filt_centr_df['CentX']
                cy = filt_centr_df['CentY']
                cn = filt_centr_df.reset_index()
                nxcid.create_dataset('pos', data=cn[['Frame', 'CentX', 'CentY']], dtype=cx.dtype)
                nxcid.create_dataset('frame', data=cn['Frame'], dtype=cx.dtype)
                nxcid.create_dataset('sample_x', data=cx, dtype=cx.dtype)
                nxcid.create_dataset('sample_y', data=cy, dtype=cy.dtype)

            visitedCentrosomes = []
            for (centr_id), filt_centr_df in dfct.groupby('Centrosome'):
                nuc_id = filt_centr_df['Nuclei'].unique()[0]
                centrosomesOfNuclei = dfct[dfct['Nuclei'] == nuc_id].groupby('Centrosome')
                if len(centrosomesOfNuclei.groups) >= 2:
                    for i, ((centr_id), filt_centr_df) in enumerate(centrosomesOfNuclei):
                        if centr_id not in visitedCentrosomes:
                            visitedCentrosomes.append(centr_id)
                            self.associate_centrosome_with_nuclei(centr_id, nuc_id, experiment_tag, run, i % 2)
        dfc.merged_df.to_hdf(self.filename, '%s/%s/measurements/pandas_dataframe' % (experiment_tag, run), mode='r+')
        dfc.df_nuclei.to_hdf(self.filename, '%s/%s/measurements/nuclei_dataframe' % (experiment_tag, run), mode='r+')
        self.process_selection_for_run(experiment_tag, run)

    @property
    def dataframe(self):
        out = pd.DataFrame()
        with h5py.File(self.filename, 'r') as f:
            for experiment_tag in f:
                for run in f['%s' % experiment_tag]:
                    if 'pandas_dataframe' in f['%s/%s/processed' % (experiment_tag, run)]:
                        selection_key = '%s/%s/processed/pandas_dataframe' % (experiment_tag, run)
                        df = pd.read_hdf(self.filename, key=selection_key, mode='r')
                        df['condition'] = experiment_tag
                        df['run'] = run
                        for nuc_id, df_nuc in df.groupby('Nuclei'):
                            print 'getting run %s nuclei N%02d' % (run, nuc_id)
                            dc = dfij.dist_vel_acc_centrosomes(df_nuc)
                            if len(dc) > 0:
                                maxframe1 = df_nuc.loc[df_nuc['CentrLabel'] == 'A', 'Frame'].max()
                                maxframe2 = df_nuc.loc[df_nuc['CentrLabel'] == 'B', 'Frame'].max()
                                maxframedc = dc['Frame'].max()
                                minframe1 = min(maxframe1, maxframedc)
                                minframe2 = min(maxframe2, maxframedc)

                                idx1 = (df_nuc['CentrLabel'] == 'A') & (df_nuc['Frame'] <= minframe1)
                                idx2 = (df_nuc['CentrLabel'] == 'B') & (df_nuc['Frame'] <= minframe2)

                                df_nuc.loc[idx1, 'DistCentr'] = dc[dc['Frame'] <= minframe1]['DistCentr'].values
                                df_nuc.loc[idx1, 'SpeedCentr'] = dc[dc['Frame'] <= minframe1]['SpeedCentr'].values
                                df_nuc.loc[idx1, 'AccCentr'] = dc[dc['Frame'] <= minframe1]['AccCentr'].values
                                df_nuc.loc[idx2, 'DistCentr'] = -dc[dc['Frame'] <= minframe2]['DistCentr'].values
                                df_nuc.loc[idx2, 'SpeedCentr'] = -dc[dc['Frame'] <= minframe2]['SpeedCentr'].values
                                df_nuc.loc[idx2, 'AccCentr'] = -dc[dc['Frame'] <= minframe2]['AccCentr'].values
                                out = out.append(df_nuc)
        return out

    def selectiondicts_run(self, experiment_tag, run):
        centrosome_inclusion_dict = dict()
        centrosome_exclusion_dict = dict()
        centrosome_equivalence_dict = dict()
        joined_tracks = dict()
        with h5py.File(self.filename, 'r') as f:
            select_addr = '%s/%s/selection' % (experiment_tag, run)
            sel = f[select_addr]
            nuclei_list = [int(n[1:]) for n in sel if (n != 'pandas_dataframe' and n != 'pandas_masks')]
            centrosomes_list = [int(c[1:]) for c in f['%s/%s/measurements/centrosomes' % (experiment_tag, run)].keys()]

            for nuclei in sel:
                if nuclei == 'pandas_dataframe' or nuclei == 'pandas_masks': continue
                nid = int(nuclei[1:])
                centrosomes_of_nuclei = [c for c in centrosomes_list if int(c / 100.0) == nid]
                C = set(centrosomes_of_nuclei)
                A = set([int(c[1:]) for c in sel[nuclei + '/A']])
                B = set([int(c[1:]) for c in sel[nuclei + '/B']])
                inclusion_list = sorted(list(A | B))
                exclusion_list = sorted(list(C - A - B))
                centrosome_inclusion_dict[nid] = inclusion_list
                centrosome_exclusion_dict[nid] = exclusion_list
                centrosome_equivalence_dict[nid] = list()
                if len(A) > 1:
                    centrosome_equivalence_dict[nid].append(sorted(list(A)))
                if len(B) > 1:
                    centrosome_equivalence_dict[nid].append(sorted(list(B)))
        return nuclei_list, centrosome_inclusion_dict, centrosome_exclusion_dict, centrosome_equivalence_dict, joined_tracks

    def process_selection_for_run(self, experiment_tag, run):
        nuclei_list, centrosome_inclusion_dict, centrosome_exclusion_dict, centrosome_equivalence_dict, joined_tracks = \
            self.selectiondicts_run(experiment_tag, run)

        merge_key = '%s/%s/measurements/pandas_dataframe' % (experiment_tag, run)
        nuclei_key = '%s/%s/measurements/nuclei_dataframe' % (experiment_tag, run)
        pdhdf_measured = pd.read_hdf(self.filename, key=merge_key, mode='r')
        pdhdf_nuclei = pd.read_hdf(self.filename, key=nuclei_key, mode='r')

        # update centrosome nuclei from selection
        with h5py.File(self.filename, 'r') as f:
            for nuclei_str in f['%s/%s/selection' % (experiment_tag, run)]:
                if nuclei_str == 'pandas_dataframe' or nuclei_str == 'pandas_masks': continue
                nuclei_id = int(nuclei_str[1:])
                centrosomes_of_nuclei_a = f['%s/%s/selection/%s/A' % (experiment_tag, run, nuclei_str)].keys()
                centrosomes_of_nuclei_b = f['%s/%s/selection/%s/B' % (experiment_tag, run, nuclei_str)].keys()
                for centr_str in centrosomes_of_nuclei_a + centrosomes_of_nuclei_b:
                    centr_id = int(centr_str[1:])
                    pdhdf_measured.loc[pdhdf_measured['Centrosome'] == centr_id, 'Nuclei'] = nuclei_id

        # re-merge with nuclei data
        pdhdf_measured.drop(['NuclX', 'NuclY', 'NuclBound'], axis=1, inplace=True)
        pdhdf_merge = pdhdf_measured.merge(pdhdf_nuclei)

        proc_df, mask_df = dfij.process_dataframe(pdhdf_merge, nuclei_list=nuclei_list,
                                                  centrosome_exclusion_dict=centrosome_exclusion_dict,
                                                  centrosome_inclusion_dict=centrosome_inclusion_dict)
        if proc_df.empty:
            print 'dataframe is empty'
        else:
            with h5py.File(self.filename, 'a') as f:
                fproc = f['%s/%s/processed' % (experiment_tag, run)]
                if 'pandas_dataframe' in fproc: del fproc['pandas_dataframe']
                if 'pandas_masks' in fproc: del fproc['pandas_masks']

            with h5py.File(self.filename, 'r') as f:
                fsel = f['%s/%s/selection' % (experiment_tag, run)]
                for nuclei_str in fsel:
                    if nuclei_str == 'pandas_dataframe' or nuclei_str == 'pandas_masks': continue
                    centrosomes_of_nuclei_a = fsel['%s/A' % nuclei_str].keys()
                    centrosomes_of_nuclei_b = fsel['%s/B' % nuclei_str].keys()
                    for centr_str in centrosomes_of_nuclei_a:
                        centr_id = int(centr_str[1:])
                        proc_df.loc[proc_df['Centrosome'] == centr_id, 'CentrLabel'] = 'A'
                        mask_df.loc[mask_df['Centrosome'] == centr_id, 'CentrLabel'] = 'A'
                    for centr_str in centrosomes_of_nuclei_b:
                        centr_id = int(centr_str[1:])
                        proc_df.loc[proc_df['Centrosome'] == centr_id, 'CentrLabel'] = 'B'
                        mask_df.loc[mask_df['Centrosome'] == centr_id, 'CentrLabel'] = 'B'
            proc_df.drop(proc_df[proc_df['CentrLabel'] == None].index, inplace=True)
            mask_df = mask_df[mask_df['Nuclei'] > 0]

            proc_df.to_hdf(self.filename, key='%s/%s/processed/pandas_dataframe' % (experiment_tag, run), mode='r+')
            mask_df.to_hdf(self.filename, key='%s/%s/processed/pandas_masks' % (experiment_tag, run), mode='r+')

    def associate_centrosome_with_nuclei(self, centr_id, nuc_id, experiment_tag, run, centrosome_group=1):
        with h5py.File(self.filename, 'a') as f:
            # link centrosome to current nuclei selection
            source_cpos_addr = '%s/%s/measurements/centrosomes/C%03d/pos' % (experiment_tag, run, centr_id)
            source_npos_addr = '%s/%s/measurements/nuclei/N%02d/pos' % (experiment_tag, run, nuc_id)
            nxcpos = f[source_cpos_addr]

            target_addr = '%s/%s/selection/N%02d' % (experiment_tag, run, nuc_id)
            if source_npos_addr in f:
                if target_addr not in f:
                    nxnuc_ = f.create_group(target_addr)
                    nxnuc_.create_group('A')
                    nxnuc_.create_group('B')
                    nxnpos = f[source_npos_addr]
                    nxnuc_['pos'] = nxnpos

                cstr = 'A' if centrosome_group == 0 else 'B'
                nxnuc_ = f['%s/%s' % (target_addr, cstr)]
                nxnuc_['C%03d' % centr_id] = nxcpos

    def delete_association(self, of_centrosome, with_nuclei, experiment_tag, run):
        with h5py.File(self.filename, 'a') as f:
            centosomesA = f['%s/%s/selection/N%02d/A' % (experiment_tag, run, with_nuclei)]
            centosomesB = f['%s/%s/selection/N%02d/B' % (experiment_tag, run, with_nuclei)]
            if of_centrosome in centosomesA:
                del centosomesA[of_centrosome]
            if of_centrosome in centosomesB:
                del centosomesB[of_centrosome]

    def move_association(self, of_centrosome, from_nuclei, toNuclei, centrosome_group, experiment_tag, run):
        self.delete_association(of_centrosome, from_nuclei, experiment_tag, run)
        self.associate_centrosome_with_nuclei(of_centrosome, toNuclei, experiment_tag, run)

    def is_centrosome_associated(self, centrosome, experiment_tag, run):
        with h5py.File(self.filename, 'r') as f:
            nuclei_list = f['%s/%s/measurements/nuclei' % (experiment_tag, run)]
            sel = f['%s/%s/selection' % (experiment_tag, run)]
            for nuclei in nuclei_list:
                if nuclei in sel:
                    nuc = sel[nuclei]
                    if centrosome in nuc['A'] or centrosome in nuc['B']:
                        return True
        return False

    def add_processed(self, experiment_tag, run):
        pass


def process_dir(path, hdf5f):
    condition = os.path.abspath(args.input).split('/')[-1]

    for root, directories, filenames in os.walk(os.path.join(path, 'input')):
        for filename in filenames:
            ext = filename.split('.')[-1]
            if ext == 'tif':
                joinf = os.path.join(root, filename)
                try:
                    print '\r\n--------------------------------------------------------------'
                    groups = re.search('^(.+)-(.+).tif$', filename).groups()
                    run_id = groups[1]
                    run_str = 'run_%s' % run_id
                    print 'adding raw file: %s' % joinf
                    hdf5.add_experiment(condition, run_str)
                    hdf5f.add_tiff_sequence(joinf, condition, run_str)

                    centdata = os.path.join(path, 'data', '%s-%s-table.csv' % (condition, run_id))
                    nucldata = os.path.join(path, 'data', '%s-%s-nuclei.csv' % (condition, run_id))
                    print 'adding data file: %s' % centdata
                    hdf5f.add_measurements(centdata, condition, run_str)
                except:
                    print 'error processing %s' % joinf


if __name__ == '__main__':
    # process input arguments
    parser = argparse.ArgumentParser(
        description='Creates an HDF5 file for experiments storage.')
    parser.add_argument('input', metavar='I', type=str, help='input directory where the files are')
    args = parser.parse_args()

    # Create hdf5 file if it doesn't exist
    hdf5 = LabHDF5NeXusFile(filename='/Users/Fabio/centrosomes.nexus.hdf5', fileflag='a')
    try:
        process_dir(args.input, hdf5)
    finally:
        print '\r\n\r\n--------------------------------------------------------------'
        print 'shrinking file size...'
        call('h5repack /Users/Fabio/centrosomes.nexus.hdf5 /Users/Fabio/repack.hdf5', shell=True)
        os.remove('/Users/Fabio/centrosomes.nexus.hdf5')
        os.rename('/Users/Fabio/repack.hdf5', '/Users/Fabio/centrosomes.nexus.hdf5')

import argparse
import datetime
import logging
import os
import re
import sys
from subprocess import call

import h5py
import numpy as np
import pandas as pd
import tifffile as tf

import stats
from imagej_pandas import ImagejPandas


class LabHDF5NeXusFile():
    def __init__(self, filename='fabio_data_hochegger_lab.nexus.hdf5', imagesfile=None, fileflag='r'):
        self.filename = filename
        self.imagefile = imagesfile

        # open the HDF5 NeXus file
        if fileflag == 'w':
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

    def add_experiment(self, group, experiment_tag, timestamp=None):
        gr = '%s/%s' % (group, experiment_tag)
        timestamp = timestamp if timestamp is not None else datetime.datetime.now().isoformat()

        with h5py.File(self.filename, 'a') as f:
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

        with h5py.File(self.imagefile, 'a') as i:
            rawgr = '%s/%s/raw' % (group, experiment_tag)
            # create the NXentry experiment group
            nxentry = i.require_group(rawgr)
            nxentry.attrs['NX_class'] = 'NXentry'
            nxentry.attrs['datetime'] = timestamp
            i[rawgr].attrs['NX_class'] = 'NXgroup'

    def add_tiff_sequence(self, tiffpath, experiment_tag, run):
        _grp = '%s/%s/raw' % (experiment_tag, run)
        # open the HDF5 NeXus file
        if self.imagefile is None:
            f = h5py.File(self.filename, 'a')
        else:
            f = h5py.File(self.imagefile, 'a')

        # update the NXentry group
        nxdata = f[_grp]
        nxdata.attrs['NX_class'] = 'NXdata'
        nxdata.attrs['signal'] = 'Y'  # Y axis of default plot
        nxdata.attrs['axes'] = 'X'  # X axis of default plot
        nxdata.attrs['units'] = 'um'  # default units

        nframes = len(nxdata.items())
        if nframes == 0:
            with tf.TiffFile(tiffpath, fastij=True) as tif:
                if tif.is_imagej is not None:
                    sizeT, channels = tif.pages[0].imagej_tags.frames, tif.pages[0].imagej_tags.channels
                    sizeZ, sizeX, sizeY = 1, tif.pages[0].image_width, tif.pages[0].image_length
                    logging.info('N of frames=%d channels=%d, sizeZ=%d, sizeX=%d, sizeY=%d' % \
                                 (sizeT, channels, sizeZ, sizeX, sizeY))

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
                        p1a = tif.pages[0].asarray().reshape([sizeT, channels, sizeX, sizeY])
                        for i in range(sizeT):
                            # create a NXentry frame group
                            nxframe = nxdata.create_group('%03d' % i)
                            nxframe.attrs['units'] = 'um'
                            nxframe.attrs['resolution'] = res
                            nxframe.attrs['long_name'] = 'image um (micrometers)'  # suggested X axis plot label

                            # save XY data
                            ch1 = nxframe.create_dataset('channel-1', data=p1a[i][0], dtype=np.uint16)
                            ch2 = nxframe.create_dataset('channel-2', data=p1a[i][1], dtype=np.uint16)
                            ch3 = nxframe.create_dataset('channel-3', data=p1a[i][2], dtype=np.uint16)
                            for ch in [ch1, ch2, ch3]:
                                ch.attrs['CLASS'] = np.string_('IMAGE')
                                ch.attrs['IMAGE_SUBCLASS'] = np.string_('IMAGE_GRAYSCALE')
                                ch.attrs['IMAGE_VERSION'] = np.string_('1.2')

        f.close()

        if self.imagefile is not None and nframes == 0:
            with h5py.File(self.filename, 'a') as f:
                del f[_grp]
                f[_grp] = h5py.ExternalLink(self.imagefile, _grp)

    def add_measurements(self, csvpath, experiment_tag, run):
        dfc = ImagejPandas(csvpath)
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
            for _, filt_centr_df in dfct.groupby('Centrosome'):
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
        df_out = pd.DataFrame()
        with h5py.File(self.filename, 'r') as f:
            for experiment_tag in f:
                for run in f['%s' % experiment_tag]:
                    if 'pandas_dataframe' in f['%s/%s/processed' % (experiment_tag, run)]:
                        selection_key = '%s/%s/processed/pandas_dataframe' % (experiment_tag, run)
                        df = pd.read_hdf(self.filename, key=selection_key, mode='r')
                        df['condition'] = experiment_tag
                        df['run'] = run
                        df_out = df_out.append(df)
        return stats.reconstruct_time(df_out)

    @property
    def mask(self):
        df_msk = pd.DataFrame()
        with h5py.File(self.filename, 'r') as f:
            for experiment_tag in f:
                for run in f['%s' % experiment_tag]:
                    if 'pandas_masks' in f['%s/%s/processed' % (experiment_tag, run)]:
                        selection_key = '%s/%s/processed/pandas_masks' % (experiment_tag, run)
                        msk = pd.read_hdf(self.filename, key=selection_key, mode='r')
                        msk['condition'] = experiment_tag
                        msk['run'] = run
                        df_msk = df_msk.append(msk)
        return stats.reconstruct_time(df_msk)

    def selectiondicts_run(self, experiment_tag, run):
        centrosome_inclusion_dict = dict()
        centrosome_exclusion_dict = dict()
        centrosome_equivalence_dict = dict()
        nuclei_list = list()
        with h5py.File(self.filename, 'r') as f:
            if 'centrosomes' not in f['%s/%s/measurements' % (experiment_tag, run)]:
                raise KeyError('No data for selected condition-run.')

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
        return nuclei_list, centrosome_inclusion_dict, centrosome_exclusion_dict, centrosome_equivalence_dict

    def process_selection_for_run(self, experiment_tag, run):
        nuclei_list, centrosome_inclusion_dict, centrosome_exclusion_dict, centrosome_equivalence_dict = \
            self.selectiondicts_run(experiment_tag, run)

        # don't keep processing if there's nothing to do
        if len(nuclei_list) == 0: return

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

        # merge with cell boundary data
        with h5py.File(self.filename, 'r') as f:
            if 'boundary' in f['%s/%s/processed' % (experiment_tag, run)]:
                df_cell = pd.read_hdf(self.filename, key='%s/%s/processed/boundary' % (experiment_tag, run))
                df_cell = df_cell.loc[~df_cell['CellBound'].isnull(), :]

                if 'CellBound' in pdhdf_merge.columns:
                    logging.info('clearing CellBound')
                    pdhdf_merge.drop(['CellX', 'CellY', 'CellBound'], axis=1, inplace=True)
                pdhdf_merge = pdhdf_merge.merge(df_cell)

        pdhdf_merge['condition'] = experiment_tag
        pdhdf_merge['run'] = run

        try:
            proc_df, mask_df = ImagejPandas.process_dataframe(pdhdf_merge, nuclei_list=nuclei_list,
                                                              centrosome_inclusion_dict=centrosome_inclusion_dict)

            with h5py.File(self.filename, 'r+') as f:
                fproc = f['%s/%s/processed' % (experiment_tag, run)]
                if 'pandas_dataframe' in fproc: del fproc['pandas_dataframe']
                if 'pandas_masks' in fproc: del fproc['pandas_masks']

            dfn_join = mskn_join = pd.DataFrame()
            for nuc_id, df_nuc in proc_df.groupby('Nuclei'):
                logging.debug(
                    'processing selection for condition %s run %s nuclei N%02d' % (experiment_tag, run, nuc_id))
                msk_nuc = mask_df[mask_df['Nuclei'] == nuc_id]
                # ----------------------------------------------------------
                # join tracks based on distance of time of contact
                # ----------------------------------------------------------
                # get time of contact
                dthrsh = ImagejPandas.DIST_THRESHOLD * 1.5
                time_ofc, frame_ofc, dist_ofc = ImagejPandas.get_contact_time(df_nuc, dthrsh)
                if time_ofc is not None:
                    logging.debug('time of contact=%0.1f frame=%d dist=%0.1f' % (time_ofc, frame_ofc, dist_ofc))
                    frames_near_toc = [frame_ofc, frame_ofc - 1, frame_ofc + 1]
                    centrosomes_ofc = df_nuc[df_nuc['Frame'].isin(frames_near_toc)]['Centrosome'].unique()
                    if len(centrosomes_ofc) == 2:
                        logging.debug('--> joining centrosomes %d and %d' % (centrosomes_ofc[0], centrosomes_ofc[1]))
                        dfj, mskj = ImagejPandas.join_tracks(df_nuc, centrosomes_ofc[0], centrosomes_ofc[1])
                        dfn_join = dfn_join.append(dfj)

                        msk_and = msk_nuc.set_index(ImagejPandas.MASK_INDEX) & mskj.set_index(ImagejPandas.MASK_INDEX)
                        mskn_join = mskn_join.append(msk_and.reset_index())
                else:
                    dfn_join = dfn_join.append(df_nuc)
                    mskn_join = mskn_join.append(msk_nuc)

            with h5py.File(self.filename, 'r') as f:
                fsel = f['%s/%s/selection' % (experiment_tag, run)]
                for nuclei_str in fsel:
                    if nuclei_str == 'pandas_dataframe' or nuclei_str == 'pandas_masks': continue
                    centrosomes_of_nuclei_a = fsel['%s/A' % nuclei_str].keys()
                    centrosomes_of_nuclei_b = fsel['%s/B' % nuclei_str].keys()
                    for centr_str in centrosomes_of_nuclei_a:
                        centr_id = int(centr_str[1:])
                        dfn_join.loc[dfn_join['Centrosome'] == centr_id, 'CentrLabel'] = 'A'
                        mskn_join.loc[mskn_join['Centrosome'] == centr_id, 'CentrLabel'] = 'A'
                    for centr_str in centrosomes_of_nuclei_b:
                        centr_id = int(centr_str[1:])
                        dfn_join.loc[dfn_join['Centrosome'] == centr_id, 'CentrLabel'] = 'B'
                        mskn_join.loc[mskn_join['Centrosome'] == centr_id, 'CentrLabel'] = 'B'

            proc_df = ImagejPandas.dist_vel_acc_centrosomes(dfn_join[~dfn_join['CentrLabel'].isnull()])

            maxframe1 = proc_df.loc[proc_df['CentrLabel'] == 'A', 'Frame'].max()
            maxframedc = proc_df['Frame'].max()
            minframe1 = min(maxframe1, maxframedc)

            idx1 = (proc_df['CentrLabel'] == 'A') & (proc_df['Frame'] <= minframe1)
            proc_df.loc[idx1, 'SpeedCentr'] *= -1
            proc_df.loc[idx1, 'AccCentr'] *= -1

            mask_df = mskn_join[mskn_join['Nuclei'] > 0]

            proc_df.to_hdf(self.filename, key='%s/%s/processed/pandas_dataframe' % (experiment_tag, run), mode='r+')
            mask_df.to_hdf(self.filename, key='%s/%s/processed/pandas_masks' % (experiment_tag, run), mode='r+')
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            logging.warning('Problem processing %s-%s in line %d of hdf5_nexus.py:\r\n%s' % (
                experiment_tag, run, exc_tb.tb_lineno, e))

    def associate_centrosome_with_nuclei(self, centr_id, nuc_id, experiment_tag, run, centrosome_group=0):
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

    def clear_associations(self, experiment_tag, run):
        with h5py.File(self.filename, 'a') as f:
            fsel = f['%s/%s/selection' % (experiment_tag, run)]
            for o in fsel:
                del fsel[o]

    def delete_association(self, of_centrosome, with_nuclei, experiment_tag, run):
        with h5py.File(self.filename, 'a') as f:
            centosomesA = f['%s/%s/selection/N%02d/A' % (experiment_tag, run, with_nuclei)]
            centosomesB = f['%s/%s/selection/N%02d/B' % (experiment_tag, run, with_nuclei)]
            if 'C%03d' % of_centrosome in centosomesA:
                del centosomesA['C%03d' % of_centrosome]
            if 'C%03d' % of_centrosome in centosomesB:
                del centosomesB['C%03d' % of_centrosome]

            with h5py.File(self.filename, 'a') as f:
                fproc = f['%s/%s/processed' % (experiment_tag, run)]
                df_key = '%s/%s/processed/pandas_dataframe' % (experiment_tag, run)
                msk_key = '%s/%s/processed/pandas_masks' % (experiment_tag, run)
                _newdf = _newmsk = None
                if 'pandas_dataframe' in fproc:
                    _newdf = pd.read_hdf(self.filename, key=df_key, mode='r')
                    del fproc['pandas_dataframe']
                if _newdf is not None:
                    _idx = (_newdf['Centrosome'] == of_centrosome) & \
                           (_newdf['Nuclei'] == with_nuclei)
                    _newdf[~_idx].to_hdf(self.filename, key=df_key, mode='r+')
                if 'pandas_masks' in fproc:
                    _newmsk = pd.read_hdf(self.filename, key=msk_key, mode='r')
                    del fproc['pandas_masks']
                if _newmsk is not None:
                    _idx = (_newmsk['Centrosome'] == of_centrosome) & \
                           (_newmsk['Nuclei'] == with_nuclei)
                    _newmsk[~_idx].to_hdf(self.filename, key=msk_key, mode='r+')

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


def move_images(filefrom, fileto):
    src = h5py.File(filefrom, 'a')
    dst = h5py.File(fileto, 'a')
    for cond in src.iterkeys():
        for run in src[cond].iterkeys():
            _grp = '%s/%s/raw' % (cond, run)
            if isinstance(src[_grp], h5py.SoftLink):
                logging.debug(_grp, 'group already a soft link')
            if isinstance(src[_grp], h5py.HardLink):
                logging.debug(_grp, 'group already a hard link')
            if isinstance(src[_grp], h5py.Group):
                # Get the name of the parent for the group we want to copy
                group_path = src[_grp].parent.name
                # Check that this group exists in the destination file; if it doesn't, create it
                # This will create the parents too, if they don't exist
                group_id = dst.require_group(group_path)
                # Copy src -> dst
                if _grp not in dst:
                    logging.debug(_grp, 'copying raw images')
                    src.copy(_grp, group_id, name='raw')
                del src[_grp]
                src[_grp] = h5py.ExternalLink(fileto, _grp)
    src.close()
    dst.close()


def process_dir(path, hdf5f):
    condition = os.path.abspath(args.input).split('/')[-1]

    for root, directories, filenames in os.walk(os.path.join(path, 'input')):
        for filename in filenames:
            ext = filename.split('.')[-1]
            if ext == 'tif':
                joinf = os.path.join(root, filename)
                logging.info('--------------------------------------------------------------')
                groups = re.search('^(.+)-(.+).tif$', filename).groups()
                run_id = groups[1]
                run_str = 'run_%s' % run_id
                centdata = os.path.join(path, 'data', 'run-%s-table.csv' % run_id)
                nucldata = os.path.join(path, 'data', 'run-%s-nuclei.csv' % run_id)

                if os.path.isfile(centdata) and os.path.isfile(nucldata) and os.path.isfile(joinf):
                    logging.info('adding raw file: %s' % joinf)
                    hdf5.add_experiment(condition, run_str)
                    logging.info('adding tiff:', joinf)
                    hdf5f.add_tiff_sequence(joinf, condition, run_str)
                    logging.info('adding data file: %s' % centdata)
                    hdf5f.add_measurements(centdata, condition, run_str)


if __name__ == '__main__':
    # process input arguments
    parser = argparse.ArgumentParser(
        description='Creates an HDF5 file for experiments storage.')
    parser.add_argument('input', metavar='I', type=str, help='input directory where the files are')
    args = parser.parse_args()

    # Create hdf5 file if it doesn't exist
    hdf5 = LabHDF5NeXusFile(filename='/Users/Fabio/centrosomes.nexus.hdf5',
                            imagesfile='/Users/Fabio/centrosomes-images.nexus.hdf5', fileflag='a')
    try:
        process_dir(args.input, hdf5)
        move_images('/Users/Fabio/centrosomes.nexus.hdf5', '/Users/Fabio/centrosomes-images.nexus.hdf5')

        logging.info('--------------------------------------------------------------')
        logging.info('shrinking file size...')
        call('h5repack /Users/Fabio/centrosomes.nexus.hdf5 /Users/Fabio/repack.hdf5', shell=True)
        os.remove('/Users/Fabio/centrosomes.nexus.hdf5')
        os.rename('/Users/Fabio/repack.hdf5', '/Users/Fabio/centrosomes.nexus.hdf5')
        call('h5repack /Users/Fabio/centrosomes-images.nexus.hdf5 /Users/Fabio/repack.hdf5', shell=True)
        os.remove('/Users/Fabio/centrosomes-images.nexus.hdf5')
        os.rename('/Users/Fabio/repack.hdf5', '/Users/Fabio/centrosomes-images.nexus.hdf5')
    finally:
        logging.info('finished.')

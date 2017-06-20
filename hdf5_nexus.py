import argparse
import datetime
import os
import re

import h5py
import numpy as np
import tifffile as tf

import dataframe_from_imagej as ijdf


class LabHDF5NeXusFile():
    def __init__(self, filename=None, rewrite=False):
        if filename is None:
            self.filename = "fabio_data_hochegger_lab.nexus.hdf5"
        else:
            self.filename = filename

        timestamp = datetime.datetime.now().isoformat()

        # create the HDF5 NeXus file
        openflag = 'a' if (os.path.isfile(self.filename) or rewrite) else 'w'
        with h5py.File(self.filename, openflag) as f:
            # point to the default data to be plotted
            f.attrs['default'] = 'entry'
            # give the HDF5 root some more attributes
            f.attrs['file_name'] = self.filename
            f.attrs['file_time'] = timestamp
            f.attrs['creator'] = 'fabio@hochegger.lab'
            f.attrs['NeXus_version'] = '4.3.0'
            f.attrs['HDF5_Version'] = h5py.version.hdf5_version
            f.attrs['h5py_version'] = h5py.version.version

    def addExperiment(self, group, experimentTag, timestamp=None, tags=None):
        with h5py.File(self.filename, "a") as f:
            timestamp = timestamp if timestamp is not None else datetime.datetime.now().isoformat()

            gr = '%s/%s' % (group, experimentTag)
            if gr in f:
                del f[gr]

            # create the NXentry experiment group
            nxentry = f.create_group(gr)
            nxentry.attrs['NX_class'] = 'NXentry'
            nxentry.attrs['datetime'] = timestamp
            nxentry.attrs['default'] = 'processed'

            nxentry_raw = nxentry.create_group('raw')
            nxentry_meas = nxentry.create_group('measurements')
            nxentry_sel = nxentry.create_group('selection')
            nxentry_proc = nxentry.create_group('processed')

    def addTiffSequence(self, tiffpath, experimentTag, run):
        print "adding tiff:", tiffpath

        # open the HDF5 NeXus file
        f = h5py.File(self.filename, "a")

        # create the NXentry group
        nxdata = f['%s/%s/raw' % (experimentTag, run)]
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

    def addMeasurements(self, csvpath, experimentTag, run):
        with h5py.File(self.filename, "a") as f:
            nxmeas = f['%s/%s/measurements' % (experimentTag, run)]

            dfc = ijdf.DataFrameFromImagej(csvpath)
            dfnt = dfc.df_nuclei_csv.set_index('Frame').sort_index()
            for (nucID), filt_nuc_df in dfnt.groupby('Nuclei'):
                nxnucl = nxmeas.create_group('nuclei/N%02d' % nucID)
                nxnucl.attrs['NX_class'] = 'NXdata'
                nxnucl.attrs['Nuclei'] = 'N%02d' % nucID
                nx = filt_nuc_df['NuclX']
                ny = filt_nuc_df['NuclY']
                fn = filt_nuc_df.reset_index()
                p = nxnucl.create_dataset('pos', data=fn[['Frame', 'NuclX', 'NuclY']], dtype=nx.dtype)
                fr = nxnucl.create_dataset('frame', data=fn['Frame'], dtype=nx.dtype)
                sx = nxnucl.create_dataset('sample_x', data=nx, dtype=nx.dtype)
                sy = nxnucl.create_dataset('sample_y', data=ny, dtype=ny.dtype)

            dfct = dfc.df_csv.set_index('Frame').sort_index()

            for (centrID), filt_centr_df in dfct.groupby('Centrosome'):
                nucID = filt_centr_df['Nuclei'].unique()[0]
                nxcid = nxmeas.create_group('centrosomes/C%03d' % centrID)
                nxcid.attrs['NX_class'] = 'NXdata'
                cx = filt_centr_df['CentX']
                cy = filt_centr_df['CentY']
                cn = filt_centr_df.reset_index()
                p = nxcid.create_dataset('pos', data=cn[['Frame', 'CentX', 'CentY']], dtype=cx.dtype)
                p = nxcid.create_dataset('frame', data=cn['Frame'], dtype=cx.dtype)
                sx = nxcid.create_dataset('sample_x', data=cx, dtype=cx.dtype)
                sy = nxcid.create_dataset('sample_y', data=cy, dtype=cy.dtype)

            visitedCentrosomes = []
            for (centrID), filt_centr_df in dfct.groupby('Centrosome'):
                centrosomesOfNuclei = dfct[dfct['Nuclei'] == nucID].groupby('Centrosome')
                if len(centrosomesOfNuclei.groups) >= 2:
                    for i, ((centrID), filt_centr_df) in enumerate(centrosomesOfNuclei):
                        if centrID not in visitedCentrosomes:
                            visitedCentrosomes.append(centrID)
                            nucID = filt_centr_df['Nuclei'].unique()[0]
                            self.associateCentrosomeWithNuclei(centrID, nucID, experimentTag, run, i % 2)

    def associateCentrosomeWithNuclei(self, centrID, nucID, experimentTag, run, centrosomeGroup=1):
        with h5py.File(self.filename, "a") as f:
            # link centrosome to current nuclei selection
            source_cpos_addr = '%s/%s/measurements/centrosomes/C%03d/pos' % (experimentTag, run, centrID)
            source_npos_addr = '%s/%s/measurements/nuclei/N%02d/pos' % (experimentTag, run, nucID)
            nxcpos = f[source_cpos_addr]

            target_addr = '%s/%s/selection/N%02d' % (experimentTag, run, nucID)
            if target_addr not in f:
                nxnuc_ = f.create_group(target_addr)
                nxnuc_.create_group('A')
                nxnuc_.create_group('B')
                nxnpos = f[source_npos_addr]
                nxnuc_['pos'] = nxnpos

            cstr = 'A' if centrosomeGroup == 0 else 'B'
            nxnuc_ = f['%s/%s' % (target_addr, cstr)]
            nxnuc_['C%03d' % centrID] = nxcpos

    def deleteAssociation(self, ofCentrosome, withNuclei, experimentTag, run):
        with h5py.File(self.filename, "a") as f:
            centosomesA = f['%s/%s/selection/N%02d/A' % (experimentTag, run, withNuclei)]
            centosomesB = f['%s/%s/selection/N%02d/B' % (experimentTag, run, withNuclei)]
            if ofCentrosome in centosomesA:
                del centosomesA[ofCentrosome]
            if ofCentrosome in centosomesB:
                del centosomesB[ofCentrosome]

    def moveAssociation(self, ofCentrosome, fromNuclei, toNuclei, centrosomeGroup, experimentTag, run):
        self.deleteAssociation(ofCentrosome, fromNuclei, experimentTag, run)
        self.associateCentrosomeWithNuclei(ofCentrosome, toNuclei, experimentTag, run)

    def isCentrosomeAssociated(self, centrosome, experimentTag, run):
        with h5py.File(self.filename, "r") as f:
            nuclei_list = f['%s/%s/measurements/nuclei' % (experimentTag, run)]
            sel = f['%s/%s/selection' % (experimentTag, run)]
            for nuclei in nuclei_list:
                if nuclei in sel:
                    nuc = sel[nuclei]
                    if centrosome in nuc['A'] or centrosome in nuc['B']:
                        return True
        return False

    def addProcessed(self, experimentTag, run):
        pass


def process_dir(path, hdf5f):
    condition = os.path.abspath(args.input).split('/')[-1]

    for root, directories, filenames in os.walk(os.path.join(path, 'input')):
        for filename in filenames:
            ext = filename.split('.')[-1]
            if ext == 'tif':
                print '\r\n--------------------------------------------------------------'
                joinf = os.path.join(root, filename)
                groups = re.search('^(.+)-(.+).tif$', filename).groups()
                run_id = groups[1]
                run_str = 'run_%s' % run_id
                print 'adding raw file: %s' % joinf
                hdf5.addExperiment(condition, run_str)
                hdf5f.addTiffSequence(joinf, condition, run_str)

                centdata = os.path.join(path, 'data', '%s-%s-table.csv' % (condition, run_id))
                nucldata = os.path.join(path, 'data', '%s-%s-nuclei.csv' % (condition, run_id))
                print 'adding data file: %s' % centdata
                hdf5f.addMeasurements(centdata, condition, run_str)


if __name__ == '__main__':
    # process input arguments
    parser = argparse.ArgumentParser(
        description='Creates an HDF5 file for experiments storage.')
    parser.add_argument('input', metavar='I', type=str, help='input directory where the files are')
    args = parser.parse_args()

    # Create hdf5 file if it doesn't exist
    hdf5 = LabHDF5NeXusFile(filename='/Users/Fabio/centrosomes.nexus.hdf5', rewrite=True)

    process_dir(args.input, hdf5)

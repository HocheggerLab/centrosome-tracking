import pandas as pd
import numpy as np
import h5py
import datetime
import tifffile as tf
import os
import re
import argparse
import dataframe_from_imagej as ijdf


class LabHDF5NeXusFile():
    def __init__(self, filename=None):
        if filename is None:
            self.filename = "fabio_data_hochegger_lab.nexus.hdf5"
        else:
            self.filename = filename

        timestamp = datetime.datetime.now().isoformat()

        # create the HDF5 NeXus file
        with h5py.File(self.filename, "w") as f:
            # point to the default data to be plotted
            f.attrs['default'] = 'entry'
            # give the HDF5 root some more attributes
            f.attrs['file_name'] = self.filename
            f.attrs['file_time'] = timestamp
            f.attrs['creator'] = 'fabio@hochegger.lab'
            f.attrs['NeXus_version'] = '4.3.0'
            f.attrs['HDF5_Version'] = h5py.version.hdf5_version
            f.attrs['h5py_version'] = h5py.version.version

        print "wrote file:", self.filename

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
                    res = float(xr[0]) / float(xr[1])

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

                            # rgb = nxframe.create_dataset('rgb',
                            #                              data=np.reshape(p1a[channels * i:channels * i + 3],
                            #                                              (sizeX, sizeY, channels)).astype(np.uint8),
                            #                              dtype=np.uint8)
                            # rgb.attrs['CLASS'] = np.string_('IMAGE')
                            # rgb.attrs['IMAGE_SUBCLASS'] = np.string_('IMAGE_TRUECOLOR')
                            # rgb.attrs['IMAGE_VERSION'] = np.string_('1.2')
                            # rgb.attrs['INTERLACE_MODE'] = np.string_('INTERLACE_PIXEL')

        f.close()

    def addMeasurements(self, csvpath, experimentTag, run):
        dfij = ijdf.DataFrameFromImagej(csvpath)
        dft = dfij.df_csv.set_index('Frame').sort_index()

        with h5py.File(self.filename, "a") as f:
            nxmeas = f['%s/%s/measurements' % (experimentTag, run)]
            for (nucID), filt_nuc_df in dft.groupby('Nuclei'):
                nxnucl = nxmeas.create_group('nuclei/N%02d' % nucID)
                # nxcentr = nxnucl.create_group('centrosomes')
                for (centrID), filt_centr_df in filt_nuc_df.groupby('Centrosome'):
                    nxcid = nxnucl.create_group('C%03d' % centrID)
                    nxcid.attrs['NX_class'] = 'NXdata'
                    cx = filt_centr_df['CentX']
                    cy = filt_centr_df['CentY']
                    sx = nxcid.create_dataset('sample_x', data=cx, dtype=cx.dtype)
                    sy = nxcid.create_dataset('sample_y', data=cy, dtype=cy.dtype)

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
    # parser.add_argument('take', metavar='T', type=str, help='group number for images taken in the same session')
    # parser.add_argument('output', metavar='O', type=str, help='output file')
    args = parser.parse_args()

    # Create hdf5 file if it doesn't exist
    hdf5 = LabHDF5NeXusFile(filename='centrosomes.nexus.hdf5')

    process_dir(args.input, hdf5)

#!/opt/local/bin/python
import argparse
import os
import re
from shutil import copyfile

if __name__ == '__main__':
    # process input arguments
    parser = argparse.ArgumentParser(
        description='(Re)tags images from an experiment group.')
    parser.add_argument('input', metavar='I', type=str, help='input directory where the folders of images are')
    parser.add_argument('output', metavar='O', type=str, help='output directory for the tagged images')
    args = parser.parse_args()

    oroot = os.path.join(os.getcwd(), args.output, 'input')
    if not os.path.exists(oroot):
        os.makedirs(oroot)

    for _root, directories, _filenames in os.walk(args.input):
        if directories:
            dir_enum = list(enumerate(directories))

        for fname in _filenames:
            ext = fname.split('.')[-1]
            print 'file %s ' % fname
            if ext == 'tif':
                print 'tif extension'
                dir = os.path.basename(_root)
                joinf = os.path.join(_root, fname)
                groups = re.search('Capture (.+) - Position (.+).Project Maximum Z.tif$', fname).groups()
                capture_id = int(groups[0])
                pos_id = int(groups[1])

                k = [n + 1 for (n, d) in dir_enum if d == dir][0]
                new_name = 'run-%03d.tif' % (k * 100 + pos_id)
                print 'saving from %s to file: %s' % (joinf, new_name)

                imageout_path = os.path.join(oroot, new_name)
                copyfile(joinf, imageout_path)

                print '--------------------------------------------------------------\r\n'

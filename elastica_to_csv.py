import ConfigParser
import argparse
import ast
import os
import re

import pandas as pd

parser = argparse.ArgumentParser(description='Traverses a folder containing SGE result files, constructing a csv file.')
parser.add_argument('directory', metavar='D', type=str, help='initial directory')
parser.add_argument('-o', '--output', dest='out', action='store', default='elastica.csv', help='output csv file.')
args = parser.parse_args()

with open(args.out, 'w') as of:
    # write csv header
    of.write('measure,trkid,seed,L,a1,a2,E,F,gamma,x0,y0,theta,objfn\n')

for root, directories, filenames in os.walk(args.directory):
    for file in filenames:
        joinf = os.path.abspath(os.path.join(root, file))

        with open(joinf, 'r') as jobfile:
            print '\r\n--------------------------------------------------------------'
            print 'processing file: %s' % joinf
            jobtxt = jobfile.read()

            fname = os.path.abspath(os.path.join(root, '..', 'elastica.cfg.txt'))
            if os.path.isfile(fname):
                try:
                    jobid = re.search('^elastica_hpc.job.o[0-9]*.([0-9]*)$', file).group(1)
                    seed = re.search('INFO:root:time seed: ([0-9]*)', jobtxt).group(1)
                    objf = re.search('INFO:root:objective function final: (.*)', jobtxt).group(1)
                    params = re.search('INFO:root:x0=(.+?) ', jobtxt).group(1)
                    L, a1, a2, E, F, gamma, x0, y0, theta = ast.literal_eval(params)

                    with open(fname, 'r') as configfile:
                        config = ConfigParser.ConfigParser()
                        config.readfp(configfile)

                    section = 'measure-%09d' % (int(jobid) - 1)
                    trkid = config.get(section, 'id')

                    # write pandas
                    df = pd.DataFrame(data=[[jobid, trkid, seed, L, a1, a2, E, F, gamma, x0, y0, theta, objf]],
                                      columns=['measure', 'trkid', 'seed', 'L', 'a1', 'a2', 'E', 'F', 'gamma', 'x0',
                                               'y0', 'theta', 'objfn'])
                    with open(args.out, 'a') as f:
                        df.to_csv(f, header=False, index=False)
                except AttributeError:
                    print 'no enough data in file.'

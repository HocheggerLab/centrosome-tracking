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
    # write pandas
    # df = pd.DataFrame(data=[[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], columns=['id', 'seed', 'L', 'a1', 'a2', 'E', 'F', 'gamma', 'x0', 'y0', 'theta'])
    # df.to_csv(of, index=False)
    of.write('id,seed,L,a1,a2,E,F,gamma,x0,y0,theta,objfn\n')

for root, directories, filenames in os.walk(args.directory):
    for file in filenames:
        joinf = os.path.abspath(os.path.join(root, file))

        with open(joinf, 'r') as jobfile:
            print '\r\n--------------------------------------------------------------'
            print 'processing file: %s' % joinf
            jobtxt = jobfile.read()

            jobid = re.search('^elastica_hpc.job.o[0-9]*.([0-9]*)$', file).group(1)
            seed = re.search('time seed:  ([0-9]*)', jobtxt).group(1)
            objf = re.search('objective function final: (.*)', jobtxt).group(1)
            params = re.search('x0=(.+?) x=', jobtxt).group(1)
            L, a1, a2, E, F, gamma, x0, y0, theta = ast.literal_eval(params)

            # write pandas
            df = pd.DataFrame(data=[[jobid, seed, L, a1, a2, E, F, gamma, x0, y0, theta, objf]],
                              columns=['id', 'seed', 'L', 'a1', 'a2', 'E', 'F', 'gamma', 'x0', 'y0', 'theta', 'objfn'])
            with open(args.out, 'a') as f:
                df.to_csv(f, header=False, index=False)

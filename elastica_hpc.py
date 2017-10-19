import ConfigParser
import argparse
import ast
import json
import os
import os.path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import basinhopping

import elastica as e
import simpleflock

# reopen stdout file descriptor with write mode
# and 0 as the buffer size (unbuffered)
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

"""
module load mps/software python/2.7.12
# virtualenv --python=/mnt/pactsw/python/2.7.12/bin/python ~/py27
source ~/py27/bin/activate
deactivate
"""


def obj_minimize(p, yn, Np=100):
    print 'x=[%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f,\t%03.4f]. ' % tuple(p),
    slen = yn.shape[1]
    ymod = e.model_heavyplanar(p, num_points=Np)
    if ymod is not None and ymod.shape[1] >= slen:
        objfn = (ymod[0:2, 0:slen] - yn[0:2, 0:slen]).flatten()
        objfn = np.sum(objfn ** 2)
        print 'Objective function f(x)=%0.2f' % objfn
        return objfn
    else:
        print 'No solution for objective function.'
        return np.finfo('float64').max


def job(section, fname, Np=100):
    tseed = int(time.time() * 10e5 % 10e8)
    print 'time seed: ', tseed
    np.random.seed(tseed)
    if os.path.isfile(fname):
        with open(fname, 'r') as configfile:
            config = ConfigParser.ConfigParser()
            config.readfp(configfile)

        print 'sections found in file ', config.sections()

        if config.has_section(section):
            mstr = config.get(section, 'measure')
            yn = np.array(json.loads(mstr))
            param_bounds = ((5.0, 15.0), (0.1, 0.6), (0.5, 1.0), (0.01, 3.0), (0.1, 3.0), (-np.pi, np.pi),
                            (0, 512.), (0, 512.), (-np.pi, np.pi))
            x0 = [9.0, 0.2, 0.7, 0.1, 0.4, 0, 0, 0, 0]
            res = basinhopping(obj_minimize, x0, minimizer_kwargs={'bounds': param_bounds, 'args': (yn, Np)})
            print 'x0=[%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f] ' % tuple(res.x),
            print 'objective function final: %0.2f' % obj_minimize(res.x, yn)

            L, a1, a2, E, F, gamma, x0, y0, theta = res.x

            # write pandas
            if os.path.isfile('elastica.pandas.csv'):
                with simpleflock.SimpleFlock('elastica.pandas.csv', timeout=3):
                    df = pd.DataFrame(data=[[section, a1, a2, L, E, F, gamma, x0, y0, theta]],
                                      columns=['id', 'a1', 'a2', 'L', 'E', 'F', 'gamma', 'x0', 'y0', 'theta'])
                    with open('elastica.pandas.csv', 'a') as f:
                        df.to_csv(f, header=False, index=False)


def add_entry(a1, a2, L, E, F, gamma, x0, y0, theta, yn, run=None, comment=None, fname='elastica.cfg.txt'):
    if os.path.isfile(fname):
        with open(fname, 'r') as configfile:
            config = ConfigParser.ConfigParser()
            config.readfp(configfile)

            if not config.has_section('General'):
                config.add_section('General')
            config.set('General', 'Version', 'v0.1')
    else:
        raise Exception('Could not find file.')

    with open(fname, 'w') as configfile:
        if run is not None:
            section = run
        else:
            section = 'Ground Truth Parameters'
        if not config.has_section(section):
            config.add_section(section)
        config.set(section, 'a1', a1)
        config.set(section, 'a2', a2)
        config.set(section, 'L', L)
        config.set(section, 'E', E)
        config.set(section, 'F', F)
        config.set(section, 'gamma', gamma)
        config.set(section, 'x0', x0)
        config.set(section, 'y0', y0)
        config.set(section, 'theta', theta)
        config.set(section, 'measure', json.dumps(yn.tolist()))
        if comment is not None:
            config.set(section, 'comment', comment)

        config.write(configfile)


if __name__ == '__main__':
    print 'numpy ', np.version.version
    # process input arguments
    parser = argparse.ArgumentParser(
        description='Fits data to heavy elastica model on HPC cluster.')
    parser.add_argument('input', metavar='I', type=str, help='input directory where the files are')
    # args = parser.parse_args()

    filename = 'elastica.cfg.txt'
    if not os.path.isfile(filename):
        with open(filename, 'w') as configfile:
            config = ConfigParser.RawConfigParser()
            config.add_section('General')
            config.set('General', 'Version', 'v0.1')
            config.write(configfile)

        # create some test data
        L = 10.0
        a1, a2 = 0.1, 0.6
        E, F = 1.0, 0.1
        x0, y0 = 200, 250
        theta, gamma = np.pi / 6, np.pi / 2
        Np = 100
        _yn = e.gen_test_data(a1, a2, L, E, F, gamma, x0, y0, theta, Np)
        comment = [a1, a2, L, E, F, gamma, x0, y0, theta]

        numsec = 50
        for r in range(0, numsec):
            # add_entry(a1, a2, L, E, F, gamma, x0, y0, theta, yn, run='instrument-'%r)
            add_entry(0, 0, 0, 0, 0, 0, 0, 0, 0, _yn, run='instrument-%04d' % r, comment=comment, fname=filename)

    # write pandas
    if not os.path.isfile('elastica.pandas.csv'):
        df = pd.DataFrame(data=[[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                          columns=['id', 'a1', 'a2', 'L', 'E', 'F', 'gamma', 'x0', 'y0', 'theta'])
        with open('elastica.pandas.csv', 'w') as f:
            df.to_csv(f, index=False)

    try:
        taskid = os.environ['SGE_TASK_ID']
        taskid = ast.literal_eval(taskid)
        section = 'instrument-%04d' % (taskid - 1)
        Np = 100
        job(section, filename, Np)

    except KeyError:
        print "Error: could not read SGE_TASK_ID from environment"
        exit(1)

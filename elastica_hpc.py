import ConfigParser
import argparse
import ast
import json
import logging
import os
import os.path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import basinhopping

import elastica as e

logging.basicConfig(level=logging.INFO)
# np.set_printoptions(3, suppress=True)

# reopen stdout file descriptor with write mode and 0 as the buffer size (unbuffered)
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

"""
module load mps/software python/2.7.12
# virtualenv --python=/mnt/pactsw/python/2.7.12/bin/python ~/py27
source ~/py27/bin/activate
deactivate
"""




def job(section, fname, Np=100):
    tseed = int(time.time() * 10e5 % 10e8)
    logging.info('time seed: %d' % tseed)
    np.random.seed(tseed)
    if os.path.isfile(fname):
        with open(fname, 'r') as configfile:
            config = ConfigParser.ConfigParser()
            config.readfp(configfile)

            logging.debug('sections found in file ' + str(config.sections()))

        if config.has_section(section):
            mstr = config.get(section, 'measure')
            yn = np.array(json.loads(mstr))
            # L, a1, a2, E, F, gamma, x0, y0, theta
            param_bounds = ((5.0, 20.0), (0.05, 0.6), (0.5, 1.0),
                            (0.01, 2.0), (0.0, 10.0), (-np.pi, np.pi),
                            (0, 120.), (0, 120.), (-np.pi, np.pi))
            x0 = [9.0, 0.2, 0.7, 0.1, 2, 10, 0, 0, 0]
            res = basinhopping(e.obj_minimize, x0, minimizer_kwargs={'bounds': param_bounds, 'args': (yn, Np)})
            objf = e.obj_minimize(res.x, yn)
            logging.info('x0=[%f,%f,%f,%f,%f,%f,%f,%f,%f] ' % tuple(res.x))
            logging.info('objective function final: %f' % objf)


def add_entry(yn, id, run=None, comment=None, fname='elastica.cfg.txt'):
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
        config.set(section, 'id', id)
        config.set(section, 'measure', json.dumps(yn.tolist()))
        if comment is not None:
            config.set(section, 'comment', comment)

        config.write(configfile)


if __name__ == '__main__':
    logging.debug('numpy ' + np.version.version)
    # process input arguments
    parser = argparse.ArgumentParser(
        description='Fits data to heavy elastica model on HPC cluster.')
    parser.add_argument('task_id', metavar='id', type=int, nargs='?', default=-1, help='SGE task ID if available.')
    parser.add_argument('--gen', dest='gen', action='store_true', help='generate configuration and pandas file.')
    parser.add_argument('--eb3', dest='eb3', action='store', default=None,
                        help='estimate model from data in eb3 csv file.')
    parser.add_argument('-r', '--repetitions', dest='rep', type=int, action='store', default=100,
                        help='number of measuring repetitions, available when the --gen or --eb3 flag is set.')
    args = parser.parse_args()

    config_fname = 'elastica.cfg.txt'
    if args.gen and args.eb3 is not None:
        raise Exception('Can\'t use --gen and --eb3 flags together.')

    if args.gen:
        with open(config_fname, 'w') as configfile:
            config = ConfigParser.RawConfigParser()
            config.add_section('General')
            config.set('General', 'Version', 'v0.1')
            config.set('General', 'Mode', 'Playing with simulated data')
            config.write(configfile)

        # create some test data
        L, a1, a2 = 10.0, 0.1, 0.6
        E, F, gamma = 1.0, 0.1, np.pi / 2
        x0, y0, theta = 20, 35, np.pi / 6
        Np = 100
        _yn = e.gen_test_data(L, a1, a2, E, F, gamma, x0, y0, theta, Np)
        comment = [L, a1, a2, E, F, gamma, x0, y0, theta]

        for r in range(0, args.rep):
            add_entry(_yn, 1, run='measure-%09d' % r, comment=comment, fname=config_fname)

    if args.eb3 is not None:
        with open(config_fname, 'w') as configfile:
            config = ConfigParser.RawConfigParser()
            config.add_section('General')
            config.set('General', 'Version', 'v0.1')
            config.set('General', 'Mode', 'Estimating real data')
            config.write(configfile)

        eb3_df = pd.read_csv(args.eb3)
        eb3_df = eb3_df.set_index('frame').sort_index()
        i = 1
        for id, df in eb3_df.groupby('id'):
            yn = np.array([df['xm'].tolist(), df['ym'].tolist()])
            for r in range(0, args.rep):
                add_entry(yn, id, run='measure-%09d' % i, fname=config_fname)
                i += 1

    try:
        taskid = os.environ['SGE_TASK_ID']
        taskid = ast.literal_eval(taskid)
        section = 'measure-%09d' % (taskid - 1)
        Np = 100
        job(section, config_fname, Np)

    except KeyError:
        logging.error('Error: could not read SGE_TASK_ID from environment')
        exit(1)

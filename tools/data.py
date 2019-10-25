from collections import OrderedDict

import seaborn as sns

import parameters
import tools.plot_tools as sp
from .manual_data import *

log = logging.getLogger(__name__)

names = OrderedDict([('1_N.C.', '-STLC'),
                     ('1_P.C.', '+STLC'),
                     ('1_DIC', 'DIC+STLC'),
                     ('1_Dynei', 'DHC+STLC'),
                     ('1_ASUND', 'Asunder+STLC'),
                     ('1_CENPF', 'CenpF+STLC'),
                     ('1_BICD2', 'Bicaudal+STLC'),
                     ('1_MCAK', 'MCAK+STLC'),
                     ('1_chTOG', 'chTog+STLC'),
                     ('2_Kines1', 'Kinesin1+STLC'),
                     ('2_CDK1_DA', 'DHC&Asunder+STLC'),
                     ('2_CDK1_DC', 'DHC&CenpF+STLC'),
                     ('2_CDK1_DK', 'DHC&Kinesin1+STLC'),
                     ('1_No10+', 'Nocodazole 10ng+STLC'),
                     ('1_CyDT', 'Cytochalsin D+STLC'),
                     ('1_Bleb', 'Blebbistatin+STLC'),
                     ('1_FAKI', 'FAKi+STLC'),
                     ('pc', '+STLC(2)'),
                     ('hset', 'Hset+STLC'),
                     ('kif25', 'Kif25+STLC'),
                     ('hset+kif25', 'Hset&Kif25+STLC'),
                     ('mother-daughter', '+STLC mother daughter')])
pt_color = sns.light_palette(sp.SUSSEX_COBALT_BLUE, n_colors=10, reverse=True)[3]
lbl_dict = {'+STLC': '+STLC',
            'mother-daughter': 'mother-daughter',
            'DHC+STLC': 'DHC\n+STLC',
            'Asunder+STLC': 'Asunder\n+STLC',
            'Bicaudal+STLC': 'Bicaudal\n+STLC',
            'MCAK+STLC': 'MCAK\n+STLC',
            'chTog+STLC': 'chTog\n+STLC',
            'DHC&CenpF+STLC': 'DHC&CenpF\n+STLC',
            'DHC&Asunder+STLC': 'DHC&Asunder\n+STLC',
            'CenpF+STLC': 'CenpF\n+STLC'}


def rename_conditions(df):
    for k, n in names.items():
        df.loc[df['condition'] == k, 'condition'] = n
    return df


def sorted_conditions(df, original_conds):
    conditions = [names[c] for c in original_conds]
    dfc = df[df['condition'].isin(conditions)]

    # sort by condition
    sorter_index = dict(zip(conditions, range(len(conditions))))
    dfc.loc[:, 'cnd_idx'] = dfc['condition'].map(sorter_index)
    dfc = dfc.set_index(['cnd_idx', 'run', 'Nuclei', 'Frame', 'Time']).sort_index().reset_index()

    return dfc, conditions


class Data():
    def __init__(self):
        logging.info('loading data for track plotting.')
        df_m = pd.read_pickle(parameters.compiled_data_dir + 'merge.pandas')
        df_msk = pd.read_pickle(parameters.compiled_data_dir + 'mask.pandas')
        df_mc = pd.read_pickle(parameters.compiled_data_dir + 'merge_centered.pandas')

        # filter original dataframe to get just data between centrosomes
        dfcntr = df_mc[df_mc['CentrLabel'] == 'A']
        dfcntr.loc[:, 'indiv'] = dfcntr['condition'] + '-' + dfcntr['run'] + '-' + dfcntr['Nuclei'].map(int).map(str)
        dfcntr.drop(
            ['CentrLabel', 'Centrosome', 'NuclBound', 'CentX', 'CentY', 'NuclX', 'NuclY', 'Speed', 'Acc'],
            axis=1, inplace=True)

        df_m.loc[:, 'indiv'] = df_m['condition'] + '|' + df_m['run'] + '|' + df_m['Nuclei'].map(int).map(str)
        df_m.loc[:, 'trk'] = df_m['indiv'] + '|' + df_m['Centrosome'].map(int).map(str)
        df_mc.loc[:, 'indiv'] = df_mc['condition'] + '|' + df_mc['run'] + '|' + df_mc['Nuclei'].map(int).map(str)
        df_mc.loc[:, 'trk'] = df_mc['indiv'] + '|' + df_mc['Centrosome']

        for id, dfc in df_m.groupby(['condition']):
            log.info('condition %s: %d tracks' % (id, len(dfc['indiv'].unique()) / 2.0))
        df_m = rename_conditions(df_m)
        df_mc = rename_conditions(df_mc)
        dfcntr = rename_conditions(dfcntr)

        self.df_m = df_m
        self.df_mc = df_mc
        self.df_msk = df_msk
        self.dfcntr = dfcntr

    def get_condition(self, conditions_list, centered=False):
        log.debug('getting condition {}'.format(conditions_list))
        if not centered:
            df, conds = sorted_conditions(self.df_m, conditions_list)
        else:
            df, conds = sorted_conditions(self.df_mc, conditions_list)

        return df, conds

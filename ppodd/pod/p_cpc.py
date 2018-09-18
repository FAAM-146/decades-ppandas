from ..decades import DecadesVariable
from .base import PPBase

class CPC(PPBase):

    inputs = [
        'CPC378_counts',
        'CPC378_sample_flow',
        'CPC378_sheath_flow',
        'CPC378_saturator_temp',
        'CPC378_growth_tube_temp',
        'CPC378_optics_temp'
    ]

    def declare_outputs(self):

        self.declare(
            'CPC_CNTS',
            units='1',
            frequency=10,
            long_name='Condensation Particle Counts measured by the TSI 3786'
        )

    def flag(self):
        d = self.d
        d['SATURATOR_TEMP_FLAG'] = 0
        d['GROWTH_TUBE_FLAG'] = 0
        d['OPTICS_TEMP_FLAG'] = 0
        d['SAMPLE_FLOW_FLAG'] = 0
        d['SHEATH_FLOW_FLAG'] = 0
        d['SATURATED_FLAG'] = 0

        d.loc[d['CPC378_saturator_temp'] > 6, 'SATURATOR_TEMP_FLAG'] = 1

        d.loc[d['CPC378_growth_tube_temp'] < 40.5, 'GROWTH_TUBE_FLAG'] = 1
        d.loc[d['CPC378_growth_tube_temp'] > 49.5, 'GROWTH_TUBE_FLAG'] = 1

        d.loc[d['CPC378_optics_temp'] < 40.5, 'OPTICS_TEMP_FLAG'] = 1
        d.loc[d['CPC378_optics_temp'] > 49.5, 'OPTICS_TEMP_FLAG'] = 1

        d.loc[d['CPC378_sample_flow'] < 270, 'SAMPLE_FLOW_FLAG'] = 2
        d.loc[d['CPC378_sample_flow'] > 330, 'SAMPLE_FLOW_FLAG'] = 2

        d.loc[d['CPC378_sheath_flow'] < 270, 'SHEATH_FLOW_FLAG'] = 3
        d.loc[d['CPC378_sheath_flow'] > 330, 'SHEATH_FLOW_FLAG'] = 3

        d.loc[d['CPC378_counts'] >= 1e6, 'SATURATED_FLAG'] = 2


    def process(self):
        self.get_dataframe()
        d = self.d

        self.flag()

        dv = DecadesVariable(d['CPC378_counts'], name='CPC_CNTS')

        for flag in ['SATURATOR_TEMP_FLAG', 'GROWTH_TUBE_FLAG',
                     'OPTICS_TEMP_FLAG', 'SAMPLE_FLOW_FLAG',
                     'SHEATH_FLOW_FLAG', 'SATURATED_FLAG']:

            dv.add_flag(d[flag])

        self.add_output(dv)

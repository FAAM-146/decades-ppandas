from ..decades import DecadesVariable, DecadesBitmaskFlag
from .base import PPBase
from .shortcuts import _c, _l, _o, _z

class CPC(PPBase):

    inputs = [
        'CPC378_counts',
        'CPC378_sample_flow',
        'CPC378_sheath_flow',
        'CPC378_saturator_temp',
        'CPC378_growth_tube_temp',
        'CPC378_optics_temp'
    ]

    @staticmethod
    def test():
        return {
            'CPC378_counts': (
                'data', _c([_z(30), _l(0, 2e4, 15), _l(2e4, 0, 15), _z(40)])
            ),
            'CPC378_sample_flow': ('data', 300 * _o(100)),
            'CPC378_sheath_flow': ('data', 300 * _o(100)),
            'CPC378_saturator_temp': ('data', 2 * _o(100)),
            'CPC378_growth_tube_temp': ('data', 45 * _o(100)),
            'CPC378_optics_temp': ('data', 45 * _o(100))
        }

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

        d.loc[d['CPC378_sample_flow'] < 270, 'SAMPLE_FLOW_FLAG'] = 1
        d.loc[d['CPC378_sample_flow'] > 330, 'SAMPLE_FLOW_FLAG'] = 1

        d.loc[d['CPC378_sheath_flow'] < 270, 'SHEATH_FLOW_FLAG'] = 1
        d.loc[d['CPC378_sheath_flow'] > 330, 'SHEATH_FLOW_FLAG'] = 1

        d.loc[d['CPC378_counts'] >= 1e6, 'SATURATED_FLAG'] = 1


    def process(self):
        self.get_dataframe()
        d = self.d

        self.flag()

        dv = DecadesVariable(d['CPC378_counts'], name='CPC_CNTS',
                             flag=DecadesBitmaskFlag)

        dv.flag.add_mask(d['SATURATOR_TEMP_FLAG'], 'saturator over temp')
        dv.flag.add_mask(d['GROWTH_TUBE_FLAG'], 'growth tube temp out of range')
        dv.flag.add_mask(d['OPTICS_TEMP_FLAG'], 'optics temp out of range')
        dv.flag.add_mask(d['SAMPLE_FLOW_FLAG'], 'sample flow out of range')
        dv.flag.add_mask(d['SHEATH_FLOW_FLAG'], 'sheath flow out of range')
        dv.flag.add_mask(d['SATURATED_FLAG'], 'counter saturated')

        self.add_output(dv)

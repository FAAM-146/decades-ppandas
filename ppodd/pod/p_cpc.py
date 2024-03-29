"""
Provides a processing modules for the TSI3786 condensation particle counter.
See the class doc-string for more info.
"""
from vocal.types import DerivedString

# pylint: disable=invalid-name
from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades.attributes import DocAttribute
from .base import PPBase, register_pp
from .shortcuts import _c, _l, _o, _z

SATURATOR_TEMP_VALID_MAX = 6
GROWTH_TUBE_TEMP_VALID_MIN = 40.5
GROWTH_TUBE_TEMP_VALID_MAX = 49.5
OPTICS_TEMP_VALID_MIN = 40.5
OPTICS_TEMP_VALID_MAX = 49.5
SAMPLE_FLOW_VALID_MIN = 270
SAMPLE_FLOW_VALID_MAX = 330
SHEATH_FLOW_VALID_MIN = 270
SHEATH_FLOW_VALID_MAX = 330
COUNTER_SATURATION = 1e6


@register_pp('core')
class CPC(PPBase):
    r"""
    Reports particle counts from the TSI 3786 condensation particle counter.
    Counts are reported as-is from the instrument; this module only provides
    flagging information.

    .. note::
        Prior to v23.3.0, the CPC included flags for both sample flow and 
        sheath flow. These have been combined into a single flag, and a
        cloud flag has been added, as CPC data are not valid in cloud. **The
        cloud flag is added by a separate flagging module**.
    """

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
        """
        Return some dummy input data for testing purposes.
        """
        return {
            'CPC_SN': ('const', DocAttribute(value='1234', doc_value=DerivedString)),
            'CPC378_counts': (
                'data', _c([_z(30), _l(0, 2e4, 15), _l(2e4, 0, 15), _z(40)]),
                10
            ),
            'CPC378_sample_flow': ('data', 300 * _o(100), 10),
            'CPC378_sheath_flow': ('data', 300 * _o(100), 10),
            'CPC378_saturator_temp': ('data', 2 * _o(100), 10),
            'CPC378_growth_tube_temp': ('data', 45 * _o(100), 10),
            'CPC378_optics_temp': ('data', 45 * _o(100), 10)
        }

    def declare_outputs(self):
        """
        Declare output variables.
        """

        self.declare(
            'CPC_CNTS',
            units='1',
            frequency=10,
            long_name='Condensation Particle Counts measured by the TSI 3786',
            instrument_manufacturer='TSI Incorporated',
            instrument_model=('Modified Water Filled 3786 Condensation '
                              'Particle Counter'),
            instrument_serial_number=self.dataset.lazy['CPC_SN'],
            comment=('Sampled through a modified Rosemount Aerospace Inc. '
                     'Type 102 Total Temperature Housing.')
        )

    def flag(self):
        """
        Provide flagging information. All flags here are based on simple bound
        limits.
        """

        d = self.d
        d['SATURATOR_TEMP_FLAG'] = 0
        d['GROWTH_TUBE_FLAG'] = 0
        d['OPTICS_TEMP_FLAG'] = 0
        d['SAMPLE_FLOW_FLAG'] = 0
        d['SHEATH_FLOW_FLAG'] = 0
        d['SATURATED_FLAG'] = 0

        d.loc[d['CPC378_saturator_temp'] > SATURATOR_TEMP_VALID_MAX,
              'SATURATOR_TEMP_FLAG'] = 1

        d.loc[d['CPC378_growth_tube_temp'] < GROWTH_TUBE_TEMP_VALID_MIN,
            'GROWTH_TUBE_FLAG'] = 1

        d.loc[d['CPC378_growth_tube_temp'] > GROWTH_TUBE_TEMP_VALID_MAX,
              'GROWTH_TUBE_FLAG'] = 1

        d.loc[d['CPC378_optics_temp'] < OPTICS_TEMP_VALID_MIN,
              'OPTICS_TEMP_FLAG'] = 1

        d.loc[d['CPC378_optics_temp'] > OPTICS_TEMP_VALID_MAX,
              'OPTICS_TEMP_FLAG'] = 1

        d.loc[d['CPC378_sample_flow'] < SAMPLE_FLOW_VALID_MIN,
              'SAMPLE_FLOW_FLAG'] = 1

        d.loc[d['CPC378_sample_flow'] > SAMPLE_FLOW_VALID_MAX,
              'SAMPLE_FLOW_FLAG'] = 1

        d.loc[d['CPC378_sheath_flow'] < SHEATH_FLOW_VALID_MIN,
              'SHEATH_FLOW_FLAG'] = 1

        d.loc[d['CPC378_sheath_flow'] > SHEATH_FLOW_VALID_MAX,
              'SHEATH_FLOW_FLAG'] = 1

        d.loc[d['CPC378_counts'] >= COUNTER_SATURATION, 'SATURATED_FLAG'] = 1

    def process(self):
        """
        Module entry hook.
        """

        self.get_dataframe()
        d = self.d

        self.flag()

        dv = DecadesVariable(d['CPC378_counts'], name='CPC_CNTS',
                             flag=DecadesBitmaskFlag)

        dv.flag.add_mask(
            d['SATURATOR_TEMP_FLAG'], 'saturator over temp',
            f'The saturator temperature is above {SATURATOR_TEMP_VALID_MAX} C'
        )
        dv.flag.add_mask(
            d['GROWTH_TUBE_FLAG'], 'growth tube temp out of range',
            ('Growth tube temperature is outside the valid range '
             f'[{GROWTH_TUBE_TEMP_VALID_MIN}, {GROWTH_TUBE_TEMP_VALID_MAX}] C')
        )
        dv.flag.add_mask(
            d['OPTICS_TEMP_FLAG'], 'optics temp out of range',
            ('Optics temperature is outside the valid range '
             f'[{OPTICS_TEMP_VALID_MIN}, {OPTICS_TEMP_VALID_MAX}] C')
        )
        dv.flag.add_mask(
            d['SAMPLE_FLOW_FLAG'] + d['SHEATH_FLOW_FLAG'],
            'sample or sheath flow out of range',
            ('Sample or sheath flows are outside the valid range '
             f'[{SAMPLE_FLOW_VALID_MIN}, {SAMPLE_FLOW_VALID_MAX}]')
        )
        dv.flag.add_mask(
            d['SATURATED_FLAG'], 'counter saturated',
            ('Counter has exceeded its saturation value, '
             f'{COUNTER_SATURATION:0.0f}')
        )

        self.add_output(dv)

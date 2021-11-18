"""
This module frovides flagging info for the WVSS-2 derived RH
"""
from .p_rosemount_temps import RosemountTempCloudFlag, RosemountTempDeltaFlag
from .base import FlaggingBase

class WVSS2RHTemperatureFlag(FlaggingBase):

    inputs = ['RH_ICE', 'RH_LIQ', 'TAT_DI_R']
    prerequisites = [RosemountTempCloudFlag, RosemountTempDeltaFlag]
    flagged = ['RH_ICE', 'RH_LIQ']

    def _get_flag(self):
        tat = self.dataset['TAT_DI_R']
        rh_ice = self.dataset['RH_ICE']
        rh_liq = self.dataset['RH_LIQ']
        tat_flag_ice = tat.flag().reindex(rh_ice.index)
        tat_flag_liq = tat.flag().reindex(rh_liq.index)

        return (tat_flag_ice, tat_flag_liq)

    def _flag(self, test=False):

        if test:
            tat_flag_ice = self.test_flag
            tat_flag_liq = self.test_flag
        else:
            tat_flag_ice, tat_flag_liq = self._get_flag()

        self.add_mask(
            'RH_ICE',
            tat_flag_ice, 'temperature_flagged',
            'The input temperature measurement has a non-zero flag value'
        )

        self.add_mask(
            'RH_LIQ',
            tat_flag_liq, 'temperature_flagged',
            'The input temperature measurement has a non-zero flag value'
        )

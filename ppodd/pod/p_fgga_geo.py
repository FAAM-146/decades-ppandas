"""
Provides a postprocessing module which provides geographical info for the FGGA
data file, provided from the core data file.
"""
import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from .base import PPBase, register_pp
from .shortcuts import _o

from ppodd.pod.p_gin import gin_name, gin_model, gin_manufacturer


@register_pp('fgga')
class FGGAGeo(PPBase):
    r"""
    This module takes geographic data (latitude, longitude and altitude) from
    the FAAM core data file, and reindexes it to match the temporal extent and
    resolution of data from the Fast Greenhouse Gas Analyser.
    """

    inputs = [
        'ch4_mol_frac',
        'co2_mol_frac',
        'LAT_GIN',
        'LON_GIN',
        'ALT_GIN'
    ]

    @staticmethod
    def test():
        """
        Return dummy inputs for testing.
        """
        return {
            'ch4_mol_frac': ('data', 400 * _o(100), 1),
            'co2_mol_frac': ('data', 400 * _o(100), 1),
            'LAT_GIN': ('data', 50 * _o(100), 1),
            'LON_GIN': ('data',  _o(100), 1),
            'ALT_GIN': ('data',  _o(1000), 1),
        }

    def declare_outputs(self):
        """
        Declare the outputs that are going to be written by this module.
        """

        self.declare(
            'latitude',
            units='degree_north',
            long_name=f'Latitude from {gin_name}',
            standard_name='latitude',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model,
            frequency=1
        )

        self.declare(
            'longitude',
            units='degree_east',
            long_name=f'Longitude from {gin_name}',
            standard_name='longitude',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model,
            frequency=1
        )

        self.declare(
            'altitude',
            units='m',
            long_name=f'Altitude from {gin_name}',
            standard_name='altitude',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model,
            frequency=1
        )


    def process(self):
        """
        Processing entry point.
        """
        index = self.dataset['co2_mol_frac'].index
        self.get_dataframe(method='onto', index=index)

        lat_flag = self.dataset['LAT_GIN'].flag
        lat_flag_df = lat_flag.df.reindex(index)

        lon_flag = self.dataset['LON_GIN'].flag
        lon_flag_df = lon_flag.df.reindex(index)

        alt_flag = self.dataset['ALT_GIN'].flag
        alt_flag_df = alt_flag.df.reindex(index)

        lat = DecadesVariable(
            {'latitude': self.d['LAT_GIN']}, flag=DecadesBitmaskFlag,
            flag_postfix='qcflag'
        )
        lon = DecadesVariable(
            {'longitude': self.d['LON_GIN']}, flag=DecadesBitmaskFlag,
            flag_postfix='qcflag'
        )
        alt = DecadesVariable(
            {'altitude': self.d['ALT_GIN']}, flag=DecadesBitmaskFlag,
            flag_postfix='qcflag'
        )

        for meaning in lat_flag.meanings:
            lat.flag.add_mask(lat_flag_df[meaning], meaning)

        for meaning in lon_flag.meanings:
            lon.flag.add_mask(lon_flag_df[meaning], meaning)

        for meaning in alt_flag.meanings:
            alt.flag.add_mask(alt_flag_df[meaning], meaning)

        self.add_output(lat)
        self.add_output(lon)
        self.add_output(alt)

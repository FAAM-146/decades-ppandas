"""
This module provides physical and instrument constants used in post-processing.
"""

# Speed of sount at zero altitude (m/s)
SPEED_OF_SOUND = 340.294

# ICAO Standard temperature at zero altitude
# (288.15 K = 15 C)
ICAO_STD_TEMP = 288.15

# ICAO Standard pressure at surface (hPa)
ICAO_STD_PRESS = 1013.25

# Stefan-Boltzmann constant
STEF_BOLTZ = 5.67e-8

# Max and min of the Radar Altimeter
# Rad Alt is 16 bit signed int, with least significant bit resolution of
# 0.25 ft, so max valid value is (((2**16) / 2) - 1) / 4
RADALT_MAX = 8191.75
RADALT_MIN = 0

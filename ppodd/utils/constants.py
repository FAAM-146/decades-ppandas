"""
This module provides physical constants used in post-processing.
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

# Freezing point in K
ZERO_C_IN_K = 273.15

# Gas constant of dry air (J / (kg K))
R_d = 287.058

# Gas constant of water vapour (J / (kg K))
R_v = 461.5

# Specific heats of dry air (d) and water vapour (v) at constant pressure (p)
# and constant volume (v)
c_vd = (5. / 2.) * R_d
c_pd = (7. / 2.) * R_d
c_vv = 3. * R_v
c_pv = 4. * R_v

# Molecular mass of chemical species
MOL_MASS_H20 = 18.0185
MOL_MASS_DRY_AIR = 28.97

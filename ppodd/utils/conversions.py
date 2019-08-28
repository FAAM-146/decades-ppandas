import numpy as np

def kelvin_to_celsius(myvar):
    return myvar - 273.15

def celsius_to_kelvin(myvar):
    return myvar + 273.15

def metres_to_feet(metres):
    return metres / 0.3048

def feet_to_metres(feet):
    return feet * 0.3048

def uv_to_spddir(u, v):
    _spd = (u**2 + v**2) ** .5
    _dir = np.arctan2(u/_spd, v/_spd) * 180 / np.pi
    return _spd, _dir

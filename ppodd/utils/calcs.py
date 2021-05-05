import numpy as np


def sp_mach(psp, sp, flag=False):

    STATIC_MIN = 100
    STATIC_MAX = 1050
    PITOT_MIN = 0
    PITOT_MAX = 125

    if len(psp) != len(sp):
        raise ValueError('Inputs are not the same length')

    _flag = np.zeros_like(psp)

    _flag[sp > STATIC_MAX] = 2
    _flag[sp < STATIC_MIN] = 2
    _flag[psp > PITOT_MAX] = 2
    _flag[psp < PITOT_MIN] = 2

    _flag[psp/sp < 0] = 3

    mach = np.sqrt(5 * ((1 + psp / sp)**(2 / 7) - 1))

    _flag[mach > .9] = 3
    _flag[mach <= 0] = 3

    if flag:
        return mach, _flag

    return mach


def true_air_temp(iat, mach, recovery_factor=1, flag=False):

    if len(iat) != len(mach):
        raise ValueError('Inputs are not the same length')

    _flag = np.zeros_like(iat)

    tat = iat / (1 + (0.2 * mach**2 * recovery_factor))

    if flag:
        return tat, _flag
    return tat


def true_air_temp_variable(iat, mach, eta=0, gamma=1):
    return iat * ((1. - eta) * (1. + ((gamma - 1.) / 2.) * mach**2.))**(-1.)

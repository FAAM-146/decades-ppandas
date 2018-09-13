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

    if flag:
        mach[_flag == 3] = 0
        return mach, _flag

    return mach




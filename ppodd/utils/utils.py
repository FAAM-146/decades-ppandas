import numpy as np

pd_freq = {
    1: '1S',
    2: '500L',
    4: '250L',
    8: '125L',
    10: '100L',
    16: '62500U',
    20: '50L',
    32: '31250000N',
    64: '15625000N',
    128: '7812500N',
    256: '3906250N'
}

def get_range_flag(var, limits, flag_val=2):
    """
    Get a flag variable which flags when a variable is outside a specified
    range.

      Args:
          var: the variable on which to base the flagging, np.array-like.
          limits: a 2-element array specifying valid-min and valid-max.

      Kwargs:
          flag_val: the value of the flag when var is outside limits. Default
                    2.

      Returns:
          flag: a flag [np.array] of same dimensions as var, with value 0
                wherever var in inside limits, and flag_val where are is
                outside limits.
    """
    flag = np.zeros_like(var)

    flag[var < limits[0]] = flag_val
    flag[var > limits[1]] = flag_val

    return flag

import datetime
import pandas as pd
import numpy as np
import yaml
import json

import ppodd.pod
from ppodd.decades import DecadesDataset, DecadesVariable

__all__ = ['NetCDFVariableVocabulary']

class NetCDFVariableVocabulary(object):
    def __init__(self):
        _vars = {}

        for pp_module in ppodd.pod.pp_modules:
            _mod = pp_module.test_instance()
            _mod.process()
            _mod.finalize()
            for _var in _mod.outputs:
                _vars[_var] = {}
                for attrk, attrv in _mod.outputs[_var].attrs.items():
                    if attrv is not None:
                        _vars[_var][attrk] = attrv

                _flag_var = '{}_FLAG'.format(_var)
                _vars[_flag_var] = {}
                for attrk, attrv in _mod.outputs[_var].flag.cfattrs.items():
                    if attrv is not None:
                        _vars[_flag_var][attrk] = self._escape_np(attrv)

        self._vars = _vars

    def __getitem__(self, key):
        return self._vars[key]

    def _escape_np(self, item):
        _np_ints = (np.int8, np.int16, np.int32, np.int64)

        if type(item) in _np_ints:
            return int(item)

        if type(item) == str or not np.iterable(item):
            return item

        _retlist = []
        for i in item:
            if type(i) in _np_ints:
                _retlist.append(int(i))
            else:
                _retlist.append(i)

        return _retlist

    def _escape_np_dict(self, _dict):
        _ret_dict = {}

        for key, value in _dict.items():
            _ret_dict[key] = self._escape_np(value)

        return _ret_dict

    @property
    def yaml(self):
        return yaml.dump(self._vars, Dumper=yaml.Dumper)

    @property
    def json(self):
        return json.dumps(self._vars, indent=4)

    @property
    def dict(self):
        return self._vars

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
        self.d = DecadesDataset(datetime.datetime.now())

        for pp_module in ppodd.pod.pp_modules:
            _mod = pp_module.test_instance(dataset=self.d)

        self.d.process()
        _vars = {}
        for _var in self.d.outputs:
            _vars[_var.name] = {}
            _attrs = self.d[_var.name].attrs
            for _attr_key, _attr_val in _attrs.items():
                if _attr_val is not None:
                    _vars[_var.name][_attr_key] = self._escape_np(_attr_val)

            _flag_var = '{}_FLAG'.format(_var.name)

            _vars[_flag_var] = self._escape_np_dict(
                self.d[_var.name].flag.cfattrs
            )

        self._vars = _vars

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
        return json.dumps(self._vars)

    @property
    def dict(self):
        return self._vars

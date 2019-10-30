import datetime
import pandas as pd
import yaml
import json

import ppodd.pod
from ppodd.decades import DecadesDataset, DecadesVariable

__all__ = ['NetCDFStructure']

class NetCDFStructure(object):
    def __init__(self):
        d = DecadesDataset(datetime.datetime.now())

        for pp_module in ppodd.pod.pp_modules:
            _mod = pp_module(d)
            for dec in _mod.declarations:
                faketime = datetime.datetime.now()
                data = pd.DataFrame(index=(faketime,))
                data[dec] = 0
                _mod.add_output(DecadesVariable(data, name=dec))
            _mod.finalize()

        _vars = {}
        for _var in d.variables:
            _vars[_var] = {}
            _attrs = d[_var].attrs
            for _attr_key, _attr_val in _attrs.items():
                if _attr_val is not None:
                    _vars[_var][_attr_key] = _attr_val

            _flag_var = '{}_FLAG'.format(_var)
            _vars[_flag_var] = d[_var].flag.cfattrs

        self._vars = _vars

    @property
    def yaml(self):
        return yaml.dump(self._vars, Dumper=yaml.Dumper)

    @property
    def json(self):
        return json.dumps(self._vars)

    @property
    def dict(self):
        return self._vars

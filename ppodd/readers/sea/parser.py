from datetime import datetime
import pytz

from .datatypes import *

__all__ = ['parser_f']

tz = pytz.UTC

parser_f = dict()
parser_f['d0'] = {
    'descr': 'Raw element power and temperature',
    'dtypes': d0_types,
    'names': d0_names,
    'long_names': d0_longnames,
    'units': d0_units,
    'converters': None
}
parser_f['d1'] = {
    'descr': 'Calculated total and liquid water contents',
    'dtypes': d1_types,
    'names': d1_names,
    'long_names': d1_longnames,
    'units': d1_units,
    'converters': None
}
parser_f['d2'] = {
    'descr': 'Element status information',
    'dtypes': d2_types,
    'names': d2_names,
    'long_names': d2_longnames,
    'units': d2_units,
    'converters': None
}
parser_f['d3'] = {
    'descr': 'Aircraft parameters',
    'dtypes': d3_types,
    'names': d3_names,
    'long_names': d3_longnames,
    'units': d3_units,
    'converters': {
        1: lambda x: datetime.strptime(x.decode(), '%Y/%m/%d').date(),
        2: lambda x: datetime.strptime(x.decode(), '%H:%M:%S.%f').time().replace(
            tzinfo=tz
        )
    }
}
parser_f['c0'] = {
    'descr': 'Sense element information',
    'dtypes': c0_types,
    'names': c0_names,
    'long_names': c0_longnames,
    'units': c0_units,
    'converters': None
}
parser_f['cpbx'] = {
    'descr': 'Power box configuration',
    'dtypes': cpbx_types,
    'names': cpbx_names,
    'long_names': cpbx_longnames,
    'units': cpbx_units,
    'converters': None
}
parser_f['cprb'] = {
    'descr': 'Probe configuration',
    'dtypes': cprb_types,
    'names': cprb_names,
    'long_names': cprb_longnames,
    'units': cprb_units,
    'converters': None
}
parser_f['cmcb'] = {
    'descr': 'Main cable configuration',
    'dtypes': cmcb_types,
    'names': cmcb_names,
    'long_names': cmcb_longnames,
    'units': cmcb_units,
    'converters': None
}
parser_f['cscb'] = {
    'descr': 'Secondary cable configuration',
    'dtypes': cscb_types,
    'names': cscb_names,
    'long_names': cscb_longnames,
    'units': cscb_units,
    'converters': None
}

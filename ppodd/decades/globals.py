import collections
import warnings

from ppodd.standard import faam_globals

STR_DERIVED_FROM_FILE = '<derived from file>'
GLOBAL_NOT_SET = 'GLOBAL_NOT_SET'


class GlobalNotSetError(Exception):
    pass


class NonStandardGlobalError(Exception):
    pass


class GlobalsCollection(object):
    REQUIRED_GLOBALS = [
        g for g in faam_globals['Globals'].keys()
        if faam_globals['Globals'][g]['required']
    ]

    OPTIONAL_GLOBALS = [
        g for g in faam_globals['Globals'].keys()
        if not faam_globals['Globals'][g]['required']
    ]

    def __init__(self, dataset=None, strict_mode=True):
        self._dataset = dataset
        self._globals = []
        self._data_globals = {}
        self._compliance = False
        self._strict_mode = strict_mode

        for key in self.REQUIRED_GLOBALS:
            self.add(Global(key, GLOBAL_NOT_SET))

    def __getitem__(self, key):
        for g in self._globals:
            if g.key == key:
                return self._compliancify(g)

        raise KeyError('{} not a global'.format(key))

    def __setitem__(self, key, value):
        if type(value) is dict:
            for _k, _v in value.items():
                __k = '_'.join((key, _k))
                self[__k] = _v

        else:
            self.add(Global(key, value))

    def __call__(self):
        return self.dict

    def _compliancify(self, glo):
        if self._compliance:
            if callable(glo.value) or glo.value == GLOBAL_NOT_SET:
                return STR_DERIVED_FROM_FILE
        else:
            if callable(glo.value):
                return glo.value()
            else:
                return glo.value
        return glo.value

    def set_compliance_mode(self, comp):
        self._compliance = comp

    def add(self, glo):
        if not isinstance(glo, Global):
            raise TypeError('globals must be of type <Global>')

        for i in self._globals:
            if i.key == glo.key:
                self._globals.remove(i)
                break

        if glo.key not in self.REQUIRED_GLOBALS + self.OPTIONAL_GLOBALS:
            _message = f'Global \'{glo.key}\' is not defined in standard'
            if self.strict_mode:
                raise NonStandardGlobalError(_message)
            else:
                warnings.warn(_message)

        self._globals.append(glo)

    @property
    def strict_mode(self):
        return self._strict_mode

    @strict_mode.setter
    def strict_mode(self, strict):
        self._strict_mode = bool(strict)

    @property
    def is_compliant(self):
        for required in self.REQUIRED_GLOBALS:
            if self[required] == GLOBAL_NOT_SET:
                return False
        return True

    def add_data_global(self, param, attrs):
        self._data_globals[param] = attrs

    def static_items(self):
        return self._as_dict(dynamic=False).items

    def _as_dict(self, dynamic=True):
        _dict = {}
        for glo in self._globals:
            _dict[glo.key] = self._compliancify(glo)

        if not dynamic:
            return _dict

        for name, _pack in self._data_globals.items():
            key, attrs = _pack
            try:
                var = self._dataset[key]
            except KeyError:
                if self._compliance:
                    _dict[name] = STR_DERIVED_FROM_FILE
                continue

            for _attr in attrs:
                var = getattr(var, _attr)

            _dict[name] = self._compliancify(Global(name, var))

        return _dict

    @property
    def keys(self):
        return self.dict.keys()

    @property
    def values(self):
        return self.dict.values()

    @property
    def dict(self):
        _dict = self._as_dict()
        _sorted = collections.OrderedDict()
        for key in sorted(_dict.keys()):
            _sorted[key] = _dict[key]
        return _sorted


class Global(object):
    def __init__(self, key, value):
        self._key = key
        self._value = value

    def __repr__(self):
        return r'Global({!r}, {!r})'.format(self.key, self.value)

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

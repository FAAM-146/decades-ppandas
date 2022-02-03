import abc
from atexit import register
from re import I
import uuid

from typing import TYPE_CHECKING

from ppodd.utils.utils import stringify_if_datetime, try_to_call

if TYPE_CHECKING:
    from ppodd.decades import DecadesDataset

attribute_helpers = []
def register_attribute_helper(cls):
    attribute_helpers.append(cls)
    return cls


class AttributeHelper(abc.ABC):   

    def __init__(self, dataset):
        self.dataset = dataset
        self.__post_init__()

    def __post_init__(self):
        pass

    @property
    def attributes(self):
        _attrs = []
        for attr in dir(self):
            if attr == 'attributes':
                continue
            try:
                if isinstance(getattr(type(self), attr), property):
                    _attrs.append(attr)
                
            except AttributeError:
                pass

        return _attrs

    
@register_attribute_helper
class FlightTime(AttributeHelper):

    def __post_init__(self):
        self._takeoff_time = None
        self._landing_time = None

    @property
    def takeoff_time(self):
        """
        :term:`datetime-like`: Return the latest takeoff time of the data set,
        as determined by the last time at which PRTAFT_wow_flag changing 1 -> 0
        """

        if self._takeoff_time is not None:
            return self._takeoff_time

        try:
            wow = self.dataset['PRTAFT_wow_flag']()
        except KeyError:
            return None

        try:
            self._takeoff_time = wow.diff().where(
                wow.diff() == -1
            ).dropna().tail(1).index[0]
        except IndexError:
            return None

        return self._takeoff_time

    @property
    def landing_time(self):
        """
        :term:`datetime-like`: Return the latest landing time of the dataset,
        as determined by the last time at which PRTAFT_wow_flag changes from
        0 -> 1
        """

        if self._landing_time is not None:
            return self._landing_time

        try:
            wow = self.dataset['PRTAFT_wow_flag']()
        except KeyError:
            return None

        try:
            self._landing_time = wow.diff().where(
                wow.diff() == 1
            ).dropna().tail(1).index[0]
        except IndexError:
            return None

        return self._landing_time
        

@register_attribute_helper
class IDProvider(AttributeHelper):

    def __post_init__(self):
        self._data_id = None

    @property
    def data_id(self):
        return lambda: self._data_id

    @data_id.setter
    def data_id(self, data_id):
        self._data_id = data_id


@register_attribute_helper
class UUIDProvider(AttributeHelper):

    @property
    def uuid(self):
        def _closure():
            try:
                _id = self.dataset.globals['id']
                _date = stringify_if_datetime(
                    self.dataset.globals['date_created']
                ) 
            except Exception:
                return str(uuid.uuid4())

            _string = f'{_date}{_id}'

            return str(uuid.uuid3(uuid.NAMESPACE_DNS, _string))

        return _closure


@register_attribute_helper
class DurationProvider(AttributeHelper):

    def _get_duration(self):
        start = self.dataset.lazy['TIME_MIN_CALL']
        end = self.dataset.lazy['TIME_MAX_CALL']
        delta = (end - start).total_seconds()

        hours = int(delta // 3600)
        minutes = int(delta // 60 % 60)
        seconds = int(delta % 60)

        a = f'PT{hours}H{minutes}M{seconds}S'
        return a

    @property
    def time_coverage_duration(self):
        def _closure():
            start = self.dataset.lazy['TIME_MIN_CALL']
            end = self.dataset.lazy['TIME_MAX_CALL']
            delta = (end() - start()).total_seconds()
            hours = int(delta // 3600)
            minutes = int(delta // 60 % 60)
            seconds = int(delta % 60)

            return f'PT{hours}H{minutes}M{seconds}S'

        return _closure


@register_attribute_helper
class GeoBoundsProvider(AttributeHelper):

    def get_map(self, attr):
        lower = getattr(self, f'{attr}_min')
        upper = getattr(self, f'{attr}_max')
        return {
            'l': lower,
            'u': upper
        }

    def point(self, defstr):
        lon_id, lat_id  = defstr

        lon_map = self.get_map('lon')
        lat_map = self.get_map('lat')

        try:
            return f'{lon_map[lon_id]:0.2f} {lat_map[lat_id]:0.2f}'
        except TypeError:
            return None

    def get_props(self):
        self.lat_min = self.dataset.globals['geospatial_lat_min']
        self.lat_max = self.dataset.globals['geospatial_lat_max']
        self.lon_min = self.dataset.globals['geospatial_lon_min']
        self.lon_max = self.dataset.globals['geospatial_lon_max']

    @property
    def geospatial_bounds(self):
        self.get_props()
        p1 = self.point('ll')
        p2 = self.point('lu')
        p3 = self.point('uu')
        p4 = self.point('ul')

        try:
            return f'POLYGON(({p1}, {p2}, {p3}, {p4}, {p1}))'
        except TypeError:
            return None
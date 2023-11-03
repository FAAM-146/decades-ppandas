import abc
import uuid

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from ppodd.utils.utils import stringify_if_datetime

if TYPE_CHECKING:
    from ppodd.decades import DecadesDataset

attribute_helpers = []
def register_attribute_helper(cls):
    attribute_helpers.append(cls)
    return cls


class AttributeHelper(abc.ABC):
    """
    An abstract class which should be implemented to produce an attrubute
    helper. Implementations should be decorated with @register_attribute_helper
    to ensure that they are attached to DecadesDatasets.
    """   

    def __init__(self, dataset: 'DecadesDataset') -> None:
        self.dataset = dataset
        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    @property
    def attributes(self) -> list[property]:
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
    """
    Provides `takeoff_time` and `landing_time` attributes
    """

    def __post_init__(self) -> None:
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
    """
    Provides a `data_id` attribute. This is expected to be set at some point
    during the processing chain - canonically when the filename of the 
    resulting output has been set.
    """

    def __post_init__(self):
        self._data_id = None

    @property
    def data_id(self) -> str:
        """
        Return the `data_id`
        """
        return lambda: self._data_id

    @data_id.setter
    def data_id(self, data_id: str) -> None:
        """
        Set the `data_id`
        """
        self._data_id = data_id


@register_attribute_helper
class UUIDProvider(AttributeHelper):
    """
    Provides a `uuid` attribute
    """

    @property
    def uuid(self) -> Callable[[], str]:
        """
        Provide a UUID, factored as a callable. If the dataset globals `id`
        and `date_created` attributes are available, returns a UUID3 of 
        id+date_created, hashed with `NAMESPACE_DNS`, otherwise returns a
        random UUID4.
        """
        def _closure() -> str:
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
    """
    Provides a `time_coverage_duration` atttibute, giving the length of the 
    dataset in ISO format (e.g. PT1H23M12S).
    """

    def _get_duration(self) -> str:
        start = self.dataset.lazy['TIME_MIN_CALL']
        end = self.dataset.lazy['TIME_MAX_CALL']
        delta = (end - start).total_seconds()

        hours = int(delta // 3600)
        minutes = int(delta // 60 % 60)
        seconds = int(delta % 60)

        a = f'PT{hours}H{minutes}M{seconds}S'
        return a

    @property
    def time_coverage_duration(self) -> Callable[[], str]:
        """
        Provide `time_coverage_duration`. This is calculated as the difference
        between `TIME_MAX_CALL` and `TIME_MIN_CALL`, formatted in ISO8601
        timedelta format.
        """
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
    """
    Provides a `geospatial_bounds` attribute.
    """

    def get_map(self, attr: str) -> dict[str, float]:
        lower = getattr(self, f'{attr}_min')
        upper = getattr(self, f'{attr}_max')
        return {
            'l': lower,
            'u': upper
        }

    def point(self, defstr: str) -> Optional[str]:
        lon_id, lat_id  = defstr

        lon_map = self.get_map('lon')
        lat_map = self.get_map('lat')

        try:
            return f'{lon_map[lon_id]:0.2f} {lat_map[lat_id]:0.2f}'
        except TypeError:
            return None

    def get_props(self) -> None:
        self.lat_min = self.geospatial_lat_min
        self.lat_max = self.geospatial_lat_max
        self.lon_min = self.geospatial_lon_min
        self.lon_max = self.geospatial_lon_max

    @property
    def geospatial_bounds(self) -> Optional[str]:
        """
        Provides a WKT representation of the flight envelope, as a rectangle
        with upper and lower lat/lon bounds.
        """
        self.get_props()
        p1 = self.point('ll')
        p2 = self.point('lu')
        p3 = self.point('uu')
        p4 = self.point('ul')

        if None in (p1, p2, p3, p4):
            return None

        try:
            return f'POLYGON(({p1}, {p2}, {p3}, {p4}, {p1}))'
        except TypeError:
            return None
        
    @property
    def geospatial_lat_min(self) -> Optional[float]:
        """
        Provides the lower latitude bound of the flight envelope
        """
        try:
            return np.float32(self.dataset[self.dataset.lat].min())
        except Exception:
            return None
        
    @property
    def geospatial_lat_max(self) -> Optional[float]:
        """
        Provides the upper latitude bound of the flight envelope
        """
        try:
            return np.float32(self.dataset[self.dataset.lat].max())
        except Exception:
            return None
        
    @property
    def geospatial_lon_min(self) -> Optional[float]:
        """
        Provides the lower longitude bound of the flight envelope
        """
        try:
            return np.float32(self.dataset[self.dataset.lon].min())
        except Exception:
            return None
        
    @property
    def geospatial_lon_max(self) -> Optional[float]:
        """
        Provides the upper longitude bound of the flight envelope
        """
        try:
            return np.float32(self.dataset[self.dataset.lon].max())
        except Exception:
            return None
    
    @property
    def geospatial_alt_min(self) -> Optional[float]:
        """
        Provides the lower altitude bound of the flight envelope
        """
        try:
            return np.float32(self.dataset[self.dataset.alt].min())
        except Exception:
            return None
        
    @property
    def geospatial_alt_max(self) -> Optional[float]:
        """
        Provides the upper altitude bound of the flight envelope
        """
        try:
            return np.float32(self.dataset[self.dataset.alt].max())
        except Exception:
            return None
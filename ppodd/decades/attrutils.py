import abc
from typing import TYPE_CHECKING


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
        self._data_id = lambda: None

    @property
    def data_id(self):
        return lambda: self._data_id

    @data_id.setter
    def data_id(self, data_id):
        self._data_id = data_id

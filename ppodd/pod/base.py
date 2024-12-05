"""
This module provides an abstract base class for processing modules. Processing
modules should subclass PPBase to be included in the processing.
"""

# pylint: disable=invalid-name, too-many-arguments

from __future__ import annotations

import datetime
import logging
from typing import Any, Callable, Literal, cast

import numpy as np
import pandas as pd

from ppodd.utils import pd_freq, unwrap_array
from ppodd.decades import DecadesDataset, DecadesVariable
from ppodd.decades import DecadesBitmaskFlag, DecadesClassicFlag
from ppodd.decades import flags

DFJoinMethods = Literal["outerjoin", "onto"]
TestDataValue = tuple[Literal["const"], Any] | tuple[Literal["data"], np.ndarray, int]

logger = logging.getLogger(__name__)


class PPRegister:
    """
    Register for processing modules. This is used to store the processing
    modules, and to allow the processing code to find the appropriate modules
    to run.
    """

    def __init__(self) -> None:
        """
        Initialize the register.
        """
        self._dict: dict[str, list[type[PPBase]]] = {}

    def append(self, pp_group: str, cls: type[PPBase]) -> None:
        """
        Add a processing module to the register.

        Args:
            pp_group: the group to which the processing module belongs.
            cls: the processing module to add.
        """
        try:
            self._dict[pp_group].append(cls)
        except KeyError:
            self._dict[pp_group] = [cls]

    def modules(
        self, group_name: str, date: datetime.date | None = None
    ) -> list[type[PPBase]]:
        """
        Get a list of processing modules for a given group. Modules are filtered
        by the date given.
        """
        _modules = self._dict[group_name]
        if date is not None:
            _modules = [
                i
                for i in _modules
                if (date > i.VALID_AFTER) and (date < i.DEPRECIATED_AFTER)
            ]
        return _modules


pp_register = PPRegister()


def register_pp(pp_group: str) -> Callable[[type[PPBase]], type[PPBase]]:
    """
    Provide a decorator to register processing modules with the
    processing code.
    """

    def inner(cls: type[PPBase]) -> type[PPBase]:
        try:
            pp_register.append(pp_group, cls)
        except KeyError:
            pp_register.append(pp_group, cls)
        return cls

    return inner


class PPBase(object):
    """
    PPBase is an abstract base class from which processing modules should
    inherit to be picked up by the processing code.
    """

    freq = pd_freq

    inputs: list[str] = []
    ignored_upstream_flags: list[str] = []

    VALID_AFTER = datetime.date.min
    DEPRECIATED_AFTER = datetime.date.max

    def __init__(self, dataset: DecadesDataset, test_mode: bool = False) -> None:
        """
        Initialize a class instance

        Args:
            dataset: the DecadesDataset from which the processing is running

        Kwargs:
            test_mode: if true, run the module in test mode, for unit testing,
                building documentation etc.
        """
        self.dataset = dataset
        self.outputs: dict[str, DecadesVariable] = {}
        self.declarations: dict[str, dict[str, Any]] = {}
        self.test_mode = test_mode
        self.declare_outputs()
        self.d: pd.DataFrame | None = None

    def __str__(self) -> str:
        """
        Implement str() - simply return the name of the class.
        """
        return self.__class__.__name__

    def declare_outputs(self) -> None:
        """
        Add outputs to be written. This is run during initialization of the
        module, and should contain a call to self.declare for each output
        variable.
        """
        raise NotImplementedError

    def process(self) -> None:
        """Do the actual postprocessing - entry point for the module"""
        raise NotImplementedError

    def declare(self, name: str, **kwargs: Any) -> None:
        """
        Declare the output variables that the processing module is going to
        create.

        Args:
            name: the name of the declared variable

        Kwargs: key-value pairs used to initialize the decades variable.
        """
        if name in self.dataset.variables:
            if not self.dataset.allow_overwrite:
                raise ValueError(
                    f"Cannot declare {name}, as it already exists in Dataset"
                )

            self.dataset.remove(name)

        self.declarations[name] = kwargs

    def add_flag_mod(self, flag: pd.Series, output: DecadesVariable) -> None:

        def _add_mask_flag(flag: pd.Series, output: DecadesVariable) -> None:
            assert isinstance(output.flag, DecadesBitmaskFlag)

            output.flag.add_mask(
                flag,
                "flagged in qc",
                (
                    "Manually flagged during QC. Check metadata for the reason "
                    "for flagging."
                ),
            )

        def _add_classic_flag(flag: pd.Series, output: DecadesVariable) -> None:
            assert isinstance(output.flag, DecadesClassicFlag)

            values = sorted(list(output.flag.cfattrs["flag_values"]))
            if not values:
                flag_value = 1
            else:
                flag_value = int(values[-1]) + 1
            output.flag.add_meaning(
                flag_value,
                "flagged in qc",
                (
                    "Manually flagged during QC. Check metadata for the reason "
                    "for flagging."
                ),
            )
            output.flag.add_flag(flag_value * flag)

        if output.flag is None:
            return

        if isinstance(output.flag, DecadesBitmaskFlag):
            return _add_mask_flag(flag, output)

        return _add_classic_flag(flag, output)

    def finalize(self) -> None:
        """
        Finalization tasks: ensure all declared outputs have been written and
        propogate the outputs to the calling DecadesDataset.
        """

        for declaration in self.declarations:
            if declaration not in self.outputs:
                raise RuntimeError(
                    "Output declared but not written: {}".format(declaration)
                )

        try:
            input_flag = self.get_input_flag()
        except Exception:
            logger.warning(
                f"Failed to get input flag for " f"{self.__class__.__name__}"
            )

            input_flag = pd.Series([], dtype=np.int8)  # TODO: check this

        for name, output in self.outputs.items():
            # Apply any modifications specified in self.dataset._variable_mods
            # - canonically as specified in the flight constants file
            # Note that we shouldn't really be accessing what's nominally a
            # private method of dataset.
            #
            # pylint: disable=protected-access
            # TODO: add a public interface to DecadesDataset
            if name in self.dataset._variable_mods:
                for key, value in self.dataset._variable_mods[name].items():
                    setattr(output, key, value)

            # Apply any extra flagging specified in the flight constants file
            if name in self.dataset._flag_mods:
                flag_mods = self.dataset._flag_mods[name]
                flag = pd.Series(np.zeros_like(output.data), index=output.index)
                for flag_mod in flag_mods:
                    start = datetime.datetime.strptime(
                        flag_mod["start"], "%Y-%m-%dT%H:%M:%S"
                    )
                    end = datetime.datetime.strptime(
                        flag_mod["end"], "%Y-%m-%dT%H:%M:%S"
                    )
                    flag.loc[(flag.index >= start) & (flag.index <= end)] = 1
                self.add_flag_mod(flag, output)

            if type(output.flag) is DecadesBitmaskFlag:
                if np.any(~np.isnan(input_flag)):
                    input_flag = input_flag.reindex(output.index)
                    input_flag = input_flag.bfill()
                    input_flag = input_flag.ffill()
                    output.flag.add_mask(
                        input_flag,
                        flags.DEPENDENCY,
                        (
                            "A dependency, used in the derivation of this "
                            "variable, has a non-zero flag."
                        ),
                    )

            if type(output.flag) is DecadesClassicFlag:
                try:
                    a = output.flag.meanings
                    flag_value = max(list(output.flag.meanings)) + 1
                except ValueError:
                    flag_value = 1

                if np.any(~np.isnan(input_flag)) and np.any(input_flag):
                    input_flag = input_flag.reindex(output.index)
                    input_flag = input_flag.bfill()
                    input_flag = input_flag.ffill()
                    input_flag[input_flag > 0] = flag_value

                    output.flag.add_meaning(
                        flag_value,
                        flags.DEPENDENCY,
                        (
                            "A dependency, used in the derivation of this "
                            "variable has a non-zero flag."
                        ),
                    )

                    output.flag.add_flag(input_flag)

            if self.test_mode:
                output.doc_mode = True

            # And append the output to the dataset
            self.dataset.add_output(output)

        self.outputs = {}
        self.d = None

    @staticmethod
    def onto(
        dataframe: pd.DataFrame,
        index: pd.DatetimeIndex,
        limit: int = 1,
        period: int | None = None,
    ) -> pd.DataFrame:
        """
        Project a dataframe onto another index, interpolating across any gaps
        introduced to a given limit.

        Args:
            dataframe: the Pandas DataFrame or Series to reindex.
            index: the index to reindex dsataframe onto.

        Kwargs:
            limit [1]: maximum number of consecutive gaps to fill, passed to
                index.interpolate
            period [None]: passed to index.interpolate.

        Returns:
            the DataFrame dataframe reindexed onto index.
        """

        return (
            dataframe.reindex(index.union(dataframe.index).sort_values())
            .interpolate("time", limit=limit, period=period)
            .loc[index]
        )

    def get_input_flag(self) -> pd.Series:
        cflag = pd.Series(dtype=np.int8)
        for _input in self.inputs:
            if _input in self.ignored_upstream_flags:
                continue
            try:
                flag = self.dataset[_input].flag()
                flag = flag[flag > 0].dropna()
            except (TypeError, AttributeError):
                # TypeError for no flag, AttributeError for constant
                continue

            cindex = cflag.index.union(flag.index).sort_values().unique()

            # TODO: we probably don't want to fill with 0 in the case of
            # differing frequencies
            cflag = cflag.reindex(cindex)
            cflag = cflag.bfill()
            cflag = cflag.ffill()
            cflag.loc[flag.index] = np.maximum(flag, cflag.loc[flag.index])

        cflag[cflag > 1] = 1
        return cflag

    def get_dataframe(
        self,
        method: DFJoinMethods = "outerjoin",
        index: pd.DatetimeIndex | None = None,
        limit: int = 1,
        circular: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        """
        Create a dataframe containing all of the inputs required to run the
        processing module. The dataframe is stored in the class attribute 'd'.

        Kwargs:
            method ['outerjoin']: specify how to merge variables into a single
                dataframe. Should be one of 'outerjoin' (default), which uses a
                union of indexes, or 'onto', which puts all variables onto a
                specified index. If 'onto', index should preferably be
                specified. If index is not given, the index of the first
                required input will be used.
            index [None]: the index to reindex input variables onto. Used when
                method='onto'. If method is onto but no index is given, the
                index of the first required input will be used.
            limit [1]: the maximum number of consecutive empty datapoints to
                interpolate over when reindexing.
            circular [None]: an iterable of the names of any input variables
                which are congruent to themselves mod 360.
            exclude [None]: an iterable of the names of input variables to
                exclude from the resulting dataframe.
        """

        if circular is None:
            circular = []

        if exclude is None:
            exclude = []

        df: pd.Series | pd.DataFrame = pd.DataFrame()

        _inputs = [
            i
            for i in self.inputs
            if i not in self.dataset.constants and i not in exclude
        ]

        if method == "outerjoin":

            # Create a joined dataframe
            for _input in _inputs:
                df = df.join(self.dataset[_input]().dropna(), how="outer")

            # Ensure the dataframe has a continuous index at the frequency of
            # the highest frequency input.
            _freq = np.max([self.dataset[i].frequency for i in _inputs])
            df = df.reindex(
                pd.date_range(start=df.index[0], end=df.index[-1], freq=pd_freq[_freq])
            )

        elif method == "onto":
            # Interpolate onto a given index when creating the dataframe

            if index is None:
                df = self.dataset[_inputs[0]]()
                index = cast(pd.DatetimeIndex, df.index)
                _start = 1
            else:
                df = pd.DataFrame(index=index)
                _start = 0

            for _input in _inputs[_start:]:
                _input_name = _input

                if _input in circular:
                    # Deal with periodic (e.g. heading) data

                    _tmp = self.dataset[_input_name]()
                    _data = _tmp.values.copy()

                    _input_df = pd.DataFrame([], index=_tmp.index)

                    _input_df[_input_name] = _data
                    _input_df[_input_name] = unwrap_array(_input_df[_input_name])

                else:
                    _input_df = self.dataset[_input_name]()

                # Interpolate onto the instance dataframe
                df[_input_name] = (
                    _input_df.reindex(index.union(_input_df.index).sort_values())
                    .interpolate("linear", limit=limit)
                    .loc[index]
                )

                if _input_name in circular:
                    df[_input_name] %= 360

        self.d = pd.DataFrame(df)

    def add_output(self, variable: DecadesVariable) -> None:
        """
        Add an output variable. Initially outputs are stored in the processing
        module 'outputs' attribute (a list). These are propagated to the parent
        dataset when self.finalize is called.

        Args:
            variable: the output variable to add.
        """

        # All outputs must be declared before being added (see self.declare)
        if variable.name not in self.declarations:
            raise RuntimeError("Output {} has not been declared".format(variable.name))

        # Set all of the variable attribute specified in self.declare
        for item, value in self.declarations[variable.name].items():
            setattr(variable, item, value)

        try:
            # Find where the first and last non-nan data points are, and trim
            # the data to this period.
            good_start = int(np.min(np.where(~np.isnan(variable.data))))
            _df = variable()
            start_time = _df.index[good_start]
            end_time = _df.index[-1]
            variable.trim(start_time, end_time)  # type: ignore #  we cant type Series index

        except ValueError:
            # If a value is raised, there's no good data. In this case there's
            # no point including the data in the final product, so set its
            # write attribute to false.
            if not self.test_mode:
                variable.write = False
                print("Warning: no good data: {}".format(variable.name))

        self.outputs[variable.name] = variable

    def ready(self) -> tuple[bool, list[str] | None]:
        """
        Determine if the module can run. A module can run if all of its
        required inputs are present in its parent dataset.

        Returns:
            a 2-tuple, the first element of which is a boolean indicating if
                the module is ready. If the module able to run, the second
                element will be None, otherwise it will be a list of variables
                which are required to run the module, but which are not
                available in the parent dataset.
        """
        _missing_variables = []

        for _name in self.inputs:
            _inputs = self.dataset.variables + list(self.dataset.constants.keys())
            if _name not in _inputs:
                _missing_variables.append(_name)

        if _missing_variables:
            return False, _missing_variables

        return True, None

    @classmethod
    def test_instance(cls, dataset: DecadesDataset | None = None) -> PPBase:
        """
        Return a test instance of a postprocessing module, initialized with a
        DecadesDataset containing the modules test data.
        """
        now = datetime.datetime.now().replace(microsecond=0)

        if dataset is None:
            d = DecadesDataset(now.date())
        else:
            d = dataset

        if callable(cls.test):
            # pylint: disable=not-callable
            _test = cls.test()
        else:
            _test = cls.test

        for key, val in _test.items():

            if val[0] == "const":
                d.constants[key] = val[1]

            elif val[0] == "data":

                if len(val) != 3:
                    raise ValueError("Test data must be a tuple of (data, freq)")

                _dvalues, _freq = val[1:]
                _dt = datetime.timedelta(seconds=1 / _freq)
                freq = "{}N".format((1 / _freq) * 1e9)

                start_time = datetime.datetime(*d.date.timetuple()[:3])
                end_time = start_time + datetime.timedelta(seconds=len(_dvalues)) - _dt

                hz1_index = pd.date_range(
                    start=start_time, periods=len(_dvalues), freq="S"
                )
                full_index = pd.date_range(start=start_time, end=end_time, freq=freq)

                data = pd.Series(_dvalues, hz1_index).reindex(full_index).interpolate()

                var = DecadesVariable(data, name=key, frequency=_freq)
                d.add_input(var)

        return cls(d, test_mode=True)

    @staticmethod
    def test() -> dict[str, TestDataValue]:
        raise NotImplementedError
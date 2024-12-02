import datetime
import logging
import re

import numpy as np
import pandas as pd

from ppodd.decades.variable import DecadesVariable
from ppodd.readers import register, FileReader
from ppodd.decades.flags import DecadesBitmaskFlag, DecadesClassicFlag


FFI_1001 = "1001"
DFMT = "%Y%m%d"

logger = logging.getLogger(__name__)


class NasaAmes1001Reader(object):
    def __init__(self, ames_file):
        self.ames_file = ames_file
        self.nlhead: int | None = None
        self.ffi = None
        self.meta = {}

        with open(ames_file, "r") as f:
            _nlhead, ffi = f.readline().strip().split()

        self.nlhead = int(_nlhead)
        self.ffi = ffi

        if self.ffi != FFI_1001:
            raise ValueError(f"Can only read FFA={FFI_1001} files. This file is {ffi}.")

        logger.debug(f"{self.nlhead} header lines.")

    def read(self):
        header = []
        with open(self.ames_file, "r") as f:
            if not self.nlhead:
                raise ValueError("No header lines found.")
            for i in range(self.nlhead):
                header.append(f.readline())

        header = header[1:]
        self.meta["oname"] = header.pop(0).strip()
        self.meta["org"] = header.pop(0).strip()
        self.meta["sname"] = header.pop(0).strip()
        self.meta["mname"] = header.pop(0).strip()
        self.meta["ivol"], self.meta["nvol"] = [int(i) for i in header.pop(0).split()]

        _dates = header.pop(0).split()
        self.meta["date"] = datetime.datetime.strptime("".join(_dates[:3]), DFMT)
        self.meta["rdate"] = datetime.datetime.strptime("".join(_dates[3:]), DFMT)
        self.meta["dx"] = int(header.pop(0))
        self.meta["xname"] = header.pop(0).strip()
        self.meta["nv"] = int(header.pop(0))
        self.meta["vscal"] = [float(i) for i in header.pop(0).split()]
        self.meta["vmiss"] = [float(i) for i in header.pop(0).split()]

        self.meta["vname"] = []
        for i in range(self.meta["nv"]):
            self.meta["vname"].append(header.pop(0).strip())

        self.meta["nscoml"] = int(header.pop(0))

        self.meta["scom"] = []
        for i in range(self.meta["nscoml"]):
            self.meta["scom"].append(header.pop(0).strip())

        self.meta["nncoml"] = int(header.pop(0))
        self.meta["ncom"] = []
        for i in range(self.meta["nncoml"]):
            self.meta["ncom"].append(header.pop(0).strip())

        data = pd.read_csv(
            self.ames_file,
            skiprows=self.nlhead,
            header=None,
            delimiter=" ",
            index_col=0,
            names=self.meta["vname"],
        )

        # Remove data marked as missing
        for col, missing in zip(self.meta["vname"], self.meta["vmiss"]):
            data.loc[data[col] == missing, col] = np.nan

        # Scale data
        for col, scale in zip(self.meta["vname"], self.meta["vscal"]):
            data[col] *= scale

        data.index = [
            self.meta["date"] + datetime.timedelta(seconds=i) for i in data.index
        ] # type: ignore # TODO

        self.data = data


@register(patterns=["faam-fgga.+.na"])
class FGGAReader(FileReader):
    def _extract_units(self):
        pattern = re.compile("(?P<name>.+), in (?P<units>.*).*")
        self.metadata["long_names"] = []
        self.metadata["units"] = []
        for i in self.metadata["vname"]:
            x = pattern.findall(i)
            if x:
                self.metadata["long_names"].append(x[0][0])
                self.metadata["units"].append(x[0][1])
            else:
                self.metadata["long_names"].append(i)
                self.metadata["units"].append(None)

    def read(self):
        if len(self.files) > 1:
            raise ValueError("Only 1 FGGA file currently accepted")
        

        reader = NasaAmes1001Reader(self.files[0].filepath)
        reader.read()
        self.metadata = reader.meta
        self.data = reader.data
        self._extract_units()

        short_names = ["FGGA_CO2", "FGGA_CO2_FLAG", "FGGA_CH4", "FGGA_CH4_FLAG"]

        dataset = self.files[0].dataset
        if dataset is None:
            raise ValueError("No dataset associated with file.")

        for i in (0, 2):
            var = DecadesVariable(
                {short_names[i]: self.data[self.metadata["vname"][i]]},
                long_name=self.metadata["long_names"][i],
                units=self.metadata["units"][i],
                frequency=1,
                flag=None,
                write=False,
            )

            flag = DecadesVariable(
                {short_names[i + 1]: self.data[self.metadata["vname"][i + 1]]},
                units=None,
                frequency=1,
                flag=None,
                write=False,
            )
            dataset.add_input(var)
            dataset.add_input(flag)

        dataset.add_global("comment", "\n".join(self.metadata["ncom"]))

import datetime
import logging
import re

import pandas as pd

from ppodd.decades import DecadesVariable
from ppodd.readers import register, FileReader


logger = logging.getLogger(__name__)


@register(patterns=['CCN.+.csv'])
class CCNReader(FileReader):

    def read(self):
        for _file in self.files:
            date = None
            rex = re.compile('Date.*(\d{2}/\d{2}/\d{2}).*')
            cnt = 0
            with open(_file.filepath, 'r') as f:
                while True:
                    # Scan through the file until we have the date and the
                    # header line
                    cnt += 1
                    line = f.readline()
                    header = [i.strip() for i in line.split(',')]
                    if date is None:
                        _date = rex.findall(line)
                        if _date:
                            date = datetime.datetime.strptime(
                                _date[0], '%d/%m/%y'
                            ).date()
                    if len(header) > 50:
                        break

            # The header isn't quite right - We first want to remove the first
            # element (Time, as this will be the index) and add 'A' and 'B'
            # versions of the final element
            header = header[1:]
            header.append(header[-1])
            header[-2] += ' A'
            header[-1] += ' B'

            # Read the data using pandas, ensuring uniqueness of the index
            data = pd.read_csv(_file.filepath, header=0, names=header,
                               skiprows=cnt-1, index_col=0, parse_dates=True)
            data = data.groupby(data.index).last()

            # It's only the time that's reported. Using parse_dates will use
            # today as the date, so we need to correct for this using the date
            # extracted from the header of the data file
            delta = datetime.date.today() - date
            data.index -= delta

            for col in data.columns:
                name = col.replace(' ', '_')
                if not name.startswith('CCN_'):
                    name = f'CCN_{name}'
                _file.dataset.add_input(
                    DecadesVariable(
                        {name: data[col]},
                        units=None,
                        frequency=1,
                        flag=None
                    )
                )

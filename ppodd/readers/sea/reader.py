import numpy as np

from ppodd.readers import FileReader, register
from ppodd.decades import DecadesVariable

from .utils import to_dataframe
from .parser import parser_f

@register(patterns=['.*\.wcm'])
class WcmFileReader(FileReader):
    def read(self):
        for _file in self.files:
            print(f'Reading {_file}')
            dfs, metadata = to_dataframe(_file.filepath)


            for k in dfs.keys():
                df = dfs[k]
                for i, name in enumerate(parser_f[k]['names']):
                    _freq = int(
                        np.timedelta64(1, 's') / dfs[k].index.freq.delta
                    )

                    _data = df[name].values

                    _varname = 'SEAPROBE_{}_{}'.format(
                        k, name.replace('el', '')
                    )

                    _var = DecadesVariable(
                        {_varname: _data},
                        index=df.index,
                        long_name=parser_f[k]['long_names'][i],
                        frequency=_freq,
                        units=parser_f[k]['units'][i]
                    )

                    _file.dataset.add_input(_var)

        for key, value in metadata.items():
            _file.dataset.constants['SEA_{}'.format(key)] = value

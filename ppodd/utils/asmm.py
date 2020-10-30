import argparse
import datetime

import numpy as np

from bs4 import BeautifulSoup
from netCDF4 import Dataset

__all__ = ('asmm_geofill',)

REPLACEMENTS = (
    ('MissionMetadata.GeographicalRegion.GeographicBoundingBox.westBoundLongitude',
     'LON_GIN', np.min),
    ('MissionMetadata.GeographicalRegion.GeographicBoundingBox.eastBoundLongitude',
     'LON_GIN', np.max),
    ('MissionMetadata.GeographicalRegion.GeographicBoundingBox.northBoundLatitude',
     'LAT_GIN', np.max),
    ('MissionMetadata.GeographicalRegion.GeographicBoundingBox.southBoundLatitude',
     'LAT_GIN', np.min),
    ('MissionMetadata.GeographicalRegion.GeographicBoundingBox.minAltitude',
     'ALT_GIN', np.min),
    ('MissionMetadata.GeographicalRegion.GeographicBoundingBox.maxAltitude',
     'ALT_GIN', np.max)
)

def _nc_reduce(filename, var, func):
    """
    Take a netCDF file, extract a given variable and reduce it with a given
    function.

    Args:
        filename - a netCDF file to read.
        var - the variable name to extract from netCDF file.
        func - a callable which acts to reduce data from var to a single value

    Returns:
        The return value of 'func' when given 'var' from netCDF file 'filename'
    """
    with Dataset(filename) as nc:
        data = nc[var][:].ravel()
        return func(data)

def _load_asmm(asmm_file):
    """
    Load an ASMM xml file using BS4 and return the element tree.

    Args:
        asmm_file - the asmm xml file to load.

    Returns:
        the BS4 representation of the contents of 'asmm_file'
    """
    with open(asmm_file, 'r') as f:
        return BeautifulSoup(f.read(), 'lxml-xml')

def _asmm_get_flightdata(asmm_file):
    """
    Return flight number and flight date from an ASMM file

    Args:
         asmm_file - the asmm xml file to load.

    Returns:
        A dict containing 'flight_number' and 'date'
    """

    asmm = _load_asmm(asmm_file)

    return {
        'flight_number':
        asmm.MissionMetadata.FlightInformation.FlightNumber.string.strip(),
        'date': datetime.datetime.strptime(
            asmm.MissionMetadata.FlightInformation.Date.string.strip(),
            '%Y-%m-%d'
        )
    }

def _get_output_file(asmm_file, user=None):
    """
    Return a FAAM-standard ASMM filename from an arbitrarily named ASMM file.

    Args:
        asmm_file - the asmm xml file to load

    Kwargs:
        user [None] - the user who created this asmm file. Should be a three
                      character code. Defaults to 'fm1'.

    Returns:
        A filename of the form asmm_faam_{date}_{fltnum}_{user}.xml
    """

    if user is None:
        user = 'fm1'

    file_format = 'asmm_faam_{date}_{fltnum}_{user}.xml'
    flight_data = _asmm_get_flightdata(asmm_file)

    return file_format.format(
        date=flight_data['date'].strftime('%Y%m%d'),
        fltnum=flight_data['flight_number'].lower(),
        user=user
    )


def asmm_geofill(asmm_file, core_file, output_file):
    """
    Add geographic bounds information to an ASMM file.

    Args:
        asmm_file - the asmm file to operate on
        core_file - the FAAM core netcdf containing geographical info
        output_file - the output asmm file to write
    """

    asmm = _load_asmm(asmm_file)

    if output_file is None:
        output_file = _get_output_file(asmm_file)

    for node_def, var, reducer in REPLACEMENTS:
        for _node in node_def.split('.'):
            node = getattr(asmm, _node)

        _value = _nc_reduce(core_file, var, reducer)
        node.string = f'{_value:0.2f}'

    with open(output_file, 'w') as f:
        f.write(asmm.prettify())

    return output_file

def _get_parser():
    """
    Build a command line parser.

    Returns:
        parser - A command line parser for this program.
    """

    parser = argparse.ArgumentParser(
        description='Add geographic information to a EUFAR ASMM file'
    )

    parser.add_argument('asmm', metavar='asmm', type=str, nargs=1,
                       help='The ASMM file to edit')

    parser.add_argument('core', metavar='corefile', type=str, nargs=1,
                       help='The core data file providing geographical info')

    parser.add_argument('--user', '-u', metavar='user', type=str, nargs=1,
                        default=[None],
                        help='Who created this ASMM? Three character code')

    parser.add_argument(
        '--output', '-o', metavar='output',
        help=('Filename to write. By default a standard FAAM ASMM filename is '
              'used'), type=str, nargs=1, default=None
    )

    parser.add_argument(
        '--delete', dest='delete', action='store_true',
        help='delete the original ASMM file',
    )

    return parser

if __name__ == '__main__':
    # Script entry point
    parser = _get_parser()
    args = parser.parse_args()

    if args.output:
        output = args.output[0]
    else:
        output = _get_output_file(args.asmm[0], args.user[0])

    asmm_geofill(args.asmm[0], args.core[0], output)

    if(args.delete):
        os.remove(args.asmm[0])

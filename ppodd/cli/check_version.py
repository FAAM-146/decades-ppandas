import argparse
import logging
import os
import tempfile
import warnings

import numpy as np

from netCDF4 import Dataset

import ppodd

from ppodd.decades import DecadesDataset

from vocal.application.check import print_checks, p, TS
from vocal.checking import ProductChecker
from vocal.netcdf import NetCDFReader


version = ppodd.version()
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


def get_parser() -> argparse.ArgumentParser:
    """
    Argument parser for the check_data script. This function will return an argument
    parser for the check_data script.
    """
    parser = argparse.ArgumentParser(
        description='Check processed data against example data'
    )
    parser.add_argument('data_dir', type=str, help='Directory containing data files')
    parser.add_argument('-p', '--previous', action='store_true', help='Check against the previous version')
    return parser


def main() -> None:
    """
    Main function for checking data. This function will compare the processed data
    with the example data.
    """
    
    parser = get_parser()
    args = parser.parse_args()

    data_files = get_data_files(args.data_dir, previous=args.previous)

    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        processed_files = process_data(data_files['input_files'])

        print('Running checks...')
        for f in processed_files:
            compare_files(f, [i for i in data_files['output_files'] if os.path.basename(i) == f][0])
    

def compare_files(file1: str, file2: str) -> None:
    """
    Compare two netCDF files. This function will compare the two files and print out
    any differences between the two files. It leverages the vocal library to do the
    bulk of the comparison.

    Args:
        file1: The first file to compare (processed)
        file2: The second file to compare (example)
    """
    
    # First run vocal to check metadata etc.
    checker = ProductChecker('.')
    f1 = NetCDFReader(file1).dict
    f2 = NetCDFReader(file2).dict
    checker.compare_container(f2, f1)
    p.ignore_info = True
    p.ignore_warnings = True
    print_checks(checker, file1, file2)
    
    # Now compare the data
    nc1 = Dataset(file1)
    nc2 = Dataset(file2)

    for v in nc1.variables:

        if v not in nc2.variables:
            print(f'{TS.FAIL}{v} not in example{TS.ENDC}')
            continue

        if not np.array_equal(nc1.variables[v][:], nc2.variables[v][:]):
            print(f'{TS.FAIL}{v} differ{TS.ENDC}')

    for v in nc2.variables:
        if v not in nc1.variables:
            print(f'{TS.FAIL}{v} not in processed{TS.ENDC}')
            

def process_data(input_files: list[str]) -> tuple[str, str]:
    """
    Run a processing job on the input files. This function will process the input files
    and write the output to a temporary directory.

    Args:
        input_files: A list of input files to process

    Returns:
        tuple[str, str]: The processed files
    """

    ppodd.package_logger.setLevel(logging.CRITICAL)
    d = DecadesDataset()

    for f in input_files:
        d.add_file(f)
    d.trim = True
    print('Loading input data...')
    d.load()

    print('Running processing...')
    d.process()

    date = d.date.strftime('%Y%m%d')
    fltnum = d.globals['flight_number']

    print('Writing output files...')
    core_file = f'core_faam_{date}_v005_r0_{fltnum}_prelim.nc'
    core_1hz_file = f'core_faam_{date}_v005_r0_{fltnum}_1hz_prelim.nc'
    d.write(core_file)
    d.write(core_1hz_file, freq=1)

    return core_file, core_1hz_file


def get_data_files(data_dir: str, previous: bool) -> dict[str, list[str]]:
    """
    Collect the data files from the data directory. This function will return a dictionary
    containing the input and output files.

    Args:
        data_dir: The directory containing the data files
        previous: Whether to check against the previous version

    Returns:
        dict[str, list[str]]: The input and output files
    """
    minor_version = '.'.join([i for i in ppodd.version().split('.')[:2]])

    if previous:
        minor_version_x, minor_version_y = minor_version.split('.')
        minor_version_y = str(int(minor_version_y) - 1)
        minor_version = f'{minor_version_x}.{minor_version_y}'

    version_data_dir = os.path.join(data_dir, minor_version)
    input_files = os.listdir(os.path.join(version_data_dir, 'input'))
    output_files = os.listdir(os.path.join(version_data_dir, 'output'))

    if not os.path.exists(version_data_dir):
        raise FileNotFoundError(f'Data directory {version_data_dir} does not exist')
    
    print(f'\nSoftware version is {ppodd.version()}')
    print(f'Checking against {os.path.basename(version_data_dir)}\n')

    return {
        'input_files': [os.path.join(os.getcwd(), version_data_dir, 'input', i) for i in input_files],
        'output_files': [os.path.join(os.getcwd(), version_data_dir, 'output', i) for i in output_files]
    }


if __name__ == '__main__':
    main()
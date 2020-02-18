import datetime
import os
import glob
import zipfile
import shutil
import re
import logging
import tempfile
import sys
import pathlib
import yaml

def read_config():
    config_dir  = os.path.join(
        os.path.expanduser('~'), '.decades-ppandas'
    )

    settings_file = os.path.join(config_dir, 'settings.yaml')

    with open(settings_file, 'r') as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    return settings


logger = logging.getLogger('autoproc')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(read_config()['logging']['autoproc'])
fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(fh_formatter)
logger.addHandler(fh)


class DecadesPPandasProcessor(object):
    def __init__(self, fltnum, fltdate, files, tempdir=None, flight_folder=None):
        self.fltnum = fltnum
        self.fltdate = fltdate
        self.files = files
        self.tempdir = tempdir
        self.flight_folder = flight_folder

    def process(self):
        sys.path.append(
            os.path.join(pathlib.Path().home(), 'vcs', 'decades-ppandas')
        )
        from ppodd.decades import DecadesDataset
        from ppodd.writers import NetCDFWriter
        from ppodd.report import ReportCompiler

        if self.tempdir and os.path.exists(self.tempdir):
            os.chdir(self.tempdir)
        else:
            self.tempdir = tempfile.TemporaryDirectory().name
            os.makedirs(self.tempdir)
            os.chdir(self.tempdir)

        logger.info('processing in {}'.format(self.tempdir))

        d = DecadesDataset(self.fltdate)
        for _file in self.files:
            d.add_file(_file)
        d.load()
        d.process()

        output_file = 'core_faam_{date}_v005_r0_{fltnum}{postfix}_prelim.{ext}'
        full_file = output_file.format(
            date=self.fltdate.strftime('%Y%m%d'),
            fltnum=self.fltnum,
            postfix='',
            ext='nc'
        )
        onehz_file =  output_file.format(
            date=self.fltdate.strftime('%Y%m%d'),
            fltnum=self.fltnum,
            postfix='_1hz',
            ext='nc'
        )


        writer = NetCDFWriter(d)

        logger.info(f'writing {full_file}')
        writer.write(full_file)

        logger.info(f'writing {onehz_file}')
        writer.write(onehz_file, freq=1)

        d.run_qa()

        report = ReportCompiler(
            d, flight_number=self.fltnum,
            token='dbc98605-be9c-4197-a953-ef37418c80ef',
            flight_folder=self.flight_folder
        )

        report.make()


    def publish(self, output_dir, secondary_nc_dir=None):
        def _logmv(f, t):
            logger.debug(f'moving {f} -> {t}')
            shutil.move(f, os.path.join(t, os.path.basename(f)))
        def _logcp(f, t):
            logger.debug(f'copying {f} -> {t}')
            shutil.copy2(f, os.path.join(t, os.path.basename(f)))

        qa_dir = os.path.join(output_dir, 'qa_figures')
        _dirs = [output_dir, qa_dir]

        if secondary_nc_dir is not None:
            _dirs.append(secondary_nc_dir)

        for _dir in _dirs:
            os.makedirs(_dir, exist_ok=True)

        for _file in glob.glob(os.path.join(self.tempdir, '*.pdf')):
            if _file.startswith('flight-report'):
                continue
            _logmv(_file, qa_dir)

        for _file in glob.glob(os.path.join(self.tempdir, '*')):

            if _file.endswith('csv'):
                continue

            if _file.endswith('nc'):
                _logcp(_file, secondary_nc_dir)

            _logmv(_file, output_dir)

    def cleanup(self):
        shutil.rmtree(self.tempdir)

class AutoProcessor(object):

    TANK_PREFERENCES = ['FISH', 'SEPTIC']
    RAWDLU_TEMPLATE = 'core_faam_{date}_r0_{fltnum}_rawdlu.zip'
    FLTCST_PATTERN = 'flight-cst_faam_{date}_r*_{fltnum}.yaml'

    def __init__(self, tcp_def_dir=None, fltcst_dir=None, output_base=None,
                 flight_folder_glob=None, secondary_output_base=None):

        self.tcp_def_dir = tcp_def_dir
        self.fltcst_dir = fltcst_dir
        self.output_base = output_base
        self.secondary_output_base = secondary_output_base
        self.flight_folder_glob = flight_folder_glob
        self.tempdir = None

    @property
    def _temp_set_exists(self):
        return self.tempdir and os.path.isdir(self.tempdir)

    def _chdir_temp(self):
        os.chdir(self.tempdir)

    def _get_tank_filename(self, fltnum):
        fltnum = fltnum.upper()

        flight_folder = self.flight_folder_glob.format(flight=fltnum)
        decades_folder = os.path.join(flight_folder, f'{fltnum}_decades')

        logger.info(f'DECADES folder: {decades_folder}')

        tank_files = glob.glob(os.path.join(decades_folder, '*.zip'))
        logger.debug(f'available tank files: {tank_files}')
        tank_file = None

        for pref in self.TANK_PREFERENCES:
            _tank_file = f'{fltnum}_{pref}.zip'
            _tank_files_base = [os.path.basename(i) for i in tank_files]

            if _tank_file in _tank_files_base:
                tank_file = os.path.join(
                    os.path.dirname(tank_files[0]), _tank_file
                )
                logger.debug(f'using {tank_file}')
                return tank_file

        raise RuntimeError('No tank file found')

    def _date_from_dlu_file(self, dlu_file, fltnum):
        logger.debug(f'getting date from {dlu_file} for {fltnum}')
        dlu = dlu_file.split('_')[0]
        fltnum = fltnum.upper()

        date = datetime.datetime.strptime(
            dlu_file,
            f'{dlu}_%Y%m%d_%H%M%S_{fltnum}'
        )

        logger.debug(f'parsed date as {date}')
        return date

    @property
    def flights_to_process(self):
        """
        Return a list of tuples containing flight numbers and corresponding
        flight dates.
        """

        flights = []
        _files = glob.glob(
            os.path.join(
                self.fltcst_dir,
                self.FLTCST_PATTERN.replace(
                    '{date}', '*'
                ).replace(
                    '{fltnum}', '*'
                )
            )
        )

        if not _files:
            return None

        rex_fltnum = re.compile('.+([a-z][0-9]{3}).+')
        rex_date = re.compile('_faam_([0-9]{8})_r.+')

        for _file in _files:
            _fltnum = rex_fltnum.findall(_file)[0]
            _fltdate = datetime.datetime.strptime(
                rex_date.findall(_file)[0], '%Y%m%d'
            )

            flights.append((_fltnum, _fltdate))

        return flights

    def make_rawdlu(self, fltnum):

        if not self._temp_set_exists:
            raise RuntimeError('No temp dir available')
        self._chdir_temp()

        tank_file = self._get_tank_filename(fltnum)
        logger.info(f'Using {tank_file}')

        shutil.copy(
            tank_file, os.path.join(self.tempdir, os.path.basename(tank_file))
        )

        logger.info('Extracting {}'.format(os.path.basename(tank_file)))
        _zipfile = zipfile.ZipFile(os.path.basename(tank_file))
        _zipfile.extractall()

        logger.debug('removing {}'.format(os.path.basename(tank_file)))
        os.remove(os.path.basename(tank_file))

        _local_decades_folder = os.path.join(self.tempdir, fltnum.upper())
        if os.path.isdir(_local_decades_folder):
            local_decades = True
            os.chdir(_local_decades_folder)
        else:
            local_decades = False

        dlu_flightnums = list(
            set([i.split('_')[-1] for i in glob.glob('CORCON*')])
        )

        try:
            dlu_flightnums.remove('XXXX')
        except ValueError:
            pass

        logger.debug(f'flight numbers from DLU data: {dlu_flightnums}')

        if len(dlu_flightnums) > 1:
            logger.info('removing DLU data with no flight number')
            for _file in glob.glob('*XXXX'):
                logger.debug(f'removing {_file}')
                os.remove(_file)

            fltnum_to_rm = sorted(dlu_flightnums)[0].upper()
            logger.info(f'removing DLU data from {fltnum_to_rm}')
            for _file in glob.glob(f'*{fltnum_to_rm}'):
                logger.debug(f'removing {_file}')
                os.remove(_file)

        else:
            for _file in glob.glob('*XXXX'):
                logger.debug(f'renaming {_file} for {fltnum}')
                os.rename(_file, _file.replace('XXXX', fltnum.upper()))

        for _file in glob.glob('*'):
            size = os.stat(_file).st_size
            if size == 0:
                logging.debug(f'removing zero-size file {_file}')
                os.remove(_file)

        # TODO: This is a BAAAD way to do this!
        gin_files = sorted(glob.glob('GIN*'))
        for _file in gin_files[:-1]:
            logger.debug(f'removing {_file}')
            os.remove(_file)

        dlus = list(
            set([i.split('_')[0] for i in glob.glob('*')])
        )

        _corcon_file = sorted(glob.glob('CORCON*'))[-1]
        flight_date = self._date_from_dlu_file(
            dlu_file=_corcon_file, fltnum=fltnum
        )

        for dlu in dlus:
            try:
                tcp_file = glob.glob(
                    os.path.join(self.tcp_def_dir, f'{dlu}*csv')
                )[0]
                logger.info(f'using definition {tcp_file} for {dlu}')
            except IndexError:
                logger.warning(f'failed to get definition file for {dlu}')
                continue

            shutil.copy(
                tcp_file,
                os.path.join(self.tempdir, os.path.basename(tcp_file))
            )

        rawdlu_file = self.RAWDLU_TEMPLATE.format(
            fltnum=fltnum.lower(), date=flight_date.strftime('%Y%m%d')
        )

        all_files = glob.glob('*')

        logger.info(f'creating raw file {rawdlu_file}')
        with zipfile.ZipFile(rawdlu_file, 'w') as _zip:
            for _file in all_files:
                logger.debug(f'zipping {_file}')
                _zip.write(_file)

        for _file in all_files:
            logger.debug(f'removing {_file}')
            os.remove(_file)

        if local_decades:
            logger.debug('renaming {rawdlu_file} (local_decades)')
            os.rename(
                rawdlu_file,
                os.path.join(self.tempdir, os.path.basename(rawdlu_file))
            )

    def run(self):
        lockfile = os.path.join(
            os.path.expanduser('~'),
            '.decades-ppandas',
            '.lock'
        )

        if os.path.exists(lockfile):
            logger.info('Autoproc is locked')
            return

        with open(lockfile, 'w'):
            pass

        try:
            self._run()
        except:
            raise
        finally:
            os.remove(lockfile)

    def _run(self):
        if not self.flights_to_process:
            logger.info('Nothing to do')
            return
        else:
            logger.info(
                'flights to process: {}'.format(self.flights_to_process)
            )

        for (fltnum, fltdate) in self.flights_to_process:
            logger.info(f'processing {fltnum}')
            with tempfile.TemporaryDirectory() as _temp:
                self.tempdir = _temp

                self.make_rawdlu(fltnum)
                os.chdir(self.tempdir)

                try:
                    shutil.rmtree(fltnum.upper())
                except Exception:
                    logger.warning('Failed to remove dir: {}'.format(fltnum.upper()))

                constants_pattern = os.path.join(
                    self.fltcst_dir,
                    self.FLTCST_PATTERN.format(
                        fltnum=fltnum.lower(),
                        date=fltdate.strftime('%Y%m%d')
                    )
                )

                constants_file = sorted(glob.glob(constants_pattern))[-1]

                shutil.copy(
                    constants_file,
                    os.path.join(
                        self.tempdir, os.path.basename(constants_file)
                    )
                )

                try:
                    flight_folder = glob.glob(
                        self.flight_folder_glob.format(flight=fltnum.upper())
                    )[0]
                except IndexError:
                    flight_folder = None

                processor = DecadesPPandasProcessor(
                    fltnum, fltdate,
                    [os.path.join(self.tempdir, i) for i in os.listdir(_temp)],
                    _temp, flight_folder=flight_folder
                )

                processor.process()

                flight_dir = fltdate.strftime(
                    '{fltnum}-%b-%d'.format(fltnum=fltnum)
                ).lower()

                output_dir = os.path.join(
                    self.output_base, fltdate.strftime('%Y'), flight_dir
                )

                secondary_nc_dir = os.path.join(
                    self.secondary_output_base, fltdate.strftime('%Y'),
                    flight_dir
                )

                processor.publish(output_dir, secondary_nc_dir=secondary_nc_dir)

                os.remove(constants_file)


if __name__ == '__main__':
    AutoProcessor(**read_config()['autoproc']).run()

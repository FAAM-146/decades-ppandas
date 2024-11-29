import collections
import logging
import traceback

import ppodd.flags
import ppodd

logger = logging.getLogger(__name__)


class DecadesProcessor:
    def __init__(self, dataset):
        self.dataset = dataset

    def process(self, modname=None):
        """
        Run processing modules.

        Args:
            modname (str, optional): the name of the module to run.
        """

        from ppodd.pod.base import pp_register

        if self.dataset.trim:
            self.dataset._trim_data()

        # If a module name is given, then we're only going to try to run that
        # module.
        if modname is not None:
            # Import all of the processing modules
            mods = pp_register.modules(self.dataset.pp_group, date=self.dataset.date)

            # Find an run the correct module
            for mod in mods:
                if mod.__name__ is modname:
                    _mod = mod(self.dataset)
                    _mod_ready, _missing = _mod.ready()
                    if _mod_ready:
                        # We have all the inputs to run the module
                        _mod.process()
                        _mod.finalize()
                        self.dataset.run_process_hooks(module_name=modname)
                    else:
                        # The module  could not be run, due to lacking inputs.
                        logger.debug(
                            ('Module {} not ready: '
                             'inputs not available {}').format(
                                 mod.__name__, ','.join(_missing))
                        )

            # Don't run further modules
            return

        # Initialize qa modules
        #self.qa_modules = [qa(self) for qa in ppodd.qa.qa_modules]

        # Initialize postprocessing modules
        _pp_modules = []
        for pp in pp_register.modules(self.dataset.pp_group, date=self.dataset.date):
            try:
                _pp_modules.append(pp(self.dataset))
            except Exception as err: # pylint: disable=broad-except
                logger.warning('Couldn\'t init {}: {}'.format(pp, str(err)))
        self.pp_modules = collections.deque(_pp_modules)

        # Initialize flagging modules
        self.flag_modules = [flag(self) for flag in ppodd.flags.flag_modules]

        # Clear all outputs whice have been defined previously
        self.dataset._backend.clear_outputs()

        self.completed_modules = []
        self.failed_modules = []

        temp_modules = []

        module_ran = True
        while self.pp_modules and module_ran:

            module_ran = False

            pp_module = self.pp_modules.popleft()

            _mod_ready, _missing = pp_module.ready()
            if not _mod_ready:
                logger.debug('{} not ready (missing {})'.format(
                    pp_module, ', '.join(_missing)
                ))
                temp_modules.append(pp_module)
                module_ran = True
                del pp_module
                continue
            if str(pp_module) in self.dataset._mod_exclusions:
                logger.info('Skipping {} (excluded)'.format(pp_module))
                module_ran = True
                del pp_module
                continue
            try:
                logger.info('Running {}'.format(pp_module))
                pp_module.process()
                pp_module.finalize()
                self.dataset.run_process_hooks(module_name=pp_module.__class__.__name__)
            except Exception as err: # pylint: disable=broad-except
                logger.error('Error in {}: {}'.format(pp_module, err))
                traceback.print_exc()
                self.failed_modules.append(pp_module)
            else:
                self.completed_modules.append(pp_module)

            module_ran = True
            while temp_modules:
                self.pp_modules.append(temp_modules.pop())

            self.dataset._collect_garbage()
            self.dataset._backend.decache()

        self.run_flagging()

        # Modify any attributes on inputs, canonically specified in flight
        # constants file.
        for var in self.dataset._backend.inputs:
            name = var.name
            if name in self.dataset._variable_mods:
                for key, value in self.dataset._variable_mods[name].items():
                    setattr(var, key, value)

        bounds = self.dataset.time_bounds()

        try:
            self.dataset._trim_data(start_cutoff=bounds[0], end_cutoff=bounds[1])
        except Exception as err:
            logger.error('Error trimming dataset', exc_info=True)

        self.dataset._finalize()

        self.dataset.run_process_hooks()


    def run_flagging(self):
        """
        Run flagging modules. Flagging modules provide data quality flags which
        cannot be defined at processing time, typically due to requiring data
        from some other processing module which can't be guaranteed to exist.
        """

        run_modules = []
        temp_modules = []
        module_ran = True
        self.flag_modules = [flag(self.dataset) for flag in ppodd.flags.flag_modules]
        flag_modules = collections.deque(self.flag_modules)

        while flag_modules and module_ran:
            module_ran = False
            flag_module = flag_modules.popleft()

            if not flag_module.ready(run_modules):
                logger.debug(f'{flag_module} not ready')
                temp_modules.append(flag_module)
                module_ran = True
                continue

            try:
                logger.info(f'Running {flag_module}')
                flag_module.flag()
            except Exception as err:
                logger.error(f'Error in {flag_module}', exc_info=True)

            run_modules.append(flag_module.__class__)

            module_ran = True
            while temp_modules:
                flag_modules.append(temp_modules.pop())
import os
import re
import importlib

from .base import FlaggingBase


def load_flagging_modules():
    pysearchre = re.compile('.py$', re.IGNORECASE)

    flag_files = filter(
        pysearchre.search,
        os.listdir(os.path.dirname(__file__))
    )

    flaggers = map(lambda f: '.' + os.path.splitext(f)[0], flag_files)

    importlib.import_module('ppodd.flags')

    modules = []
    for flagger in flaggers:
        if not flagger.startswith('_'):
            modules.append(
                importlib.import_module(flagger, package='ppodd.flags')
            )

    return modules


load_flagging_modules()
flag_modules = FlaggingBase.__subclasses__()

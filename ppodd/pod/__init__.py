import os
import re
import importlib
import sys

from .base import PPBase

def load_plugins():
    pysearchre = re.compile('.py$', re.IGNORECASE)

    pluginfiles = filter(
        pysearchre.search,
        os.listdir(os.path.dirname(__file__))
    )

    plugins = map(lambda f: '.' + os.path.splitext(f)[0], pluginfiles)

    importlib.import_module('ppodd.pod')

    modules = []
    for plugin in plugins:
        if not plugin.startswith('_'):
            modules.append(importlib.import_module(plugin, package="ppodd.pod"))

    return modules


def compile_cython():
    """
    Does a basic cython compile if required.
    """
    import pyximport
    pyximport.install()


compile_cython()
load_plugins()
pp_modules = PPBase.__subclasses__()

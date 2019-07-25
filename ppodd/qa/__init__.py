import os
import re
import importlib

from .base import QAMod


def load_qa_modules():
    pysearchre = re.compile('.py$', re.IGNORECASE)

    qa_files = filter(
        pysearchre.search,
        os.listdir(os.path.dirname(__file__))
    )

    qas = map(lambda f: '.' + os.path.splitext(f)[0], qa_files)

    importlib.import_module('ppodd.qa')

    modules = []
    for qa in qas:
        if not qa.startswith('_'):
            modules.append(importlib.import_module(qa, package='ppodd.qa'))

    return modules


load_qa_modules()
qa_modules = QAMod.__subclasses__()

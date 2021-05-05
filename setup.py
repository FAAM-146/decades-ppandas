from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.egg_info import egg_info
from ppodd import version, githash
import os


class Install(install):
    def run(self):
        write_githash()
        super().run()


class Develop(develop):
    def run(self):
        super().run()


class EggInfo(egg_info):
    def run(self):
        write_githash()
        super().run()


def write_githash():
    """
    Write the current git commit hash to a module for install.
    """
    with open(os.path.join('ppodd', 'githash_freeze.py'), 'w') as f:
        f.write(f'githash = "{githash()}"')


setup(
    name='ppodd',
    version=version(),
    description='DECADES-PPandas Postprocessing',
    author='FAAM',
    author_email='dave.sproson@faam.ac.uk',
    packages=find_packages(),
    package_data={'': ['*.yaml', '*.yml']},
    cmdclass={
        'install': Install,
        'develop': Develop,
        'egg_info': EggInfo
    }
)

import contextlib
import os
import inspect
import logging
import subprocess

import ppodd

__version__ = '0.15.1'
URL = 'https://github.com/faam-146/decades-ppandas'
DOI = '10.5281/zenodo.5711136'

formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)-8s %(name)s (%(funcName)s) - %(message)s'
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

package_logger = logging.getLogger(__name__.split('.')[0])
package_logger.setLevel(logging.INFO)
package_logger.addHandler(handler)

def version():
    return __version__

@contextlib.contextmanager
def flipdir(path):
    """
    Provides a quick-and-dirty context manager to quickly flip to a specified
    directory, do some stuff, and change back.

    Args:
        path: the path to change to.
    """
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield path
    except Exception:
        raise
    finally:
        os.chdir(cwd)

def githash():
    """
    Return the hash of the current git commit assuming that either

    The code is sitting in a git repo and the git cli is available on the
    system.

    or

    The code has been installed using included setup.py

    Returns:
        str: the hash of the current git commit, or None if this cannot be
            obtained.
    """
    path = os.path.dirname(inspect.getfile(ppodd))

    with flipdir(path) as _path:
        try:
            with open(os.devnull) as devnull:
                output = subprocess.check_output(
                    ['git', 'log', '-1', '--oneline'],
                    stderr=devnull
                )
        except subprocess.CalledProcessError:
            output = None

    if output:
        try:
            output = output.decode().split()[0]
        except (AttributeError, IndexError):
            return None

    if output is None:
        githash_path = os.path.join(
            os.path.dirname(ppodd.__file__), 'githash'
        )

        try:
            from ppodd.githash_freeze import githash as gh
            return gh
        except Exception:
            raise

    return output

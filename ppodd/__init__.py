import contextlib
import os
import inspect
import subprocess

import ppodd

__version__ = '0.9.2'
URL = 'https://github.com/faam-146/decades-ppandas'

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
    Return the hash of the current git commit assuming that
        a) the code is sitting in a git repo;
        b) the git cli is available on the system.

    Returns:
        output: the hash of the current git commit, or None if this cannot be
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

    return output

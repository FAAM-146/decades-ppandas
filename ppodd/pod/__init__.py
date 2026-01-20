import os
import re
import importlib
import logging
from types import ModuleType


logger = logging.getLogger(__name__)


def load_plugins() -> list[ModuleType]:
    """
    Find all plugin files in the pod directory or its subdirectories
    that match the pattern p_*.p, import them, and add them to a list of modules.
    """

    plugin_modules: list[ModuleType] = []
    pod_dir = os.path.dirname(__file__)
    pattern = re.compile(r"^p_.*\.py$")

    # Walk through the pod directory and its subdirectories
    for root, _, files in os.walk(pod_dir):
        for file in files:
            if pattern.match(file):
                # Construct the module name relative to the pod directory
                module_path = os.path.splitext(
                    os.path.relpath(os.path.join(root, file), pod_dir)
                )[0].replace(os.sep, ".")

                # Import the module
                full_module_name = f"ppodd.pod.{module_path}"
                try:
                    module = importlib.import_module(full_module_name)
                    logger.debug(f"Imported plugin module: {full_module_name}")
                    plugin_modules.append(module)
                except ImportError:
                    logger.warning(
                        f"Failed to import plugin module: {full_module_name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error occurred while importing module {full_module_name}: {e}"
                    )
    return plugin_modules


load_plugins()

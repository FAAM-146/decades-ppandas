import glob
import os

modules = {}

module_files = [
    f for f in glob.glob(os.path.join(os.path.dirname(__file__),'*.py'))
    if os.path.basename(f)[0] != '_'
]

print(module_files)

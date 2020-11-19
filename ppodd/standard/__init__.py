import os
import yaml

GLOBALS_FILE = os.path.join(os.path.dirname(__file__), 'globals.yaml')

with open(GLOBALS_FILE, 'r') as f:
    faam_globals = yaml.load(f, Loader=yaml.CLoader)

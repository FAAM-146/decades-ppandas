import os
import yaml

GLOBALS_FILE = os.path.join(os.path.dirname(__file__), 'globals.yaml')
ATTRS_FILE = os.path.join(os.path.dirname(__file__), 'variable_attrs.yaml')

with open(GLOBALS_FILE, 'r') as f:
    faam_globals = yaml.load(f, Loader=yaml.CLoader)

with open(ATTRS_FILE, 'r') as f:
    faam_attrs = yaml.load(f, Loader=yaml.CLoader)

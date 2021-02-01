import os
import yaml

GLOBALS_FILE = os.path.join(os.path.dirname(__file__), 'global_attrs.yaml')
ATTRS_FILE = os.path.join(os.path.dirname(__file__), 'variable_attrs.yaml')

with open(GLOBALS_FILE, 'r') as f:
    dataset_globals = yaml.load(f, Loader=yaml.CLoader)['Globals']

with open(ATTRS_FILE, 'r') as f:
    variable_attrs = yaml.load(f, Loader=yaml.CLoader)['VariableAttrs']

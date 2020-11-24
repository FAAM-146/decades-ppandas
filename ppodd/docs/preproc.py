import os
import shutil
import sys

from ppodd.standard import faam_globals
from ppodd.pod import pp_modules
from ppodd.decades.flags import (DecadesBitmaskFlag, DecadesClassicFlag)

BASE_DIR = 'base_rst'
DYN_DIR = 'dynamic_content'


core_base = os.path.join(BASE_DIR, 'coredata_base.rst')
modules_base = os.path.join(BASE_DIR, 'modules_base.rst')
core = os.path.join(DYN_DIR, 'coredata.rst')
modules = os.path.join(DYN_DIR, 'modules.rst')


shutil.copy2(core_base, core)
shutil.copy2(modules_base, modules)

def trim_docstr(docstring):
    """
    Handle indentation in docstrings. Taken directly from PEP-0257.
    """
    if not docstring:
        return ''
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return '\n'.join(trimmed)

def get_global_attrs_string(required=True):
    filtered_globals = sorted(
        [i for i in faam_globals['Globals'].items() if
        i[1]['required'] == required], key=lambda x: x[0]
    )

    globals_str = ''
    for attr, details in filtered_globals:
        versions = ', '.join([f'{i:0.1f}' for i in details['versions']])
        globals_str += f'* **{attr}** - {details["description"]}'
        globals_str += f' **Versions**: {versions}\n'
    return globals_str

def get_module_doc(module):
    m = module.test_instance()
    m.process()
    m.finalize()
    _doc = module.__doc__
    if _doc is None:
        _doc = 'No module documentation has been provided'

    txt = '-' * len(module.__name__) + '\n'
    txt += f'{module.__name__}\n'
    txt += '-' * len(module.__name__) + '\n\n'
    txt += trim_docstr(_doc.strip()) + '\n'

    txt += get_module_vardoc(m)
    txt += get_module_flagdoc(m)

    return txt

def get_module_vardoc(m):
    output = ''
    output += '\nOutputs\n'
    output += '-' * 7 + '\n\n'

    if not m.dataset.outputs:
        output += 'No outputs are declared by this module.\n'
        return output

    for out_var in m.dataset.outputs:
        output += (
            f'* **{out_var.name}** - {out_var.long_name} (``{out_var.units}``) '
            f'at {out_var.frequency} Hz\n'
        )

    output += '\n'
    return output


def get_module_flagdoc(m):
    _dict = {}
    for _var in m.dataset.outputs:
        var = str(_var)
        _dict.update(m.dataset[var].flag.descriptions)

    output = ''
    output += '\nFlags\n'
    output += '-' * 5 + '\n\n'

    if not m.dataset.outputs:
        output += 'No flags are defined in this module.\n\n'
        return output

    if type(m.dataset[var].flag) is DecadesBitmaskFlag:
        output += ('Variables in this module use bitmask flags. Some, all or '
                   'none of these flags may be applied to each variable. '
                   'Interrogate flag variable attributes ``flag_masks``'
                   ' and ``flag_meanings`` to find the flags for each '
                   'variable.\n')
    elif type(m.dataset[var].flag) is DecadesClassicFlag:
        output += ('Variables in this module use classic, value based '
                   'flagging.\n')

    if type(m.dataset[var].flag) is DecadesBitmaskFlag:
        if _dict:
            output += '\n'
            for flag, desc in _dict.items():
                if desc is None:
                    desc = 'No flag description provided.'
                output += '* ``' + flag + '`` - ' + desc + '\n'
            output += '\n'
        else:
            output += 'No flagging information provided.\n'

    elif type(m.dataset[var].flag) is DecadesClassicFlag:
        output += '\n'
        for var in m.dataset.outputs:
            output += f'* {var.name}\n'
            meanings = m.dataset[var.name].flag.cfattrs['flag_meanings'].split()
            values = m.dataset[var.name].flag.cfattrs['flag_values']
            for value, meaning in zip(values, meanings):
                output += '    * ``{}``: ``{}`` - {}\n'.format(
                    value, meaning, m.dataset[var.name].flag.descriptions[value]
                )
            output += '\n'

    return output




def replace_tag(path, tag, content):
    with open(path, 'r') as f:
        text = f.read()
    with open(path, 'w') as f:
        f.write(text.replace(tag, content))

def append(path, content):
    with open(path, 'a') as f:
        f.write(content)

replace_tag(
    core, 'TAG_REQUIRED_GLOBAL_ATTRIBUTES', get_global_attrs_string()
)

replace_tag(
    core, 'TAG_OPTIONAL_GLOBAL_ATTRIBUTES', get_global_attrs_string(False)
)

for mod in pp_modules:
    print(mod.__name__)
    append(modules, get_module_doc(mod))

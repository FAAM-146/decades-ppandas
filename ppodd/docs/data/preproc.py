import importlib
import os
import shutil
import sys

from ppodd.pod.base import pp_register
from ppodd.decades.flags import (DecadesBitmaskFlag, DecadesClassicFlag)

BASE_DIR = 'base_rst'
DYN_DIR = 'dynamic_content'


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


def get_global_attrs_string(required=True, standard_name=None,
                            standard_version=None):
    """
    Get a restructured text representation of global attributes as defined by a
    'standard'.

    Kwargs:
        required [True]: If True, return globals required by the standard,
            otherwise return optional global attributes.
        standard_name [None]: The name of the standard from which defines the
            globals.
        standard_version [None]: the version of <standard_name> to use.
    """

    if standard_name is None or standard_version is None:
        raise ValueError('Standard name and version must be given')

    dataset_globals = getattr(importlib.import_module(
        '.'.join(['ppodd.standard', standard_name])
    ), 'dataset_globals')

    filtered_globals = sorted(
        [i for i in dataset_globals.items() if
        i[1]['required'] == required
        and standard_version in i[1]['versions']],
        key=lambda x: x[0]
    )

    globals_str = ''
    for attr, details in filtered_globals:
        versions = ', '.join([f'``{i:0.1f}``' for i in details['versions']])
        globals_str += f'* ``{attr}`` - {details["description"]}'
        globals_str += f' *Versions*: {versions}\n'

    return globals_str


def get_variable_attrs_string(required=True, standard_name=None,
                              standard_version=None):
    """
    Get a restructured text representation of variable attributes as defined by
    a 'standard'.

    Kwargs:
        required [True]: If True, return globals required by the standard,
            otherwise return optional variable attributes.
        standard_name [None]: The name of the standard from which defines the
            variable attributes.
        standard_version [None]: the version of <standard_name> to use.
    """

    if standard_name is None or standard_version is None:
        raise ValueError('Standard name and version must be given')

    variable_attrs = getattr(importlib.import_module(
        '.'.join(['ppodd.standard', standard_name])
    ), 'variable_attrs')


    filtered_attrs = sorted(
        [i for i in variable_attrs.items() if
        i[1]['required'] == required
        and standard_version in i[1]['versions']],
        key=lambda x: x[0]
    )

    attrs_str = ''
    for attr, details in filtered_attrs:
        versions = ', '.join([f'``{i:0.1f}``' for i in details['versions']])
        attrs_str += f'* ``{attr}`` - {details["description"]}'
        attrs_str += f' *Versions*: {versions}\n'

    return attrs_str


def get_module_doc(module):
    """
    Returns the module documentation for a given postprocessing module, as
    restructured text.

    Args:
        module: The post processing module (probably a subclass of
            ppodd.pod.Base) to document

    Returns:
        A restructured text string containing the module documentation.
    """

    m = module.test_instance()
    m.process()
    m.finalize()
    _doc = module.__doc__
    if _doc is None:
        _doc = 'No module documentation has been provided'

    txt = '.. _{}:\n\n'.format(module.__name__)
    txt += '-' * len(module.__name__) + '\n'
    txt += f'{module.__name__}\n'
    txt += '-' * len(module.__name__) + '\n\n'
    txt += trim_docstr(_doc.strip()) + '\n'

    txt += get_module_vardoc(m)
    txt += get_module_flagdoc(m)

    return txt

def get_module_vardoc(module):
    """
    Returns the module variable documentation for a given postprocessing module,
    as restructured text.

    Args:
        module: The post processing module (probably a subclass of
            ppodd.pod.Base) to document

    Returns:
        A restructured text string containing the module variable documentation.
    """

    output = ''
    output += '\nOutputs\n'
    output += '-' * 7 + '\n\n'

    if not module.dataset.outputs:
        output += 'No outputs are declared by this module.\n'
        return output

    for out_var in module.dataset.outputs:
        out_var.attrs.set_compliance_mode(True)
        output += f'* ``{out_var.name}``\n'
        for attr in out_var.attrs().items():
            output += f'    * ``{attr[0]}``: {attr[1]}\n'

    output += '\n'
    return output


def get_module_flagdoc(module):
    """
    Returns the module flagging documentation for a given postprocessing module,
    as restructured text.

    Args:
        module: The post processing module (probably a subclass of
            ppodd.pod.Base) to document

    Returns:
        A restructured text string containing the module flagging documentation.
    """

    _dict = {}
    for _var in module.dataset.outputs:
        var = str(_var)
        _dict.update(module.dataset[var].flag.descriptions)

    output = ''
    output += '\nFlags\n'
    output += '-' * 5 + '\n\n'

    if not module.dataset.outputs:
        output += 'No flags are defined in this module.\n\n'
        return output

    if type(module.dataset[var].flag) is DecadesBitmaskFlag:
        output += ('Variables in this module use bitmask flags. Some, all or '
                   'none of these flags may be applied to each variable. '
                   'Interrogate flag variable attributes ``flag_masks``'
                   ' and ``flag_meanings`` to find the flags for each '
                   'variable.\n')
    elif type(module.dataset[var].flag) is DecadesClassicFlag:
        output += ('Variables in this module use classic, value based '
                   'flagging.\n')

    if type(module.dataset[var].flag) is DecadesBitmaskFlag:
        if _dict:
            output += '\n'
            for flag, desc in _dict.items():
                if desc is None:
                    desc = 'No flag description provided.'
                output += '* ``' + flag + '`` - ' + desc + '\n'
            output += '\n'
        else:
            output += 'No flagging information provided.\n'

    elif type(module.dataset[var].flag) is DecadesClassicFlag:
        output += '\n'
        for var in module.dataset.outputs:
            output += f'* {var.name}_FLAG\n'
            meanings = module.dataset[var.name].flag.cfattrs['flag_meanings']
            meanings = meanings.split()

            values = module.dataset[var.name].flag.cfattrs['flag_values']
            for value, meaning in zip(values, meanings):
                output += '    * ``{}``: ``{}`` - {}\n'.format(
                    value, meaning,
                    module.dataset[var.name].flag.descriptions[value]
                )
            output += '\n'

    return output

def get_variables_tex(default=True):
    """
    Return a list of variables, in rst but suitable for compilation to pdf via
    latex. The latex/pdf version omits standard name from the variable table
    due to a lack of space in portrait A4.

    Kwargs:
        default [True]: If True, return only variable whch are written by
        default, otherwise return variables not written by default.

    Returns:
        An RST table of variables.
    """
    mods = [i.test_instance() for i in pp_modules]
    output = '.. csv-table:: Default Variables\n'
    output += '    :header: "Name", "Long Name", "Processing Module"\n'
    output += '    :widths: 30, 40, 30\n\n'
    _vars = []
    for m in mods:
        for name, dec in m.declarations.items():
            long_name = dec['long_name']
            mod_link = ':ref:`{0}`'.format(m.__class__.__name__)
            if default:
                if 'write' not in dec.keys() or dec['write']:
                    _vars.append((name, long_name, mod_link))
            else:
                if 'write' in dec.keys() and not dec['write']:
                    _vars.append((name, long_name, mod_link))

    for var in sorted(_vars, key=lambda x: x[0]):
        output += f'    "{var[0]}", "{var[1]}", "{var[2]}"\n'
    return output


def get_dimensions():
    """
    Get a list of dimensions produced by the processing module group.
    """

    mods = []
    for i in pp_modules:
        mods.append(i.test_instance())
    dimstr = ''
    _dims = set()
    for m in mods:
        for name, dec in m.declarations.items():
            _freq = dec['frequency']
            if _freq == 1:
                continue
            _dims.add((f'sps{_freq:02d}', _freq))

    _dims = sorted(list(_dims))
    for dim in _dims:
        dimstr += (f'* ``{dim[0]}`` - {dim[1]} samples per second. '
                   f'A dimension of length {dim[1]}.\n')

    return dimstr


def get_variables_web(default=True):
    """
    Return a list of variables, in rst but suitable for compilation to html.
    The html version includes standard name in the variable table.

    Kwargs:
        default [True]: If True, return only variable whch are written by
        default, otherwise return variables not written by default.

    Returns:
        An RST table of variables.
    """

    mods = [i.test_instance() for i in pp_modules]
    output = '.. csv-table:: Default Variables\n'
    output += '    :header: "Name", "Long Name", "Standard Name", "Processing Module"\n'
    output += '    :widths: 15, 40, 15, 30\n\n'
    _vars = []
    for m in mods:
        for name, dec in m.declarations.items():
            long_name = dec['long_name']
            mod_link = ':ref:`{0}`'.format(m.__class__.__name__)
            try:
                standard_name = dec['standard_name']
            except KeyError:
                standard_name = ''
            if standard_name is None:
                standard_name = ''
            if default:
                if 'write' not in dec.keys() or dec['write']:
                    _vars.append((name, long_name, standard_name, mod_link))
            else:
                if 'write' in dec.keys() and not dec['write']:
                    _vars.append((name, long_name, standard_name, mod_link))

    for var in sorted(_vars, key=lambda x: x[0]):
        output += f'    "{var[0]}", "{var[1]}", "{var[2]}", "{var[3]}"\n'
    return output

def replace_tag(path, tag, content):
    """
    Replace a given placeholder tag in a file.

    Args:
        path: the path to the file in which a tag should be replaced.
        tag: the placeholder tag to replace.
        content: the content to substitute fot <tag>
    """

    with open(path, 'r') as f:
        text = f.read()
    with open(path, 'w') as f:
        f.write(text.replace(tag, content))

def append(path, content):
    """
    Append some content to a given file.

    Args:
        path: the path of the file to append to.
        content: the content to append to the file at <path>.
    """

    with open(path, 'a') as f:
        f.write(content)


if __name__ == '__main__':

    # Get copies of documentation templates to populate.
    core_base = os.path.join(BASE_DIR, 'coredata_base.rst')
    modules_base = os.path.join(BASE_DIR, 'modules_base.rst')
    core = os.path.join(DYN_DIR, 'coredata.rst')
    modules = os.path.join(DYN_DIR, 'modules.rst')

    shutil.copy2(core_base, core)
    shutil.copy2(modules_base, modules)

    # We're only supporting compilation to pdf via latex, or html currently
    if 'latex' in sys.argv[1]:
        get_variables = get_variables_tex
    else:
        get_variables = get_variables_web

    # The standard to use is specified in the environment variable
    # PPODD_STANDARD, which should be of the format
    # <standard_name>@<standard_version>, for example PPODD_STANDARD=core@1.0
    standard = os.environ['PPODD_STANDARD']
    standard_name, standard_version = standard.split('@')
    standard_version = float(standard_version)

    # The module group to document should be specified in the environment group
    # PP_GROUP, for example PP_GROUP=core
    module_group = os.environ['PP_GROUP']
    pp_modules = pp_register[module_group]

    replace_tag(
        core, 'TAG_REQUIRED_GLOBAL_ATTRIBUTES',
        get_global_attrs_string(True, standard_name=standard_name,
                                standard_version=standard_version)
    )

    replace_tag(
        core, 'TAG_OPTIONAL_GLOBAL_ATTRIBUTES',
        get_global_attrs_string(False, standard_name=standard_name,
                                standard_version=standard_version)
    )

    replace_tag(
        core, 'TAG_REQUIRED_VARIABLE_ATTRIBUTES',
        get_variable_attrs_string(True, standard_name=standard_name,
                                  standard_version=standard_version)
    )

    replace_tag(
        core, 'TAG_OPTIONAL_VARIABLE_ATTRIBUTES',
        get_variable_attrs_string(False, standard_name=standard_name,
                                  standard_version=standard_version)
    )

    replace_tag(core, 'TAG_SPS_DIMENSIONS', get_dimensions())

    replace_tag(core, 'TAG_DEFAULT_VARIABLES', get_variables())
    replace_tag(core, 'TAG_OPTIONAL_VARIABLES', get_variables(False))

    for mod in pp_modules:
        append(modules, get_module_doc(mod))
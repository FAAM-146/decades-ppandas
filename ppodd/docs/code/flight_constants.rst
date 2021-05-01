=========================
The Flight Constants File
=========================

The flight constants file is a ``yaml`` file which provides metadata, constants used in the
processing, and directives as to what the processing should do. Its basic format is

.. code-block:: yaml
    :linenos:

    Globals:
        global_key_1: global_value_1
        global_key_2: global_value_2
        global_group:
            key_1: value_1
            key_2: value_2
    
    Modifications:
        Variables:
            VAR_1:
                write: true
                comment: comment about variable
            VAR_2:
                write: false
        Exclude:
            - ModuleToExclude
            - OtherModuleToExclude

    Constants:
        group1:
            array_key: [1, 2, 3]
            number_key: 7.3
            string_key: 'STRING'
        group2:
            array_key2: [3, 2, 1]

The constants file is split into three main sections: ``Globals`` (lines 1-6),
``Modifications`` (lines 8-16), and ``Constants`` (lines 18-24).

Globals
=======

The ``Globals`` group is responsible for providing global (dataset) level metadata.
In the context of writing to a netCDF file, these correspond to global attributes;
other output formats may deal with these differently.

Grouping Globals
----------------

Globals are generally a simple ``key: value`` pair, which map directly to attributes.
However, for convenience, it is also possible to collect globals into groups which
will be flattened by joining with an underscore when read. For example, the globals

.. code-block:: yaml

    Globals:
        processing_software:
            url: https://github.com/FAAM-146/decades-ppandas
            version: 1.0

is equivalent to

.. code-block:: yaml

    Globals:
        processing_software_url: https://github.com/FAAM-146/decades-ppandas
        processing_software_version: 1.0


String Interpolation
--------------------

Basic string interpolation is allowed within the globals group, using standard
pythonic curly-brace syntax, so that, for example,

.. code-block:: yaml

    Globals:
        flight_number: c123
        title: Data from {flight_number}

will resolve to

.. code-block:: yaml

    Globals:
        flight_number: c123
        title: Data from c123


Modifications
=============

The ``Modifications`` group is split into two subgroups: ``Variables`` and
``Exclude``. 

Variables
---------

The ``Variables`` group allows modification to variable attributes after all processing modules have run. In the example above, the variable ``VAR_1`` will have its ``write`` attribute set to ``true``, indicating that it should be written to
file, and its ``comment`` attribute set to 'comment about variable'. The variable
``VAR_2`` will have its ``write`` attribute set to ``false``, indicating that it
should not be written to file.

Exclude
-------

The ``Exclude`` group is simply a list of processing modules which should not be
automatically run during the processing, even if the data they require to run is
available. Each entry should be the class name of the module to exclude, not
including the classpath.

Constants
=========

The constants group provide data that are required during processing, for example
calibraion coefficients, information about the instruments fitted or switches to
control flow within processing modules.

Constants should be divided into groups, however this is purely for convenience, and
the groups will be ignored when the constants are read. Therefore constant keys must
be globally unique.

The keys of constants must be strings, however the values can be any native data
type - numeric, string, array, or map.

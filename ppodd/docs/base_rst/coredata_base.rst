==================
The Core Data File
==================

--------------------
Filename conventions
--------------------

The core file is named as ``core_faam_<date>_<version>_<revision>_<fltnum>[_<freq>].nc``, where:

* **date** indicates the date on which the flight took place, formatted as ``YYYYmmdd``, where ``YYYY`` is the four-digit year, ``mm`` is the two-digit month, and ``dd`` is the two-digit day.
* **version** indicates the version of the core file, formatted as v\ *nnn*, where *nnn* is a three-digit integer. The most recent version is 5 (*v005*), which this document applies to. It is important to note that older versions do not strictly follow the conventions outlined in this document.
* **revision** indicates the revision number of the data, formatted as r\ *n*, where *n* is an integer. Larger values of *n* indicate a more recent release. Where more than one data revision is available, the file with the largest revision number should be used.
* **fltnum** indicates the flight number of the flight, formatted as *xnnn* where *x* is a lowercase letter, and *nnn* a three digit integer.
* **freq** indicates the frequency of the dataset, formatted as *n*\ hz, where n is an integer. If not present, the file is at 'full' frequency, with the frequency differing between variables. Typically 'full' and 1hz files are provided.

----------
Dimensions
----------

All variables in the core dataset are timeseries with the same start and end times, and with frequencies of 1 Hz or greater. 
There is only one time dimension, ``Time``, with a length equal to the duration of the dataset in seconds. 
Variables which are recorded at 1 Hz will have ``Time`` as their only dimension. 
Variables which are recorded at higher than 1 Hz will be stored as a two-dimensional array, with ``Time`` as the first dimension and ``spsNN`` as the second.
Here, the ``NN`` repesents the frequency in Hz.
For example, a 4 Hz variable would have dimensions ``Time`` and ``sps04``, and would be stored as an :math:`n\times4`, where `n` is the length of the dataset in seconds.

The dimensions which may be included in the full temporal resolution core data file are:

* ``Time`` - An unlimited dimension whose length corresponds to the length of the dataset, in seconds.
TAG_SPS_DIMENSIONS

-----------------
Global Attributes
-----------------

Required Global Attributes
--------------------------

TAG_REQUIRED_GLOBAL_ATTRIBUTES

Optional Global Attribtues
--------------------------

TAG_OPTIONAL_GLOBAL_ATTRIBUTES

-------------------
Variable Attributes
-------------------

Required Variable Attributes
----------------------------

TAG_REQUIRED_VARIABLE_ATTRIBUTES

Optional Variable Attributes
----------------------------

TAG_OPTIONAL_VARIABLE_ATTRIBUTES

---------
Variables
---------

Below is a list of all output variables which can be created during postprocessing, split into variables which will always be written to file if available, and those which are generally only used internally, but which can be written to file if requested.

Written if Available
--------------------

TAG_DEFAULT_VARIABLES

Used internally
-----------------------

TAG_OPTIONAL_VARIABLES

-----
Flags
-----

Every variable has an associated flag variable named ``<variable_name>_FLAG``, and referenced in the variable's ``ancillary_variables`` attribute, which provides some quality or situational information about the data in the variable.
There are two different flagging strategies used in the core data file; the first is a value-based quality flag, referred to as a value-based or classic flag, and the second is a packed boolean representation.
Flagging is largely automatic, other than the ``flagged_in_qc`` flag meaning, which indicates that in the opinion of the person performing the quality control, that data should be treated with caution.
It is up to the user to decide whether or not to use data which has been flagged. If in doubt, users should contact FAAM for advice.

Value-based Flags
-----------------

Value-based flags represent the quality of the corresponding data variable, with a flag value of 0 representing data which are presumed to be of sufficient quality. 
Larger values of the flag generally correspond to lower quality data, though this isn't always the case.

For example, consider a variable array with values

``1 2 6 5 4 3 6 5 4 3 2 1 4 5 6 7 7 6 5 3 2``

and its corresponding flag with values

``0 0 0 0 1 1 0 1 1 1 2 2 1 0 0 0 0 0 0 0 0``

To understand the meaning of these flag values, we can look at the ``flag_values`` and ``flag_meanings`` attributes of the flag variable, which may look like

* ``flag_meanings``: ``data_good minor_data_quality_issue major_data_quality_issue``
* ``flag_values``: ``0b 1b 2b``

We can see that there are three different flag meanings and three different flag values, and can deduce that a flag value of 0 indicates the data are considered good, a flag value of 1 indicates a minor data quality issue, and a flag value of 2 indicates a major data quality issue.
A user may choose, for example, to eliminate data with a major quality issue. To do this, they would simply mask/nan the variable whereever the flag variable has a value of 2, leaving the variable array as

``1 2 6 5 4 3 6 5 4 3 - - 4 5 6 7 7 6 5 3 2``.

Value-based flags will have the same dimensions as their associated variable, and the following variable attributes:

* ``_FillValue``: -128b
* ``standard_name``: 'status_flag' if the associated variable has no standard name, otherwise '<variable_standard_name> status_flag'
* ``long_name``: Flag for <variable_name>
* ``flag_values``: An array of the values that the flag variable can take. Typically runs from 0 to <length of flag_meanings> - 1.
* ``flag_meanings``: A space separated string of the meanings of each of the values in flag_values.

Bitmask flags
-------------

While the value-based flags map the values of an array to a single meaning, bitmask flags allow the representation of a boolean array for every ``flag_meaning``.
This is done by mapping each flag meaning to an increasing power of 2, which allows the representation of every possible state of every meaning using values from 1 to :math:`2^{\text{num. flags}-1}`.
A value of 0 indicates that no flags are set, and is set as a fill value.
In order to a bitmask flag it must first be unpacked. This adds to the complexity of using the flag, but makes flags much more powerful, so most variables in the FAAM core data product use bitmask flags.

For example, consider a variable array with values

``1 2 6 5 4 3 6 5 4 3 2 1 4 5 6 7 7 6 5 3 2``

and its corresponding flag with values

``1 1 3 3 2 2 4 4 4 4 6 6 6 6 8 8 5 5 3 3 1``

To understand the meaning of these flag values, we can look at the ``flag_masks`` and ``flag_meanings`` attributes of the flag variable, which may look like

* ``flag_meanings``: ``aircraft_on_ground flow_out_of_range temp_out_of_range data_out_of_bounds``
* ``flag_masks``: ``1b 2b 4b 8b``

There are four meanings, with each associated with a value of :math:`2^n` with :math:`n` taking the four values 0, 1, 2, 3. In the FAAM core data, flag values are guaranteed to be increasing powers of 2, thus the flag array can be unpacked simply by progressively right-bitshifting the flag array, and taking the result modulo 2.
In python, this can be achieved with the following code:

.. code::

    # Note that we don't need to worry about using the flag_masks attribute, as it
    # is guaranteed to be powers of 2 from 1 to 2^(n-1)
    unpacked = {}
    for i, meaning in enumerate(flag_var.flag_meanings.split()):
        unpacked[meaning] = (a >> i) % 2

this would leave us with the following in ``unpacked``:

.. code::

    {
        aircraft_on_ground: array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        flow_out_of_range: array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]),
        temp_out_of_range: array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0]),
        data_out_of_bounds: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
    }

Bitmask flags will have the same dimensions as their associated variable, and the following variable attributes:

* ``_FillValue``: 0b
* ``standard_name``: 'status_flag' if the associated variable has no standard name, otherwise '<variable_standard_name> status_flag'
* ``long_name``: Flag for <variable_name>
* ``valid_range``: The valid range of values in the flag variable array. Should be 1b, 2^(<number of flag_meanings>) - 1
* ``flag_masks``: An array of the values that the flag variable can take, which will runs from 1 to 2^(<number of flag_meanings> - 1).
* ``flag_meanings``: A space separated string of the meanings of each of the values in flag_values.

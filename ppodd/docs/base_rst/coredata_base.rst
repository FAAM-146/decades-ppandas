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

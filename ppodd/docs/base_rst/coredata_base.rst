==================
The Core Data File
==================

--------------------
Filename conventions
--------------------

The core file is named as ``core_faam_<date>_<version>_<revision>_<fltnum>[_<freq>].nc``, where:

* **date** indicates the date on which the flight took place, formatted as ``YYYYmmdd``, where ``YYYY`` is the four-digit year, ``mm`` is the two-digit month, and `dd` is the two-digit day.
* **version** indicates the version of the core file, formatted as v\ *nnn*, where *nnn* is a three-digit integer. The most recent version is 5 (*v005*), which this document applies to. It is important to note that older versions do not strictly follow the conventions outlined in this document.
* **revision** indicates the revision number of the data, formatted as r\ *n*, where *n* is an integer. Larger values of *n* indicate a more recent release. Where more than one data revision is available, the file with the largest revision number should be used.
* **fltnum** indicates the flight number of the flight, formatted as *xnnn* where *x* is a lowercase letter, and *nnn* a three digit integer.
* **freq** indicates the frequency of the dataset, formatted as *n*\ hz, where n is an integer. If not present, the file is at 'full' frequency, with the frequency differing between variables. Typically 'full' and 1hz files are provided.

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

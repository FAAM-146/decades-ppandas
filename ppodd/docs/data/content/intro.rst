============
Introduction
============

-------
Summary
-------

This document provides a description of the FAAM core data product, including a summary of the processing that takes place and a standard which describes the format of, and metadata associated with, the core data product.

---------------
FAIR Principles
---------------

Data provided by FAAM are intended to follow the FAIR principles, introduced by Wilkinson et al., 2016 (http://dx.doi.org/10.1038/sdata.2016.18). FAIR means that the data are **F**\ indable, **A**\ cessible, **I**\ nteroperable and **R**\ eusable. A full description of the FAIR principles is available at https://go-fair.org/fair-principles/. For convenience, these are reproduced in part below.

Findable
--------

The first step in (re)using data is to find them. Metadata and data should be easy to find for both humans and computers. Machine-readable metadata are essential for automatic discovery of datasets and services, so this is an essential component of the FAIRification process.

* (Meta)data should be assigned a globally unique and persistent identifier.
* Data should be described with rich metadata.
* Metadata should clearly and explicitly include the identifier of the data they describe.
* (Meta)data are registered or indexed in a searchable resource.

Accessible
----------

Once the user finds the required data, she/he needs to know how can they be accessed, possibly including authentication and authorisation.

* (Meta)data should be retreivable by their identifier using a standardised communications protocol
    * The protocol should be open, free, and universally implementable.
    * The protocol should allow for an authentication and and authorisation procedure, where nesessary.
* Metadata should be accessible, even when the data are no longer available.

Interoperable
-------------

The data usually need to be integrated with other data. In addition, the data need to interoperate with applications or workflows for analysis, storage, and processing.

* (Meta)data should use a formal, accessible, shared, and broadly applicable language for knowledge representation.
* (Meta)data should use vocabularies that follow FAIR principles.
* (Meta)data should include qualified references to other (meta)data.

Reusable
--------

The ultimate goal of FAIR is to optimise the reuse of data. To achieve this, metadata and data should be well-described so that they can be replicated and/or combined in different settings.

* (Meta)data should be richly described with a plurality of accurate and relevant attributes.
    * (Meta)data should be released with a clear and accessible data usage license.
    * (Meta)data should be associated with detailed provenance.
    * (Meta)data should meet domain-relevant community standards.

------
NetCDF
------

The FAAM core data product is provided in the **netCDF** format. NetCDF (Network Common Data Form) is a set of software libraries and platform independent data formats which are designed to support the creation, access, and sharing of array-oriented scientific data. NetCDF is designed to be

* **Self-describing.** A netCDF file include information about the data it contains (i.e. metadata).
* **Portable.** A netCDF file can be accessed by computers with different ways of storing integers, characters, and floating-point numbers.
* **Scalable.** A small subset of a large dataset may be accessed efficiently.
* **Appendable.** Data may be appended to a properly structured netCDF file without copying the dataset or redefining its structure.
* **Shareable.** One writer and many readers may simultaneously access the same netCDF file.
* **Archivable.** Access to all earlier forms of netCDF data will be supported by current and future versions of the software.

NetCDF is extremely commonly used, and is almost ubiquitous within the Earth sciences. Unidata (https://unidata.ucar.edu) provide and maintain software libraries for accessing netCDF data using C, C++, Java, and FORTRAN. Third-party libraries (which are generally bindings or wrappers to the Unidata libraries) are available for Python, IDL, MATLAB, R, Ruby, and Perl, among others.

NetCDF files generally consist of four components:

* **Attributes.** Attributes are metadata which can be attached to the netCDF file itself (called global attributes), to variables, and to groups (variable attributes and groups attributes, respectively). Attributes may be textual or numeric; numeric attributes may be arrays.
* **Groups.** Groups (available since netCDF4) provide a method to encapsulate related *dimensions,* *variables,* and *attributes.* They can be thought of as somewhat analogous to directories in a filesystem.
* **Dimensions.** Dimensions specify the size of a single axis of a variable within a netCDF file. Common dimensions for geophysical data include time, latitude, and longitude, though they do not need to correspond to physical dimensions. There is no practical limit to the number of dimensions which may be defined in a netCDF file.
* **Variables.** Variables are named *n*\ -dimensional (thus associated with *n* *dimensions*) arrays of a specified data type. Variables may have zero or more *attributes*, which act as metadata to describe the contents of the variable.

A minimal example of accessing a 1-dimensional variable, *data*, along with its *units* attribute and a global *title* attribute from a netCDF file, using python, is given below. Note that the netCDF library, *netCDF4*, is not included as part of the python standard library, but may be installed using your system package manager, pip, or conda.

.. code-block:: python
    :linenos:

    from netCDF4 import Dataset

    with Dataset('somefile.nc', 'r') as nc:
        title = nc.title
        data_units = nc['data'].units
        data_data = nc['data'][:]

Python software libraries to aid in accessing FAAM data are in development, and will be made available in due course.

------------------------
Data access and archival
------------------------

FAAM aim to process and make available a preliminary version of the core data product within 24 hours of a flight, although this may take slightly longer when on detachment. 
The preliminary file, indicated by the postfix ``_prelim`` in the filename will initially be made available to registered users through the FAAM website, where it will also be available in an interactive visualisation tool.
The preliminary file is intended to be used for visualisation and initial analysis. 

Once all of the variables in this file have been checked by a FAAM staff member, the data will be archived at the Centre for Environmental Data Analysis (CEDA; https://www.ceda.ac.uk).
The archived verison will not include the ``_prelim`` postfix, and having gone through QC, may differ from the preliminary file.
Users can access the data by first registering as a CEDA user, and then applying for access to FAAM core data. The core data file is generally freely available, however access may be restricted for upto one year at the request of a project PI.

Usage License
-------------

FAAM data are licensed under the Open Government Licence (http://www.nationalarchives.gov.uk/doc/open-government-licence).

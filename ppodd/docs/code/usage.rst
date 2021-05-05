=====
Usage
=====

Basic Usage
-----------

A very basic workflow with ``decades-ppandas`` might look like

.. code-block:: python
    :linenos:

    from ppodd.decades import DecadesDataset

    d = DecadesDataset()
    d.add_glob('*yaml')
    d.add_glob('*zip')
    d.load()
    d.process()
    d.write()

* *Line 1*: Import the :obj:`ppodd.decades.DecadesDataset` class from the module ``ppodd.decades``. This is the primary class used for interacting with ``decades-ppandas``.

* *Line 3*: Create a new ``DecadesDataset`` instance. If you'll be loading a flight constants file, no arguments are required. If not loading a flight constants file, for example during initial data exploration, you should pass the flight date to the constructor: :code:`d = DecadesDataset(datetime.datetime(2000, 1, 1))`.

* *Lines 4-5*: Use the :code:`add_glob()` method to add any yaml or zip files in the current directory to the ``DecadesDataset``. Yaml files are typically flight constant files, of which only one should be added to any dataset. Zip files are extracted before being used.

* *Line 6*: Load the data which have been added into the dataset.

* *Line 7*: Run all of the processing modules in the current processing group which can be run given the data added to the dataset. As no processing group was specified when creating the dataset, the default `core` processing group will be run.

* *Line 8*: Write the processed data to file. As no options pertaining to output were specified when creating the dataset, the :code:`write()` method will write a netCDF file with a standard name, with variables written at the maximum available frequency. A string may be passed to this method to write to a different filename, and the keyword argument ``freq=1`` may be passed to write a 1 Hz file. For example :code:`d.write('myfile.nc', freq=1)` will create a 1 Hz file named `myfile.nc`.

A More Advanced Usage Example
-----------------------------

In this example we will load up some raw data from the core console DLU, visualize it, and write it 
to file without running any processing modules. Initially the work flow is similar, adding data to
a dataset and loading it:

.. code-block:: python
    :linenos:

    from datetime import datetime
    from ppodd.decades import DecadesDataset

    d = DecadesDataset(datetime(2021, 4, 28))
    d.add_glob('CORCON*')
    d.load()

.. note::

    Note that a ``datetime`` has been passed to the ``DecadesDataset`` constructor. This is because
    no flight constants file containing a flight date is being loaded. Some raw data types may use
    relative rather than absolute timestamps, so the base date is needed in advance of loading data.

.. note::

    Loading raw DLU data requires a definition (csv) file. Assuming that such a definition is named
    as expected and in the current working directory, it will be picked up with the same glob pattern
    as the data (``CORCON*``).

Once the data are loaded, all of the available variables can be seen in the ``variables`` attribute of the dataset:

.. code-block:: python

    print(d.variables)

    ['CORCON_packet_length', 'CORCON_ptp_sync', ..., 'CORCON_padding7']

Variables can be accessed with a dict-like interface on the dataset, for example, 

.. code-block:: python

    dit_counts = d['CORCON_di_temp']

``dit_counts`` here will be of type :obj:`ppodd.decades.DecadesVariable`. The data will be stored in
the ``array`` attribute of attribute of the variable:

.. code-block:: python

    print(dit_counts.array)
    array([6143132., 6143239., 6143222., ..., 6140500., 6140543., 6140574.])

Calling the variable will return a pandas ``Series`` object, which encapsulates both the data and
its corresponding timestamps:

.. code-block:: python

    print(dit_counts())
    2020-02-11 07:29:22.000000    6143132.0
    2020-02-11 07:29:22.031250    6143239.0
    2020-02-11 07:29:22.062500    6143222.0
    2020-02-11 07:29:22.093750    6143101.0
    2020-02-11 07:29:22.125000    6143209.0
                                    ...
    2020-02-11 13:59:25.843750    6140555.0
    2020-02-11 13:59:25.875000    6140549.0
    2020-02-11 13:59:25.906250    6140500.0
    2020-02-11 13:59:25.937500    6140543.0
    2020-02-11 13:59:25.968750    6140574.0
    Freq: 31250000N, Name: CORCON_di_temp, Length: 748928, dtype: float64

This makes it very easy to quickly visualise data. For example, the following code will produce a
plot of deiced and nondeiced temperature counts:

.. code-block:: python

    import matplotlib.pyplot as plt
    plt.plot(d['CORCON_di_temp']())
    plt.plot(d['CORCON_ndi_temp']())
    plt.show()

.. note::

    It's required to call the variable to return a Series, as the full index is not stored in the
    variable. Rather, to save on memory usage, only a start time, end time, and frequency are 
    stored, and the index is build on-the-fly when the variable is called.

By default input data will not be written to file, while output data (i.e. variables produced in
processing modules) will be. In order to write input variables to file, we need to set the 
``write`` property of the variable to ``True``:

.. code-block:: python

    d['CORCON_di_temp'].write = True
    d['CORCON_ndi_temp'].write = True

We can then just call the ``write`` method of the dataset to write these raw data to a netCDF
file:

.. code-block:: python

    d.write('raw_temperatures.nc')

This will produce a simple netCDF file, with the following structure (as produced by ``ncdump -h``):

.. code-block:: none

    netcdf raw_temperatures {
    dimensions:
    	Time = UNLIMITED ; // (23404 currently)
    	sps32 = 32 ;
    variables:
    	int Time(Time) ;
    		Time:long_name = "Time of measurement" ;
    		Time:standard_name = "time" ;
    		Time:calendar = "gregorian" ;
    		Time:units = "seconds since 2020-02-11 00:00:00 +0000" ;
    	float CORCON_ndi_temp(Time, sps32) ;
    		CORCON_ndi_temp:_FillValue = -9999.f ;
    		CORCON_ndi_temp:frequency = 32 ;
    		CORCON_ndi_temp:long_name = "32Hz raw readings (in counts) taken from the NON de-iced temperture transducer" ;
    		CORCON_ndi_temp:units = "RAW" ;
    		CORCON_ndi_temp:valid_max = 6143470.f ;
    		CORCON_ndi_temp:valid_min = 6136480.f ;
    	float CORCON_di_temp(Time, sps32) ;
    		CORCON_di_temp:_FillValue = -9999.f ;
    		CORCON_di_temp:frequency = 32 ;
    		CORCON_di_temp:long_name = "32Hz raw readings (in counts) taken from the de-iced temperture transducer" ;
    		CORCON_di_temp:units = "RAW" ;
    		CORCON_di_temp:valid_max = 6888176.f ;
    		CORCON_di_temp:valid_min = 6036374.f ;
    }

Creating QC plots
-----------------

Quicklook Quality Control (QC) plots can be produced as part of the data processing workflow, using the 
``run_qa()`` method of the ``DecadesDataset``, after processing. For example

.. code-block:: python

    d.process()
    d.run_qa()

This will run all of the QC modules which can be run given the available data, each of which will
produce a pdf file. This will either be saved in the current working directory, or in the directory
specified by setting the ``qa_dir`` property of the ``DecadesDataset``. For example,

.. code-block:: python

    d.process()
    d.qa_dir = '/some/custom/dir'
    d.run_qa()

will save all of the QC figures in the directory ``/some/custom/dir``.

Creating a Flight Report
------------------------

Flight Reports can be produced during the processing process. They include crew lists, timings,
chat logs, QA figures, and Figures from the flight folder.

.. note::

    Flight reports typically include an overview figure showing the synoptic situation during
    the flight, derived from GFS data. Acquiring these data requires an external dependency which
    is not installed in the conda environment. This can be installed directly from its git 
    repository.  Ensure that the correct conda environment is activated, and then type

    .. code-block:: bash

        pip install git+git://github.com/davesproson/gribgrab.git

.. warning::

    The report is compiled via LaTeX, and requires `lualatex` to be installed and available
    in the current ``PATH``. Additionally, the report uses Nexa light and Nexa bold fonts,
    which are assumed to exist in ``/usr/share/fonts/opentype/nexa/``.

.. code-block:: python

    from ppodd.report import ReportCompiler

    report = ReportCompiler(
        d, flight_number='c231', token='123e4567-e89b-12d3-a456-426614174000',
        flight_folder='/path/to/flight_folder'
    )

    report.make()

The ``token`` keyword argument is a UUID4 string which allows access to flight information from the
``gluxe`` web application. A token can be provided by the maintainer of the FAAM website.

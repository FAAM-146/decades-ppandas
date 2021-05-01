===========================
Introduction & Installation
===========================

Introduction
============

DECADES-PPandas is a data postprocessing suite and framework. It is primarily intended for use
with data from the FAAM BAe-146 Atmospheric Research Aircraft, though it may be used more widely.
The framework includes readers for a number of different data sources, and an extensible plugin
architecture for data processing and production of data visualizations.

Access
======

DECADES-PPandas is open source, and is available on GitHub, at https://github.com/FAAM-146/decades-ppandas.

Installation
============

These instructions are aimed at linux users. A similar installation process should work on other
platforms; YMMV.

Conda
-----

It is recommended that a python environment managed by Anaconda (or Miniconda) is used. If it's not
available on your system, it can be downloaded from https://docs.conda.io/en/latest/miniconda.html.
For Linux (64 bit) users, the latest Miniconda installer can be obtained from 
https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86.sh. This is a shell script which should
executed in order to install Conda.

.. code::

    $ chmod +x Miniconda3-latest-Linux-x86.sh
    $ ./Miniconda-latest-Linux-x86.sh

This will launch the installer which will guide you through the Conda setup process.

Source Code
-----------

The DECADES-PPandas source code is available on GitHub, at https://github.com/FAAM-146/decades-ppandas.
It can be downloaded as an archive from the GitHub pages, but if it is available on your system, it is
recommended that you use the git CLI to clone the repository to your local system

.. code::

    $ git clone git@gitgub.com:FAAM-146/decades-ppandas.git

This will clone the repo into ``decades-ppandas`` in the current working directory.

Configuring an Environment
--------------------------

In the top level of the ``decades-ppandas`` repository is the file ``environment.yaml``. This contains
a list of (almost) all of the dependencies required by ``decades-ppandas``. You can use this file to
create a new conda environment with the command

.. code::

    $ conda env create -f environment.yaml

This will create a conda environment called decades-ppandas. To create an environment with a different
name, you may pass the argument :code:`-n <env_name>` to the command above. We'll assume that you've 
the default environment from here on.

The environment can be activated by typing

.. code::

    $ source activate decades-ppandas

Or, in more recent versions of Conda, the prefered alternative :code:`conda activate decades-ppandas`
may be used.

Installing decades-ppandas
--------------------------

At the top level of the ``decades-ppandas`` repository is a ``setup.py`` script. The currently 
recommended way to install this is via the ``pip`` package manager. Enusring that the correct
conda environment is active, ``decades-ppandas`` can be installed with the command

.. code::

    (decades-ppandas)$ pip install .

or 

.. code::

    (decades-ppandas)$ pip install --editable .

The difference here is that the former will copy the package into your conda environment (site-packages),
creating a stable install, while the latter will link to the source directory, meaning that any changes
to the source code will be immediately available in the installed package, allowing an update with a 
simple ``git pull``.

Running Tests
-------------

Tests are not run as part of the install process, but it is recommended that you run the test suite
after install to ensure that everything is working as expected. To run the tests, navigate into the 
``ppodd/tests`` directory and, ensuring the correct conda environment is active, run

.. code-block:: bash

    ./run_tests

This will run any available unit tests. If these pass, the installation has been successful.

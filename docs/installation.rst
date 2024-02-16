.. highlight:: shell

============
Installation
============

CoSApp is tested in environments based on Python 3.9 to 3.11.
The core package is compatible with Python 3.12; however, a few extra dependencies may not be available yet in this version of Python.

Stable release
--------------

The easiest way is to install the conda package:

.. code-block:: console

    conda install cosapp -c conda-forge

or the PyPi package:

.. code-block:: console

    pip install cosapp

If you do not know how to setup a conda environment, please refer to the guidelines below.

From sources
------------

The sources for CoSApp can be downloaded from the `Gitlab repo`_.

You can either clone the public repository:

.. code-block:: console

    git clone https://gitlab.com/cosapp/cosapp.git

Or download the `archive`_:

.. code-block:: console

    curl -OL https://gitlab.com/cosapp/cosapp/repository/master/archive.zip

Once you have a copy of the source, you can install it with:

.. code-block:: console

    python -m pip install . [tests,extra]

Extra
-----

Create a new conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conda is a powerful package manager that allows you to create isolated virtual environments, each containing their own set of packages, and their own Python version.
Creating a conda environment named ``env_name`` based on Python 3.11 containing ``cosapp`` is as simple as:

.. code-block:: console

    conda create -n env_name python=3.11 cosapp

Once the environment is created, you can activate it with:

.. code-block:: console

    conda activate env_name

Install Jupyter Lab to run tutorials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our tutorials are based on Jupyter notebooks.
Here is a list of additional dependencies you must install in your conda environment if you wish to run them locally:

.. code-block:: console

    conda install jupyterlab plotly nbformat -c conda-forge

To launch Jupyter Lab, execute command:

.. code-block:: console

    jupyter lab

and then navigate to the directory containing the tutorials (``docs/tutorials`` from the source code directory).

.. _Gitlab repo: https://gitlab.com/CoSApp/cosapp
.. _archive: https://gitlab.com/cosapp/cosapp/repository/master/archive.zip

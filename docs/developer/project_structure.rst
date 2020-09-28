Project Code Structure
----------------------

The global CoSApp software is built on top of multiple technologies and packages. This
section will describe the structure and link between them.

The code used can be split in 3 categories:

- Simulation code: the core of CoSApp_ belong in this category as well as physical models_ 
  and libraries_.
- Interactive Jupyter Widgets: `Jupyter Widgets`_ technology allow to sync data model 
  between Python it is the basis of interactive web UI elements with CoSApp models.
  In particular, `CoSApp widgets`_ provide data visualization and model interactions.
- The web UI: the user interface is built on top of a modified JupyterLab_. It comes with
  predefined extensions that are available on the internet and `customized extensions`_ for
  CoSApp.

Packages hierarchy
~~~~~~~~~~~~~~~~~~

The core software (including UI elements) is composed of three packages:

- cosapp_: A pure Python package containing the core simulation code.
- cosapp_notebook_: A Jupyter widgets package (Python and JS code) containing widgets to interact with simulation objects within JupyterLab.
- cosapp_ui_: A JupyterLab extension (Python and JS code) containing a server and frontend extension for JupyterLab.

Moreover as JupyterLab is a monolithic JavaScript application, it needs to be *compiled* to include extensions. This is complex for the end user;
especially as an access through the proxy to a public NPM packages (i.e. JavaScript packages) repository is required. The workaround used is
to distribute a dedicated JupyterLab with pre-installed extensions. That package is called jupyterlab_cosapp_.

The dependency relationships is described in the figure below:

.. mermaid::

   graph TD
      cosapp --> cosapp_notebook
      subgraph Python
        cosapp
      end
      subgraph Python & JavaScript
        cosapp_notebook --> jupyterlab_cosapp
        cosapp_ui --> jupyterlab_cosapp
      end


Due to the dependency link, installing the conda package *jupyterlab_cosapp* is sufficient
to obtain a working CoSApp environment.

It is advice to install *jupyterlab_cosapp* in the **base** conda environment. Then for
each project, you should create a new environment with at least the following packages::

   conda create -n myproject1 cosapp_notebook ipykernel


Packages description
~~~~~~~~~~~~~~~~~~~~

This section will described in more details the content of those three packages.

cosapp
^^^^^^

cosapp_ pure Python package contains the base classes of all CoSApp objects:

- :py:class:`~cosapp.ports.port.Port` defining variables at the interfaces of system
- :py:class:`~cosapp.systems.system.System` defining system; i.e. block with behavior code

It defines also resolution algorithms called *drivers*. They solve mathematical problems
addressing user simulation intentions: 

- :py:class:`~cosapp.drivers.nonlinearsolver.NonLinearSolver` to solve squared problem with or without iterative variables
- :py:class:`~cosapp.drivers.time.runge_kutta.RungeKutta` for transient simulation with Runge-Kutta explicit scheme
- :py:class:`~cosapp.drivers.time.euler.EulerExplicit` for transient simulation with Euler explicit scheme

Various helpers are also available in that package:

- :py:mod:`~cosapp.recorders`: Recorder to save simulation results
- :py:func:`~cosapp.utils.logging.set_log`: To set a simulation logger
- :py:func:`~cosapp.tools.fmu.exporter.to_fmu`: To export a CoSApp system as FMU
- :py:func:`~cosapp.tools.help.display_doc`: To display the documentation of CoSApp objects


The :py:mod:`cosapp` itself is a Python `namespace <https://docs.python.org/3/reference/import.html#namespace-packages>`_
and not a traditional Python package. This allows third-party packages to be included under the same
umbrella. This is for example used by the package cosapp_notebook_ to be included as ``cosapp.notebook``.


cosapp_notebook
^^^^^^^^^^^^^^^

cosapp_notebook_ is mixing JavaScript and Python code to build customized `Jupyter Widgets`_ to visualize and
interact with the CoSApp objects.

The Python code is stored in the package ``cosapp.notebook`` and the JavaScript is stored in ``src`` folder.

The deployment implies distributing a Python package and a NPM package. The Python package must be installed
in each kernel. But the NPM package must only be installed once in the JupyterLab frontend.


cosapp_ui
^^^^^^^^^

cosapp_ui_ is an extension for JupyterLab (NPM package) and the Jupyter server (Python package). Its main 
feature are:

- Handle file templates for CoSApp
- Handle project folder template for CoSApp
- Customized JupyterLab launcher
- Customized JupyterLab splash screen
- Add some links in the help section

The file templates are based on `Jinja2 <https://jinja.palletsprojects.com/>`_ template system. And the 
project are created from a `cookiecutter <https://cookiecutter.readthedocs.io/>`_ `template for CoSApp
<https://gitlab.com/cosapp/cookiecutter-cosapp-workspace>`_.

The deployment requires the distribution of a Python package and a NPM package. The Python package must be
installed once in the environment of the Jupyter server. And the NPM package must be installed once in
the JupyterLab frontend.

jupyterlab_cosapp
^^^^^^^^^^^^^^^^^

jupyterlab_cosapp_ is a packaged JupyterLab application with pre-installed extensions. It allows the
distribution of JupyterLab with pre-installed extensions without the need for compiling JupyterLab
for each extensions on the user machine. In particular, this avoid the need for the final user to
access a *npm* registry and to installed ``nodejs``.

That particular JupyterLab application can be launched with the command ``cosapp``. The options are
identical to the ones of ``jupyter lab``.

On Windows, when starting the application, a local proxy is launched. The proxy uses the `px-proxy <https://github.com/genotrance/px>`_
package. It reads its parameters from ``$COSAPP_CONFIG_DIR/px.ini`` - by default ``$COSAPP_CONFIG_DIR`` equals
``$HOME/.cosapp.d``.

.. note::

   To disable the proxy you can use the option ``cosapp --no-proxy``.

The deployment requires the distribution of a Python package (that includes all the needed JavaScript).

.. note::

   As it comes with pre-installed extensions and to avoid user heterogeneity, the extension manager
   contained in JupyterLab is disabled. But for interested developers, it is still possible to
   manage extensions through the command ``cosapp-labextension``. The syntax and options are identical
   to ``jupyter labextension`` command.

.. _CoSApp : https://gitlab.com/cosapp/cosapp 
.. _cosapp : https://gitlab.com/cosapp/cosapp 
.. _Jupyter Widgets : https://github.com/jupyter-widgets/ipywidgets
.. _CoSApp widgets : https://gitlab.com/cosapp/jupyterlab/cosapp_jupyter
.. _cosapp_notebook : https://gitlab.com/cosapp/jupyterlab/cosapp_jupyter
.. _JupyterLab : https://github.com/jupyterlab/jupyterlab
.. _customized extensions : https://gitlab.com/cosapp/jupyterlab/jupyterlab_cosapp_ui
.. _cosapp_ui : https://gitlab.com/cosapp/jupyterlab/jupyterlab_cosapp_ui
.. _jupyterlab_cosapp : https://gitlab.com/cosapp/jupyterlab/jupyterlab-cosapp

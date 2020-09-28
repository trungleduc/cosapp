Installation Use Cases
----------------------

CoSApp is available on Linux, MS Windows and Mac OS platforms.
To put things in context, we first give a short overview on
the technologies used to deploy CoSApp.


Technologies
~~~~~~~~~~~~

The programming language chosen for CoSApp is Python, for
* its ease of use;
* the huge and dynamic community (and therefore libraries) available out there;
* its inter-operability with Fortran and C/C++ code, commonly used by legacy simulation tools.

For the user interface, the choice went on Web technologies to forecast mixed devices
usage (professional computers, HPC and Software As A Service - SaaS) and to appeal with
a modern look and feel. More precisely a big trend today are interactive notebooks
and especially in the Python scientific community the Jupyter_ ecosystem with the
extensible frontend JupyterLab_ and a model state synchronization between Python and
JavaScript objects: `Jupyter Widgets`_. The nice thing about that ecosystem is a very dynamic
community (sometimes even too dynamic...) and a highly modular approach; extensions can
be created for all aspects.

So this narrows the technological stack down to:

- Python
   - The classical scientific suite: `numpy <https://docs.scipy.org/doc/>`_, `pandas <https://pandas.pydata.org/>`_ and `scipy <https://docs.scipy.org/doc/>`_
   - A web server: `tornado <https://www.tornadoweb.org/>`_ - package used by `Jupyter Server`_.
- Web technologies
   - `TypeScript <https://www.typescriptlang.org/docs/home.html>`_ is used instead of *JavaScript* in JupyterLab_; and this is a good thing.
   - JupyterLab_ frontend is based on a framework previously called ``phosphorjs`` and since JupyterLab 2.x `lumino <https://github.com/jupyterlab/lumino>`_.
   - `ReactJS <https://reactjs.org/>`_ plays nicely (and its usage is expanding) with JupyterLab_.

Regarding distribution of code,

- For Python, there are two major package managers around
   - ``pip`` using the public repository `Pypi.org <https://pypi.org/>`_.
   - ``conda`` using public channels from `Anaconda.org <https://anaconda.org/>`_.

- For Web technologies, JupyterLab_ uses a frozen version of `yarn <https://yarnpkg.com/>`_ called ``jlpm``. The
  public repository for the NPM packages is `npmjs.com <https://www.npmjs.com/>`_.

For Python, the choice went on *conda* as a simple shared folder can be turned in a packages channel. And to stick on
``jlpm`` for NPM packages. But to reduce the burden on the end user, the final JavaScript application is built as
a monolithic application distributed via a Python package (i.e. the JupyterLab way of distribution). And therefore
the final user needs only to access *conda* channels.

.. _Jupyter : https://jupyter.org/
.. _JupyterLab : https://jupyterlab.readthedocs.io/
.. _Jupyter Server : https://jupyter-server.readthedocs.io/
.. _Jupyter Widgets : https://github.com/jupyter-widgets/ipywidgets

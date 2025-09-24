.. scmlpick
.. =============

``scmlpick`` is a SeisComP module that performs real-time seismic phase picking
using machine-learning models (EQCCT by default), designed for operational networks
and research workflows. It supports multi-pipeline execution, per-station configuration
via SeisComP **bindings**, and **playback** for offline reprocessing.

Seamless integration with `SeisComP <https://www.seiscomp.de/>`_ (`official documentation <https://docs.gempa.de/seiscomp/current/>`_)
enables direct publishing of :term:`Pick` objects to the messaging system, use of station
bindings, and interoperability with other SeisComP modules.

Installation
=============

SeisComP System
^^^^^^^^^^^^^^^

- **Required**: `SeisComP <https://www.seiscomp.de/>`_
- **Minimum version**: Fully compatible with SeisComP releases ≥ ``4.0.0``
- **Recommended**: ``6.0.0`` or higher for improved stability and performance

Follow the official installation instructions:
`https://docs.seiscomp.de/ <https://docs.seiscomp.de/>`_
Once installed, verify that SeisComP is properly configured and accessible in your environment. 

Operating System
^^^^^^^^^^^^^^^^

- **Tested on**: ``Linux``
- **Validated on**: ``Ubuntu 22.04 LTS``

Programming Language
^^^^^^^^^^^^^^^^^^^^

- **Python ≥ 3.10**

.. note::
   Python versions prior to ``3.10`` are **not supported** due to incompatibilities
   with the ``ray`` library.

Required Python Packages
^^^^^^^^^^^^^^^^^^^^^^^^


It is strongly recommended to install these packages inside a dedicated virtual
environment (e.g., Conda) to avoid dependency conflicts.

.. code-block:: bash

      conda create -n scmlpick python=3.10
      conda activate scmlpick

Install the following Python packages:

.. code-block:: bash

      pip install ray
      pip install numpy==1.26.4      # Avoid numpy ≥ 2.0 to maintain compatibility
      pip install pandas
      pip install obspy
      pip install tensorflow
      pip install silence_tensorflow


Clone the Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the project repository into any local directory:

.. code-block:: bash

   git clone https://github.com/ut-beg-texnet/scmlpick.git

Deploy the Code into the SeisComP Installation Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copy all necessary files into your SeisComP installation using ``rsync`` to preserve
the directory structure:

.. code-block:: bash

   rsync -av /path/to/your-cloned-repository/seiscomp/ /path/to/your-seiscomp-installation/

- Replace ``/path/to/your-cloned-repository/seiscomp/`` with the absolute path to the seiscomp folder in your cloned repository.
- Replace ``/path/to/your-seiscomp-installation/`` with your SeisComP root (typically ``$SEISCOMP_ROOT/``).

.. note::
   This step integrates the module into the SeisComP environment, preserving file structure
   and permissions.

Predictor Installation
^^^^^^^^^^^^^^^^^^^^^^

The ``scmlpick-predicctor`` module must be installed **after** after the deployment of the scmlpikc reposity into seiscomp path.
The installation procedure depends on whether a virtual environment is being used.

**If NOT using a virtual environment**

.. code-block:: bash

   cd $SEISCOMP_ROOT/share/scmlpick/tools/scmlpick-predicctor
   pip3 install -e .

**If using a Conda or other virtual environment**

Activate your environment (replace ``scmlpick`` with your actual environment name):

   .. code-block:: bash

      conda activate scmlpick

Navigate to the predictor directory:

   .. code-block:: bash

      cd $SEISCOMP_ROOT/share/scmlpick/tools/scmlpick-predicctor

Install the predictor module:

   .. code-block:: bash

      pip install -e .

.. note::
   Once installed, verify that SeisComP is properly configured and accessible in your environment.

Getting Started
============

Real-Time Setup
^^^^^^^^^^^^^^^

Before starting using scmlpick in real-time you have to be sure in complete the following steps:

1. Create profile in the pipelines section in the module configuration (see :confval:`pipelines.profiles`.)
2. Create and configure pick target and location target groups in the profile. (see :confval:`pipelines.profiles.$name.pickTargetGroup` and :confval:`pipelines.profiles.$name.locTargetGroup`)
3. Create a **binding profile** with pick enable option (see :ref:`bindings`), the name should match with the module profile created in step 1.
4. Assign the binding profile to stations/streams in `scconfig <https://www.seiscomp.de/doc/apps/scconfig.html>`_ (or edit bindings files).
5. Start your messaging: ``scmaster start`` (or your standard SeisComP workflow).
6. Launch ``scmlpick``:

   .. code-block:: bash

      scmlpick -u user --debug

For executing on the command line simply call it with appropriate options, e.g:

   .. code-block:: bash

      seiscomp exec scmlpick -h

Next Steps
^^^^^^^^^^

* Create **module profiles** (see :ref:`profiles`) to separate dataflows and create specific processing for specific purposes.
* Depending the number of stations and resources available configure ray parameters (see :ref:`permormance`)
* Tune critial parameters like P and S models, probability thresholds, time windows lenght overlaps and more in the module configuration.
* Use (see :ref:`playback`) to postprocessing or validate parameters runing in playback mode with mseed files or historical data.

.. _profiles:
Module Profiles
===============

In **scmlpick**, in contrast to other SeisComP modules, creating aliases is **not** recommended.
Each alias starts a separate instance that runs in parallel and increases CPU/RAM usage.
Instead, run a single instance that processes all stations and use (see :confval:`pipelines.profiles`)
to define different workflows. Profiles let you run multiple independent pipelines
within one process, avoiding the need for ``scmlpick`` aliases.

Each profile can define its own:

* :confval:`pipelines.profiles.$name.pickTargetGroup` all picks emitted by this profile are published to that specific messaging group.
* :confval:`pipelines.profiles.$name.locTargetGroup` this profile subscribes to locations from that messaging group.
* :confval:`pipelines.profiles.$name.authorTarget` locations carrying this author are checked against the profile’s quality-control parameters.
* :confval:`pipelines.profiles.$name.region` polygon used to evaluate (filter/validate) locations processed by this profile.

Segment of scmlpick.cfg incuding two profiles:

.. code-block:: ini

   # List with profiles names.
   pipelines.profiles = PROF1, PROF2

   # This is the name of the group to send pick messages for specific profile
   pipelines.profiles.PROF1.pickTargetGroup = PPROF1

   # This is the name of the group to send location messages for specific profile
   pipelines.profiles.PROF1.locTargetGroup = LPROF1

   # This author should match the origin (location) author for locations to be
   # checked against quality control parameters.
   pipelines.profiles.PROF1.authorTarget = LOCSATPROF1

   # The BNA file defining regions (closed polygons) or tuple with 4 elements.
   # Inside this region, the origin evaluation status is TrueOrgEvalStat, and
   # outside is FalseOrgEvalStat. If this is set to none, then the origin location
   # will not be checked against any polygon.
   pipelines.profiles.PROF1.region = @DATADIR@/scmlpick/bna/PROF1.bna

   # This is the name of the group to send pick messages for specific profile
   pipelines.profiles.PROF2.pickTargetGroup = PPROF2

   # This is the name of the group to send location messages for specific profile
   pipelines.profiles.PROF2.locTargetGroup = LPROF2

   # This author should match the origin (location) author for locations to be
   # checked against quality control parameters.
   pipelines.profiles.PROF2.authorTarget = LOCSATPROF2

   # The BNA file defining regions (closed polygons) or tuple with 4 elements.
   # Inside this region, the origin evaluation status is TrueOrgEvalStat, and
   # outside is FalseOrgEvalStat. If this is set to none, then the origin location
   # will not be checked against any polygon.
   pipelines.profiles.PROF2.region = @DATADIR@/scmlpick/bna/PROF2.bna

.. note::
   Each **module profile** must have a corresponding **binding profile** that
   associates the profile with specific stations; see :ref:`bindings`.
   The **module profile name must match the binding profile name**.

.. _bindings:
Bindings
========

In SeisComP, *bindings* attach module-specific parameters to stations. ``scmlpick``
uses bindings to activate specific stations to be processed by an ``scmlpick`` profile.

Where to Configure
^^^^^^^^^^^^^^^^^^

* In `scconfig <https://www.seiscomp.de/doc/apps/scconfig.html>`_ assign the binding to the desired stations (bindings tab).
* Or edit files under ``$SEISCOMP_ROOT/etc/bindings/scmlpick/``

Each binding enables picking for the stations **bound** to it:

* :confval:`profiles.$name.pickEnable` Boolean flag to enable picking in this binding.

You can also set a station-specific **filter**. This filter is applied at the
beginning of the processing chain, before the ML algorithm runs:

* :confval:`profiles.$name.filter` Parameter following the standard SeisComP filter grammar (`reference <https://www.seiscomp.de/doc/base/filter-grammar.html>`_).

.. _playback:
Playback
========

``scmlpick`` supports **playback** for offline (non–real-time) processing. This mode is
useful for validation, benchmarking, parameter tuning, and research.

Data sources
^^^^^^^^^^^^

Playback can run against:

* **Pre-downloaded miniSEED files** fastest and fully repeatable.
* **Historical data via the configured SeisComP RecordStream** (e.g., FDSN/CAPS/SeedLink).

.. note::
   Querying large time windows from a RecordStream can be slow and will re-fetch data
   on every run. Prefer RecordStream playback for **short** time intervals or **one-off**
   executions. For repeated experiments, use local miniSEED files.

Requirements
^^^^^^^^^^^^

Playback uses the **same configuration** as real-time mode:

* A **module profile** (see :ref:`profiles`).
* **Bindings** whose profile name **matches** the module profile (see :ref:`bindings`).
* **Stations** assigned to that binding profile.
* A valid data source: miniSEED files available locally **or** a correctly configured
  RecordStream endpoint.

.. note::
   Keep :confval:`eqcct.windowLength`/:confval:`eqcct.timeShift` and other configuration parameters (see :ref:`Tuning`)
   consistent with your real-time setup if you want comparable latency and behavior between modes.

Command-line options
^^^^^^^^^^^^^^^^^^^^

Use the following options to run ``scmlpick`` in **playback** mode.

Input source (choose one)
-------------------------

* ``--recordstream``  
  Read historical data directly from a SeisComP RecordStream (e.g., FDSN/CAPS/SeedLink).

* ``--mseed-files <PATH>``  
  Read from one or more local miniSEED files, or a directory containing them.

Playback window & profile
-------------------------

* ``--profile <NAME>``  
  Select the module profile to use. The profile name must match an existing
  module profile **and** its corresponding binding profile.

* ``--start "YYYY-MM-DD HH:MM:SS"``  
* ``--end   "YYYY-MM-DD HH:MM:SS"``  
  Time window for playback, applied to the chosen input source.

Output
------

* ``--output-file <FILE>``  
  Write results to an XML file.

* ``--database``  
  Send results directly to the SeisComP database.

Ray parallelism (optional)
--------------------------

* ``--cpu-number <N>``  
  Number of CPUs available to Ray. If omitted, the value from the module configuration is used.

* ``--max-tasks <N>``  
  Maximum number of concurrent tasks sent to the predictor. If omitted, the configured
  default is used.

Running Playback
^^^^^^^^^^^^^^^^

*Using local miniSEED:*

.. code-block:: bash

   scmlpick \
     --playback \
     --profile playback \
     --mseed-files /data/mseed/2025-08-01/ \
     --start "2025-08-01 00:00:00" \
     --end   "2025-08-01 06:00:00" \
     --output-file /tmp/scmlpick_playback.xml \
     -u user \
     --debug

*Using RecordStream:*
  
.. code-block:: bash

   scmlpick \
     --playback \
     --profile playback \
     --recordstream \
     --start "2025-08-01 00:00:00" \
     --end   "2025-08-01 03:00:00" \
     --database
     -u user \
     --debug

Tuning
======

ML algorithm configuration
--------------------------
Controls which pre-trained models are used and how strict the decision rules are.
Select P/S models appropriate for your seismicity and set probability thresholds
that govern when picks are emitted; overlap and batch size affect latency and throughput.

* :confval:`eqcct.Pmodel`
* :confval:`eqcct.Smodel`
* :confval:`eqcct.eqcctPthr`
* :confval:`eqcct.eqcctSthr`
* :confval:`eqcct.eqcctOverlap`
* :confval:`eqcct.eqcctBatchSize`

Data sent for processing
------------------------
Defines the sliding-window strategy and submission timing for inference. These parameters
control window length, overlap, pre-filter warm-up, station availability requirements, and
the delays/waits that separate first/second computations and delayed-data paths.

* :confval:`eqcct.windowLength`
* :confval:`eqcct.timeShift`
* :confval:`eqcct.filterShift`
* :confval:`eqcct.minStasBulk`
* :confval:`eqcct.minDelayStasBulk`
* :confval:`eqcct.firstWait`
* :confval:`eqcct.secondWait`
* :confval:`eqcct.maxWait`
* :confval:`eqcct.startLatency`
* :confval:`eqcct.timeRemoveTrace`
* :confval:`eqcct.traceDelay`

Ray configuration
-----------------
Tunes parallelism and queueing for the predictor backend. Use these settings to balance
CPU usage and concurrency against your latency budget.

* :confval:`ray.numCPUs`
* :confval:`ray.maxTasksQueue`

Locations quality control
-------------------------
Applies simple, rule-based checks to location results. Uncertainty and geometry thresholds
determine whether an origin is assigned the “true” or “false” evaluation status; both
statuses are configurable to match your operational workflow.

* :confval:`qcheck.latUncTHR`
* :confval:`qcheck.lonUncTHR`
* :confval:`qcheck.depthUncTHR`
* :confval:`qcheck.depthTHR`
* :confval:`qcheck.azGapTHR`
* :confval:`qcheck.TrueOrgEvalStat`
* :confval:`qcheck.FalseOrgEvalStat`

.. toctree::
   :maxdepth: 1
   :caption: ML methods

   apps/ml_methods

ML methods
==========

.. toctree::
   :maxdepth: 1
   :caption: Resources Optimization

   apps/resources_optimization

Resources Optimization
======================

References
==========

You can cite in-text using footnote-style citations, for example:
“As shown by Ross et al. [ross2018]_ …”

.. [ross2018] Ross, Z.E., et al. (2018). Generalized Seismic Phase Detection with Deep Learning. *Bulletin of the Seismological Society of America*, 108(5), 2894–2901. https://doi.org/10.1785/0120180080
.. [allen1982] Allen, R.V. (1982). Automatic phase pickers: Their present use and future prospects. *Bulletin of the Seismological Society of America*, 72(6), S225–S242.


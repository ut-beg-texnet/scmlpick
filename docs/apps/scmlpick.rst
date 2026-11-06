.. scmlpick
.. =============

``scmlpick`` is a SeisComP module that performs real-time seismic phase picking.
EQCCT is the default architecture. The same process can run UTrans
(``utrans_eqcct``), PhaseNet, or EQTransformer by setting
:confval:`eqcct.architecture`. One process uses one architecture. Several
profiles inside that process split stations, messaging groups, and regions.
Per-station configuration uses SeisComP **bindings**. **Playback** reprocesses
stored waveforms.

Seamless integration with `SeisComP <https://www.seiscomp.de/>`_ (`official documentation <https://docs.gempa.de/seiscomp/current/>`_)
enables direct publishing of :term:`Pick` objects to the messaging system, use of station
bindings, and interoperability with other SeisComP modules.

Installation
============

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

- **Python 3.10, 3.11, or 3.12** on glibc Linux. Python 3.9 and older are not supported. Alpine and other musl systems are outside the TensorFlow wheels used here.

Clone the Repository
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/ut-beg-texnet/scmlpick.git
   cd scmlpick

Deploy the Module
^^^^^^^^^^^^^^^^^

Copy the ``seiscomp/`` directory onto the SeisComP root. Do not copy the repository root.

.. code-block:: bash

   rsync -a seiscomp/ "$SEISCOMP_ROOT/"

``$SEISCOMP_ROOT`` is usually ``/opt/seiscomp``. The model weights for EQCCT, UTrans, PhaseNet, and EQTransformer are under ``seiscomp/share/scmlpick/model/`` and are included by that copy.

``seiscomp/etc/scmlpick.cfg`` in this repository is one host's configuration. On another computer, edit the messaging connection, the subscriptions, the model paths, and the profiles before starting the module. Station bindings and region polygons are not stored in the repository.

Python Environment Used by the Service
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``seiscomp/bin/scmlpick`` starts ``/opt/seiscomp/share/scmlpick/pyenv/bin/python`` when that file exists. Create that environment and install this repository into it. Run the last command from the repository root.

.. code-block:: bash

   python3 -m venv /opt/seiscomp/share/scmlpick/pyenv
   /opt/seiscomp/share/scmlpick/pyenv/bin/python -m pip install --upgrade pip
   /opt/seiscomp/share/scmlpick/pyenv/bin/python -m pip install .

That installs the predictor and the libraries declared in the root ``setup.py``: NumPy 1.26, pandas 2.x, SciPy 1.15, ObsPy, Ray 2.59, TensorFlow 2.17.0, and tf-keras 2.17.0. Repeating the last command reinstalls the same package into that environment.

If SeisComP is not installed under ``/opt/seiscomp``, create the environment at ``$SEISCOMP_ROOT/share/scmlpick/pyenv`` and point ``_PY`` at the top of ``seiscomp/bin/scmlpick`` to that interpreter.

``python3 setup.py --make-env``, run from the repository root, creates ``.venv`` inside the repository. That environment is for working on the predictor. The service does not use it.

``seiscomp/share/scmlpick/tools/scmlpick-predicctor/setup.py`` installs the same package when the current directory is already the deployed SeisComP tree (``pip install -e .`` from that directory). A new installation uses the ``setup.py`` at the repository root.

Enable and Start
^^^^^^^^^^^^^^^^

.. code-block:: bash

   seiscomp update-config scmlpick
   seiscomp enable scmlpick
   seiscomp start scmlpick

Getting Started
===============

Real-Time Setup
^^^^^^^^^^^^^^^

Before starting ``scmlpick`` in real time:

1. Choose one architecture with :confval:`eqcct.architecture` and point :confval:`eqcct.Pmodel` and :confval:`eqcct.Smodel` at the matching weights.
2. Create one or more profiles in :confval:`pipelines.profiles`. Every profile in this process uses the architecture from step 1.
3. Set :confval:`pipelines.profiles.$name.pickTargetGroup` and :confval:`pipelines.profiles.$name.locTargetGroup` for each profile.
4. Create a binding profile with the same name and set :confval:`profiles.$name.pickEnable` (see :ref:`bindings`).
5. Assign that binding to the stations in `scconfig <https://www.seiscomp.de/doc/apps/scconfig.html>`_, or edit the key files.
6. Start messaging (``scmaster`` or the usual SeisComP workflow).
7. Launch ``scmlpick``:

   .. code-block:: bash

      scmlpick -u user --debug

For executing on the command line simply call it with appropriate options, e.g:

   .. code-block:: bash

      seiscomp exec scmlpick -h

Next Steps
^^^^^^^^^^

* Add profiles (see :ref:`profiles`) when one architecture must publish to more than one messaging group.
* Start a second module name when a second architecture must run at the same time. See :ref:`architectures`.
* Set the Ray parameters in :ref:`Tuning` for the number of stations on the machine.
* Use :ref:`playback` to reprocess miniSEED or a RecordStream.

.. _architectures:

Picker Architectures
====================

Set :confval:`eqcct.architecture`. The weights under
``$SEISCOMP_ROOT/share/scmlpick/model/`` match the four names below.

``eqcct``
   Compact convolutional transformer. Default.
   :confval:`eqcct.Pmodel` and :confval:`eqcct.Smodel` are the two ModelPS files
   ``test_trainer_024.h5`` and ``test_trainer_021.h5``.

``utrans_eqcct``
   EQCCT picking heads on a pretrained U-Trans foundation encoder. The encoder
   output is concatenated as a fourth input channel.
   See https://doi.org/10.1038/s41598-026-41454-x.
   Point Pmodel and Smodel at the Foundation-EQCCT checkpoints
   ``test_trainer_P_FoundationEQCCT_026.h5`` and
   ``test_trainer_S_FoundationEQCCT_025.h5``. Plain EQCCT weights do not load here.

``phasenet``
   Official PhaseNet U-Net (Zhu and Beroza, 2019). One checkpoint directory,
   ``model/PhaseNet/190703-214543``, set as :confval:`eqcct.Pmodel`.
   :confval:`eqcct.Smodel` is ignored. PhaseNet normalizes each window itself.
   Its published probability threshold is about 0.3, higher than the EQCCT defaults.

``eqtransformer``
   Official EQTransformer (Mousavi et al., 2020). Point :confval:`eqcct.Pmodel`
   at ``model/EQTransformer/EqT_original_model.h5`` or at that directory.
   :confval:`eqcct.Smodel` is ignored. This file was saved with Keras 2 and is
   opened through the ``tf-keras`` package. Published P and S thresholds for the
   mseed predictor are 0.1. The 1–45 Hz bandpass used in the bindings matches
   the STEAD preprocessing described for this model.

:confval:`eqcct.gpuID` is the GPU index. Set it to ``-1`` to run on CPU.
:confval:`ray.precision` selects ``bf16`` (default) or ``fp32`` for every architecture.
The weight files stay float32. ``bf16`` changes the arithmetic.

.. _profiles:
Module Profiles
===============

Profiles divide one process. They do not select the neural network.
:confval:`eqcct.architecture` is shared by every profile in that process.

Use several profiles when the same picker must send picks to different groups,
follow different location authors, or apply different region polygons. That is
the usual operational layout: one EQCCT process, one profile per network region.

Run a separate process when the architecture must differ. Give that process its
own module name: a symbolic link to ``bin/scmlpick``, its own
``etc/<name>.cfg``, and its own ``etc/key/<name>/`` bindings. Each process
loads one model and uses its own CPU and memory. Four architectures therefore
mean four processes, not four profiles.

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

* In `scconfig <https://www.seiscomp.de/doc/apps/scconfig.html>`_, assign the binding to the stations.
* Or edit ``$SEISCOMP_ROOT/etc/key/<module>/station_NET_STA``. ``<module>`` is ``scmlpick``, or the name of the symbolic link when more than one architecture is running. ``seiscomp update-config`` writes the corresponding files under ``etc/bindings``.

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
  Write picks to this XML file. The file is replaced as results arrive and holds every pick from the run.

* ``--database``  
  Send results directly to the SeisComP database.

Ray parallelism (optional)
--------------------------

* ``--cpu-number <N>``  
  Overrides :confval:`ray.numCPUs` for this run. In model-actor mode that is the number of processes that load the picker. If omitted, the value from the module configuration is used.

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
     --database \
     -u user \
     --debug

.. _Tuning:

Tuning
======

ML algorithm configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^
Controls which pre-trained models are used and how strict the decision rules are.
Select P/S models appropriate for your seismicity and set probability thresholds
that govern when picks are emitted; overlap and batch size affect latency and throughput.

* :confval:`eqcct.architecture`
* :confval:`eqcct.Pmodel`
* :confval:`eqcct.Smodel`
* :confval:`eqcct.eqcctPthr`
* :confval:`eqcct.eqcctSthr`
* :confval:`eqcct.gpuID`

PhaseNet and EQTransformer ignore :confval:`eqcct.Smodel`. Raise the P and S thresholds
toward the values published with those models. The EQCCT defaults, 0.001 and 0.002,
are much lower.

Data sent for processing
^^^^^^^^^^^^^^^^^^^^^^^^
The model scores :confval:`eqcct.windowLength` seconds (default 60). The process
also keeps :confval:`eqcct.filterShift` seconds before that window so the filter
can warm up. A packet is therefore ``windowLength + filterShift`` seconds long.

The next packet starts ``windowLength - timeShift`` seconds later. With
``windowLength = 60``, ``filterShift = 10``, and ``timeShift = 20``, each packet
is 70 seconds and the step is 40 seconds. The 20 seconds of
:confval:`eqcct.timeShift` are the overlap that two consecutive packets both score.

A station is sent when it holds a full packet. If its start is behind the newest
complete packet by less than :confval:`eqcct.traceDelay` (default 120 seconds),
it is caught up from its own start, one packet per pass. If it is behind by more
than ``traceDelay``, that old packet is not sent. The samples stay in memory and
the station joins the live packet when it covers it. Samples older than
:confval:`eqcct.timeRemoveTrace` (default 600 seconds) are removed.

The package defaults are ``filterShift = 2.5`` and ``timeShift = 30``. An
operational configuration commonly sets them to 10 and 20. Keep one pair of
values for a running system. Playback should use the same pair when the results
are compared with real time.

These parameters control that schedule:

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
^^^^^^^^^^^^^^^^^
:confval:`ray.numWorkers` splits the station list across worker processes. It does
not set how many Ray processes exist. The default is 1. Raise it only when one
process is holding more stations than it can cut and return inside one step.
:confval:`ray.numCPUs` is the number of model processes started in ``modelactor``
mode. Each one loads the picker once. It is not the same setting as
:confval:`ray.numWorkers`.

:confval:`ray.execution` is ``modelactor`` by default. Each Ray process loads the
picker once and reuses it for every station it receives. ``task`` loads the picker
inside a Ray task for every station. In that mode :confval:`ray.maxCalls` exits
the worker after this many finished stations so its memory is returned. The
station in progress is not interrupted. ``modelactor`` ignores ``maxCalls``.

:confval:`ray.precision` is ``bf16`` or ``fp32``.

* :confval:`ray.numWorkers`
* :confval:`ray.numCPUs`
* :confval:`ray.maxTasksQueue`
* :confval:`ray.execution`
* :confval:`ray.precision`
* :confval:`ray.maxCalls`

Locations quality control
^^^^^^^^^^^^^^^^^^^^^^^^^
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

References
==========

.. [mousavi2020] Mousavi, S.M., Ellsworth, W.L., Zhu, W., Chuang, L.Y., and Beroza, G.C. (2020). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. *Nature Communications*, 11, 3952. https://doi.org/10.1038/s41467-020-17591-w
.. [utrans] U-Trans foundation encoder used by ``utrans_eqcct``. https://doi.org/10.1038/s41598-026-41454-x
.. [zhu2019] Zhu, W., and Beroza, G.C. (2019). PhaseNet: a deep-neural-network-based seismic arrival-time picking method. *Geophysical Journal International*, 216(1), 261–273. https://doi.org/10.1093/gji/ggy423


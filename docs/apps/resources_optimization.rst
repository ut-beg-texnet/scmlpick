.. resources_optimization
.. =============

Resources Optimization
======================

``ray.numWorkers`` splits the station list across the processes that cut
waveforms. It does not create the Ray pool. The default is 1. Increase it
when one process cannot cut and return its stations inside one step.
``ray.numCPUs`` is the number of model processes started in model-actor mode.
Each one loads the picker once.

``ray.execution = modelactor`` loads each picker once per Ray process and
reuses it. This is the default. ``task`` loads the picker for every station.
In task mode, ``ray.maxCalls`` exits that process after the given number of
finished stations so its memory is returned. The station in progress is not
interrupted. Model-actor mode ignores ``maxCalls``.

``ray.precision = bf16`` is the default arithmetic. ``fp32`` is the previous
full-precision path.

A packet is ``windowLength + filterShift`` seconds. The next packet starts
``windowLength - timeShift`` seconds later. Stations more than ``traceDelay``
seconds behind the newest complete packet are not sent as their own old
packet. They are picked on the live packet when they cover it. Samples older
than ``timeRemoveTrace`` are dropped.

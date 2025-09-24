.. ml_methods
.. =============

ML Methods
==========

Overview
--------
``scmlpick`` is **model-agnostic**: it ships with EQCCT by default but can host other
ML pickers through a uniform inference interface. This section explains the ML
models supported, their inputs/outputs, and how decisions are formed.

Model-Agnostic Interface
------------------------
- **Inputs**: Continuous, equally sampled waveforms (Z or 3C), windowed with
  ``window.length`` and ``window.overlap`` (see :doc:`configuration`).
- **Outputs**: Per-sample probabilities (e.g., :math:`P(P)`, :math:`P(S)`), optional
  uncertainty/quality measures.
- **Contract**: Models must expose a callable that accepts an ``(N, C, T)`` tensor (or
  equivalent) and returns probabilities aligned to input time.

Default Model: EQCCT
--------------------
- **Architecture**: Compact convolutional front-end + transformer encoder for temporal
  context (sliding-window inference).
- **I/O**: Probabilities for P and S per sample; optional logits for calibration.
- **Strengths**: Robustness under moderate noise, good latency/throughput balance.
- **Limits**: Requires threshold tuning for very noisy stations; calibration recommended
  when domain shifts are large.

Other Supported Models
----------------------
Use separate **profiles** (see :doc:`configuration`) to run alternative pickers:

- **ONNX models**: Set ``model.file = /path/to/model.onnx`` and (optionally)
  ``model.device = cuda:0``.
- **PyTorch models**: If enabled in your build, set ``model.backend = torch`` and point
  to the ``.pt`` weights.
- **Z-only vs 3C**: Choose via binding/profile ``streams = HHZ`` or ``HHZ,HH1,HH2``.

Example (two models at once)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: ini

   [general]
   profiles = ["eqcct_ops", "alt_model_3c"]

   [profiles.eqcct_ops]
   streams = "HHZ"
   model.file = "@plugins/scmlpick/models/eqcct.onnx"
   p.threshold = 0.33
   s.threshold = 0.00

   [profiles.alt_model_3c]
   streams = "HHZ,HH1,HH2"
   model.file = "@plugins/scmlpick/models/altpicker.onnx"
   p.threshold = 0.30
   s.threshold = 0.30
   minSNR = 2.5

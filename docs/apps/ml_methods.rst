.. ml_methods
.. =============

ML Methods
==========

``scmlpick`` runs one of four pickers. The choice is
``eqcct.architecture`` in the module configuration. Every profile in that
process uses the same architecture. A second architecture needs a second
process. See the scmlpick page for the parameter names and the weight files
shipped under ``share/scmlpick/model/``.

``eqcct``
   Compact convolutional transformer. This is the default. P and S each have
   their own ModelPS checkpoint.

``utrans_eqcct``
   EQCCT picking heads fed by a pretrained U-Trans foundation encoder. The
   encoder output is a fourth input channel. Pmodel and Smodel must be the
   Foundation-EQCCT checkpoints, not the plain EQCCT weights.
   https://doi.org/10.1038/s41598-026-41454-x

``phasenet``
   Official PhaseNet U-Net (Zhu and Beroza, 2019). Pmodel is the
   ``190703-214543`` checkpoint directory. Smodel is ignored. The published
   probability threshold is about 0.3.

``eqtransformer``
   Official EQTransformer (Mousavi et al., 2020). Pmodel is
   ``EqT_original_model.h5``. Smodel is ignored. Loading it requires
   ``tf-keras`` at the same TensorFlow release. The published P and S
   thresholds for the mseed predictor are 0.1.

All four return P and S probability curves on 60-second, three-component
windows. ``ray.precision`` selects bfloat16 or float32 arithmetic. The stored
weights stay float32.

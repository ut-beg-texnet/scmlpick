# PhaseNet model directory

Official PhaseNet checkpoint from AI4EPS/PhaseNet (Zhu & Beroza, 2019),
directory name `190703-214543`. scmlpick does not retrain these weights.

To run this picker, set in scmlpick.cfg:

```
eqcct.architecture = phasenet
eqcct.Pmodel = @DATADIR@/scmlpick/model/PhaseNet/190703-214543
eqcct.eqcctPthr = 0.3
eqcct.eqcctSthr = 0.3
```

`eqcct.Smodel` is ignored. PhaseNet is one network with a 3-class softmax
(noise, P, S). The published probability threshold is 0.3. The EQCCT defaults
(0.001 / 0.002) are far below that and will keep low-probability bumps.

The module still bandpasses the stream before the model, which the official
predictor does not do unless `highpass_filter` is set. The network itself
applies PhaseNet's sliding mean/std normalization.

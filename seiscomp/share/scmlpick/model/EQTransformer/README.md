# EQTransformer model directory

Official weights from https://github.com/smousavi05/EQTransformer
(Mousavi et al., 2020). scmlpick does not retrain them.

`EqT_original_model.h5` is the model in the paper. `EqT_model_conservative.h5`
is the other file shipped in that repository.

```
eqcct.architecture = eqtransformer
eqcct.Pmodel = @DATADIR@/scmlpick/model/EQTransformer/EqT_original_model.h5
eqcct.eqcctPthr = 0.1
eqcct.eqcctSthr = 0.1
```

`eqcct.Smodel` is ignored. The Python environment needs `tf-keras==2.17.0`
beside TensorFlow, because the file was saved with Keras 2.3.

The published file leaves SpatialDropout in training mode, so a raw
`model.predict()` is random. scmlpick turns that dropout off at inference.
The weights are unchanged.

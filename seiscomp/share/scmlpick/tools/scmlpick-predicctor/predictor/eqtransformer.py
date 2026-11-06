# predictor/eqtransformer.py
#
# Official EQTransformer checkpoint (Mousavi et al., Nature Communications, 2020,
# https://doi.org/10.1038/s41467-020-17591-w), loaded from the Keras 2.3 file
# published in https://github.com/smousavi05/EQTransformer.
#
# The file is EqT_original_model.h5, the model used in the paper. It is not
# rebuilt and it is not retrained. tf_keras reads it with the three custom
# layers from that repository (SeqSelfAttention, FeedForward, LayerNormalization).
#
# The published file was saved with SpatialDropout1D(..., training=True), so a
# plain model.predict() drops channels at random and two calls on the same
# trace disagree. Before the model is built, those layers are forced to
# inference (training=False). The weights stay the official ones, and a trace
# then always produces the same probabilities.

import os

import numpy as np


def _silence_dropout_at_inference():
    """Make SpatialDropout1D follow predict()'s inference mode.

    Must run before load_model(), because the saved graph calls the layer
    with training=True baked into every residual block.
    """
    from tf_keras.layers import SpatialDropout1D

    if getattr(SpatialDropout1D.call, "_eqt_inference", False):
        return
    original = SpatialDropout1D.call

    def call(self, inputs, training=None):
        return original(self, inputs, training=False)

    call._eqt_inference = True
    SpatialDropout1D.call = call


def _checkpoint_file(path):
    path = os.fspath(path)
    if os.path.isdir(path):
        for name in ("EqT_original_model.h5", "EqT_model.h5"):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            "No EQTransformer file in %s. Expected EqT_original_model.h5 "
            "from the official repository." % path
        )
    if os.path.isfile(path):
        return path
    raise FileNotFoundError("EQTransformer checkpoint not found: %s" % path)


class _EqTransformerPicker:
    """Runs the official model and returns detection, P and S.

    The detection curve used to be dropped here so scmlpick could keep its
    two-curve interface. That made every window eligible for a P and an S
    pick. The curves are returned together and associated in predictor.py
    the way the published mseed predictor does.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, data, verbose=0, **kwargs):
        if isinstance(data, np.ndarray):
            waveforms = data
        elif hasattr(data, "__len__") and hasattr(data, "__getitem__"):
            # scmlpick's generator is a Keras 3 Sequence of {'input': batch}.
            # The checkpoint is a tf_keras model, which does not accept that
            # Sequence type, so the batches are stacked here.
            batches = []
            for index in range(len(data)):
                batch = data[index]
                if isinstance(batch, dict):
                    batch = batch["input"]
                batches.append(np.asarray(batch, dtype=np.float32))
            if not batches:
                empty = np.zeros((0, 6000, 1), dtype=np.float32)
                return empty, empty, empty
            waveforms = np.concatenate(batches, axis=0)
        else:
            waveforms = data
        detection, pred_p, pred_s = self.model.predict(waveforms, verbose=verbose)
        return detection, pred_p, pred_s


def load_eqtransformer_model(input_modelP, input_modelS=None):
    """Loader registered as architecture 'eqtransformer'.

    Same call shape as load_eqcct_model(P, S). EQTransformer is one file, so
    `input_modelS` is ignored. `input_modelP` is EqT_original_model.h5 or a
    directory that contains it.

    The official input is (batch, 6000, 3), components in the same E, N, Z
    order as the rest of scmlpick, normalized by per-channel mean and standard
    deviation. That is already what the generator does, so this loader does
    not normalize again. predict() returns the three official sigmoid curves
    (detection, P, S), each (batch, 6000, 1).
    """
    try:
        from tf_keras.models import load_model
    except ImportError as exc:
        raise ImportError(
            "EQTransformer needs the tf-keras package (Keras 2 API) next to "
            "TensorFlow: pip install tf-keras==2.17.0"
        ) from exc

    from .eqtransformer_layers import FeedForward, LayerNormalization, SeqSelfAttention

    _silence_dropout_at_inference()
    model = load_model(
        _checkpoint_file(input_modelP),
        custom_objects={
            "SeqSelfAttention": SeqSelfAttention,
            "FeedForward": FeedForward,
            "LayerNormalization": LayerNormalization,
        },
        compile=False,
    )
    # The saved file labels every layer float32. Point them at the policy
    # selected by begin_precision() before the first predict().
    from .precision import apply_loaded_policy
    model = apply_loaded_policy(model)
    return _EqTransformerPicker(model)

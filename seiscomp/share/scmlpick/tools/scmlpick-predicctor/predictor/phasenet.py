# predictor/phasenet.py
#
# Keras transcription of the official PhaseNet U-Net
# (Zhu & Beroza, Geophysical Journal International, 2019;
# https://doi.org/10.1093/gji/ggy423, https://github.com/AI4EPS/PhaseNet).
#
# The published network is a TensorFlow 1 graph (tf.compat.v1.layers) and its
# weights are the checkpoint model/190703-214543. This file rebuilds that graph
# with tf.keras so it can run next to EQCCT and UTRANS, then copies every
# trained tensor (convolution kernels and batch-norm gamma/beta/moving stats)
# by name. It does not retrain. gamma is applied because the 2019 checkpoint
# stores trained gamma values, including Adam slots.
#
# Official layout, kept here:
#   input  (batch, time, 1, 3) with components E, N, Z
#   softmax output (batch, time, 1, 3) = noise, P, S
#   depths=5, filters_root=8, kernel (7, 1), stride (4, 1), epsilon=1e-3
#
# scmlpick still feeds 60 s / 6000-sample windows. PhaseNet is fully
# convolutional, so the same weights run on 6000 samples in one pass, which is
# how the official predictor consumes a trace longer than 30 s.

import os

import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# Hyperparameters copied from ModelConfig in the official phasenet/model.py
# and from model/190703-214543/config.log.
DEPTHS = 5
FILTERS_ROOT = 8
KERNEL_SIZE = (7, 1)
POOL_SIZE = (4, 1)
BN_EPSILON = 1e-3
BN_MOMENTUM = 0.99
WINDOW_SAMPLES = 3000
SCMLPICK_SAMPLES = 6000


def normalize_long(data, axis=(0,), window=WINDOW_SAMPLES):
    """Sliding mean/std used by the official predictor on a long trace.

    Copied from AI4EPS/PhaseNet phasenet/data_reader.py (MIT, Weiqiang Zhu).
    data shape is (nt, nsta, nch).
    """
    nt, nar, nch = data.shape
    if window is None:
        window = nt
    shift = window // 2

    dtype = data.dtype
    data_pad = np.pad(data, ((window // 2, window // 2), (0, 0), (0, 0)), mode="reflect")
    t = np.arange(0, nt, shift, dtype="int")
    std = np.zeros([len(t) + 1, nar, nch])
    mean = np.zeros([len(t) + 1, nar, nch])
    for i in range(1, len(std)):
        std[i, :] = np.std(data_pad[i * shift : i * shift + window, :, :], axis=axis)
        mean[i, :] = np.mean(data_pad[i * shift : i * shift + window, :, :], axis=axis)

    t = np.append(t, nt)
    std[-1, ...], mean[-1, ...] = std[-2, ...], mean[-2, ...]
    std[0, ...], mean[0, ...] = std[1, ...], mean[1, ...]

    t_interp = np.arange(nt, dtype="int")
    std_interp = interp1d(t, std, axis=0, kind="slinear")(t_interp)
    mean_interp = interp1d(t, mean, axis=0, kind="slinear")(t_interp)
    tmp = np.sum(std_interp, axis=(0, 1))
    std_interp[std_interp == 0] = 1.0
    data = (data - mean_interp) / std_interp

    nonzero = np.count_nonzero(tmp)
    if (nonzero < 3) and (nonzero > 0):
        data *= 3.0 / nonzero

    return data.astype(dtype)


def _normalize_batch(batch):
    """Apply official normalize_long to a (batch, time, 3) window."""
    batch = np.asarray(batch, dtype=np.float32)
    out = np.empty_like(batch, dtype=np.float32)
    for i in range(batch.shape[0]):
        sample = batch[i][:, np.newaxis, :]
        out[i] = normalize_long(sample)[:, 0, :]
    return out


class OfficialNormalizeLong(layers.Layer):
    """Keras wrapper around the official sliding normalization."""

    def call(self, inputs):
        normalized = tf.numpy_function(_normalize_batch, [inputs], tf.float32)
        normalized.set_shape(inputs.shape)
        return normalized


class CropAndConcat(layers.Layer):
    """Center-crop the upsampled map onto the skip, then concatenate channels.

    Same index arithmetic as crop_and_concat() in the official model.py.
    """

    def call(self, inputs):
        skip, upsampled = inputs
        skip_shape = tf.shape(skip)
        up_shape = tf.shape(upsampled)
        offsets = tf.stack([
            tf.zeros([], dtype=tf.int32),
            (up_shape[1] - skip_shape[1]) // 2,
            (up_shape[2] - skip_shape[2]) // 2,
            tf.zeros([], dtype=tf.int32),
        ])
        size = tf.stack([skip_shape[0], skip_shape[1], skip_shape[2], up_shape[3]])
        cropped = tf.slice(upsampled, offsets, size)
        return tf.concat([skip, cropped], axis=3)


def _bn(name):
    # scale=True because model_95.ckpt stores a trained gamma for every BN.
    return layers.BatchNormalization(
        axis=-1,
        momentum=BN_MOMENTUM,
        epsilon=BN_EPSILON,
        center=True,
        scale=True,
        name=name,
    )


def build_phasenet_unet():
    """Official U-Net. Input (batch, time, 1, 3), softmax (batch, time, 1, 3)."""
    inputs = layers.Input(shape=(None, 1, 3), name="phasenet_waveform")

    net = layers.Conv2D(
        FILTERS_ROOT, KERNEL_SIZE, padding="same", use_bias=True, name="input_conv"
    )(inputs)
    net = _bn("input_bn")(net)
    net = layers.ReLU(name="input_relu")(net)

    skips = []
    for depth in range(DEPTHS):
        filters = int(2 ** depth * FILTERS_ROOT)
        level = depth + 1
        net = layers.Conv2D(
            filters, KERNEL_SIZE, padding="same", use_bias=False,
            name="down_conv1_%d" % level,
        )(net)
        net = _bn("down_bn1_%d" % level)(net)
        net = layers.ReLU(name="down_relu1_%d" % level)(net)
        skips.append(net)
        if depth < DEPTHS - 1:
            net = layers.Conv2D(
                filters, KERNEL_SIZE, strides=POOL_SIZE, padding="same", use_bias=False,
                name="down_conv3_%d" % level,
            )(net)
            net = _bn("down_bn3_%d" % level)(net)
            net = layers.ReLU(name="down_relu3_%d" % level)(net)

    for depth in range(DEPTHS - 2, -1, -1):
        filters = int(2 ** depth * FILTERS_ROOT)
        level = depth + 1
        net = layers.Conv2DTranspose(
            filters, KERNEL_SIZE, strides=POOL_SIZE, padding="same", use_bias=False,
            name="up_conv0_%d" % level,
        )(net)
        net = _bn("up_bn0_%d" % level)(net)
        net = layers.ReLU(name="up_relu0_%d" % level)(net)
        net = CropAndConcat(name="crop_concat_%d" % level)([skips[depth], net])
        net = layers.Conv2D(
            filters, KERNEL_SIZE, padding="same", use_bias=False,
            name="up_conv1_%d" % level,
        )(net)
        net = _bn("up_bn1_%d" % level)(net)
        net = layers.ReLU(name="up_relu1_%d" % level)(net)

    logits = layers.Conv2D(
        3, (1, 1), padding="same", use_bias=True, name="output_conv"
    )(net)
    probs = layers.Softmax(axis=-1, name="phasenet_softmax")(logits)
    return Model(inputs=inputs, outputs=probs, name="phasenet_unet")


def _checkpoint_prefix(path):
    path = os.fspath(path)
    if os.path.isdir(path):
        prefix = tf.train.latest_checkpoint(path)
        if prefix is None:
            raise FileNotFoundError(
                "No PhaseNet checkpoint in %s. Expected the official "
                "190703-214543 files (checkpoint, model_95.ckpt.index, "
                "model_95.ckpt.data-00000-of-00001)." % path
            )
        return prefix
    if path.endswith(".index"):
        path = path[: -len(".index")]
    if path.endswith(".meta"):
        path = path[: -len(".meta")]
    if os.path.exists(path + ".index"):
        return path
    raise FileNotFoundError("PhaseNet checkpoint prefix not found: %s" % path)


def _tensors_for_layer(layer):
    """Checkpoint tensor names that feed this Keras layer, in set_weights order."""
    name = layer.name
    if name == "input_conv" or name == "output_conv":
        scope = "Input" if name == "input_conv" else "Output"
        return ["%s/%s/kernel" % (scope, name), "%s/%s/bias" % (scope, name)]
    if name.startswith("down_conv"):
        level = int(name.rsplit("_", 1)[-1])
        depth = level - 1
        return ["DownConv_%d/%s/kernel" % (depth, name)]
    if name.startswith("up_conv"):
        level = int(name.rsplit("_", 1)[-1])
        depth = level - 1
        return ["UpConv_%d/%s/kernel" % (depth, name)]
    if name.startswith("down_bn"):
        level = int(name.rsplit("_", 1)[-1])
        depth = level - 1
        scope = "DownConv_%d/%s" % (depth, name)
    elif name.startswith("up_bn"):
        level = int(name.rsplit("_", 1)[-1])
        depth = level - 1
        scope = "UpConv_%d/%s" % (depth, name)
    elif name == "input_bn":
        scope = "Input/input_bn"
    else:
        return None
    return [
        scope + "/gamma",
        scope + "/beta",
        scope + "/moving_mean",
        scope + "/moving_variance",
    ]


def load_phasenet_weights(model, checkpoint_path):
    """Copy official checkpoint tensors into `model`. Raises if any tensor differs."""
    reader = tf.train.load_checkpoint(_checkpoint_prefix(checkpoint_path))
    loaded = 0
    for layer in model.layers:
        names = _tensors_for_layer(layer)
        if not names:
            continue
        values = []
        for tensor_name in names:
            value = reader.get_tensor(tensor_name)
            values.append(value)
        current = layer.get_weights()
        if len(current) != len(values):
            raise ValueError(
                "Layer %s expects %d tensors, checkpoint has %d (%s)"
                % (layer.name, len(current), len(values), names)
            )
        for tensor_name, have, want in zip(names, current, values):
            if tuple(have.shape) != tuple(want.shape):
                raise ValueError(
                    "Shape mismatch for %s: keras %s vs checkpoint %s"
                    % (tensor_name, have.shape, want.shape)
                )
        layer.set_weights(values)
        loaded += len(values)
    if loaded == 0:
        raise RuntimeError("No PhaseNet weights were loaded from %s" % checkpoint_path)
    return loaded


def build_phasenet_unet_loaded(checkpoint_path):
    model = build_phasenet_unet()
    # Build every weight variable before assigning the checkpoint.
    model(tf.zeros((1, WINDOW_SAMPLES, 1, 3), dtype=tf.float32), training=False)
    load_phasenet_weights(model, checkpoint_path)
    return model


def load_phasenet_model(input_modelP, input_modelS=None, samples=SCMLPICK_SAMPLES):
    """Loader registered as architecture 'phasenet'.

    Same call shape as load_eqcct_model(P, S). PhaseNet is one network, so
    `input_modelS` is ignored. `input_modelP` is the official checkpoint
    directory or the model_95.ckpt prefix.

    `.predict` accepts the scmlpick generator batches named 'input' with shape
    (batch, 6000, 3) and returns (predP, predS), each (batch, 6000, 1).
    Channels are E, N, Z, the same order as the rest of the predictor.
    Normalization is the official sliding mean/std, applied inside the model,
    so the generator must pass the waveform without a second std normalization.
    """
    unet = build_phasenet_unet_loaded(input_modelP)
    waveform = layers.Input(shape=(samples, 3), name="input")
    normalized = OfficialNormalizeLong(name="official_normalize_long")(waveform)
    image = layers.Reshape((samples, 1, 3), name="phasenet_image")(normalized)
    probs = unet(image)
    # Official softmax axis: 0 noise, 1 P, 2 S. Drop the dummy width.
    pred_p = layers.Lambda(lambda p: p[:, :, 0, 1:2], name="picker_P")(probs)
    pred_s = layers.Lambda(lambda p: p[:, :, 0, 2:3], name="picker_S")(probs)
    model = Model(inputs=waveform, outputs=[pred_p, pred_s], name="phasenet_picker")
    model.compile(optimizer="adam", loss=["mse", "mse"])
    return model

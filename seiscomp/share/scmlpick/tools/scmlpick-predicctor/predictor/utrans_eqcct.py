# predictor/utrans_eqcct.py
#
# "Foundation-EQCCT" downstream phase-picking architecture, ported from the
# U-Trans repository's downstream examples:
#   examples/downstream/pwave_eqcct/EQCCT_P_utils.py (+ 01_train.ipynb)
#   examples/downstream/swave_eqcct/EQCCT_S_utils.py (+ 01_train.ipynb)
# See also: Saad, Chen & Alkhalifah (2026), Scientific Reports,
# https://doi.org/10.1038/s41598-026-41454-x (Fig. 3: encoder latent features
# are flattened and concatenated as a 4th input channel to the downstream
# picking network).
#
# This module builds a single combined tf.keras model with ONE shared 3-channel
# waveform input and two outputs (`picker_P`, `picker_S`), exactly mirroring the
# calling convention of `load_eqcct_model()` in predictor.py, so it can be used
# as a drop-in alternative architecture: same `model.predict(...) -> (predP,
# predS)` contract, same preprocessing (6000, 3) windows.
#
# P and S are trained as two fully independent fine-tuned copies of the U-Trans
# encoder (matching how the reference notebooks train/save them separately), so
# each branch below is built from scratch and its weights loaded from its own
# checkpoint file, even though both branches read from the same shared input.

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input, Conv1D, Flatten, Reshape, LayerNormalization, MultiHeadAttention, Add,
    concatenate,
)
from tensorflow.keras.models import Model

from .utrans_layers import encoder_bottleneck, convF1, mlp, StochasticDepth as _FoundationStochasticDepth


# -----------------------------------------------------------------------------
# Picking-head hyperparameters (must match the reference downstream notebooks)
# -----------------------------------------------------------------------------
image_size = 6000
patch_size = 40
num_patches = image_size // patch_size  # 150
projection_dim = 40
num_heads = 4
transformer_units = [projection_dim, projection_dim]
transformer_layers = 4
stochastic_depth_rate = 0.1


class PatchesX(layers.Layer):
    """Conv1D-patchifier for the downstream picking CCT (patch_size=40)."""

    def __init__(self, patch_size, **kwargs):
        super(PatchesX, self).__init__(**kwargs)
        self.patch_size = patch_size

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoderX(layers.Layer):
    """Linear projection + learned positional embedding for picking-head patches."""

    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoderX, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "projection_dim": projection_dim})
        return cfg

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)


class StochasticDepthX(_FoundationStochasticDepth):
    """Alias kept for naming parity with the reference notebooks (StochasticDepthX)."""
    pass


def _picker_branch(bottleneck, raw_input, branch, D1=5):
    """
    Builds one Foundation-EQCCT picking branch (P or S).

    Parameters
    ----------
    bottleneck : Tensor
        U-Trans encoder CCT bottleneck, shape (B, 75, 80) — see
        `utrans_layers.encoder_bottleneck()`.
    raw_input : Tensor
        The shared raw 3-component waveform input, shape (B, 6000, 3).
    branch : {"P", "S"}
        Selects the exact (asymmetric) variant used by the reference
        notebooks: P uses LayerNormalization(epsilon=1e-6) and no extra
        convF1 inside the transformer loop; S uses epsilon=1e-5 and an
        extra convF1 both at the start of the loop and after attention.
        This mirrors the existing plain-EQCCT create_cct_modelP/S asymmetry
        in predictor.py, which the U-Trans downstream notebooks preserve.
    D1 : int
        Unused here (kept for signature symmetry); the decoder-feature
        convF1 widths are fixed at 80 per the reference notebooks.

    Returns
    -------
    Tensor
        Sigmoid probability curve, shape (B, 6000, 1), named
        "picker_P"/"picker_S".
    """
    if branch not in ("P", "S"):
        raise ValueError(f"branch must be 'P' or 'S', got {branch!r}")

    # Decoder of the foundation encoder: (B,75,80) -> flatten -> (B,6000,1)
    x = convF1(bottleneck, 80, 13, 0.1)
    x = convF1(x, 80, 13, 0.1)
    x = convF1(x, 80, 13, 0.1)
    x = Flatten()(x)
    x = Reshape((6000, 1))(x)

    # Ready to concatenate: ends up as a 4th input channel to the picker.
    x = concatenate([x, raw_input])

    inputs1 = convF1(x, 10, 11, 0.1)
    inputs1 = convF1(inputs1, 20, 11, 0.1)
    inputs1 = convF1(inputs1, 40, 11, 0.1)

    inputreshaped = Reshape((6000, 1, 40))(inputs1)
    patches = PatchesX(patch_size)(inputreshaped)
    encoded_patches = PatchEncoderX(num_patches, projection_dim)(patches)

    dpr = [v for v in np.linspace(0, stochastic_depth_rate, transformer_layers)]
    epsilon = 1e-6 if branch == "P" else 1e-5

    for i in range(transformer_layers):
        if branch == "S":
            encoded_patches = convF1(encoded_patches, 40, 11, 0.1)

        x1 = LayerNormalization(epsilon=epsilon)(encoded_patches)
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        if branch == "S":
            attention_output = convF1(attention_output, 40, 11, 0.1)

        attention_output = StochasticDepthX(dpr[i])(attention_output)
        x2 = Add()([attention_output, encoded_patches])

        x3 = LayerNormalization(epsilon=epsilon)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        x3 = StochasticDepthX(dpr[i])(x3)
        encoded_patches = Add()([x3, x2])

    representation = LayerNormalization(epsilon=epsilon)(encoded_patches)
    representation = Reshape((6000, 1))(representation)

    name = "picker_P" if branch == "P" else "picker_S"
    logit = Conv1D(
        1, 3, strides=1, padding="same", activation="sigmoid",
        kernel_initializer="he_normal", name=name,
    )(representation)
    return logit


def build_utrans_eqcct_graph(input_shape=(6000, 3), D1=5):
    """
    Builds the (untrained) combined Foundation-EQCCT graph: one shared input,
    two independent U-Trans-encoder + picking-head branches (P and S).

    Returns
    -------
    inputs, logitp, logits : Tensors
    """
    inputs = Input(shape=input_shape, name="input")

    bottleneck_p = encoder_bottleneck(inputs, D1)
    logitp = _picker_branch(bottleneck_p, inputs, branch="P", D1=D1)

    bottleneck_s = encoder_bottleneck(inputs, D1)
    logits = _picker_branch(bottleneck_s, inputs, branch="S", D1=D1)

    return inputs, logitp, logits


def load_utrans_eqcct_model(input_modelP, input_modelS, input_shape=(6000, 3), D1=5):
    """
    Builds the Foundation-EQCCT architecture and loads fine-tuned weights.

    Drop-in counterpart to `load_eqcct_model()`: same call signature
    (P weights path, S weights path), same output contract — a compiled
    model whose `.predict(...)` returns `(predP, predS)`.

    Each weights file (`input_modelP` / `input_modelS`) is expected to be a
    *combined* checkpoint containing both the fine-tuned U-Trans encoder
    branch and its picking head for that phase — i.e. exactly what the
    reference notebooks (examples/downstream/{pwave,swave}_eqcct/01_train.ipynb)
    save via `ModelCheckpoint(save_weights_only=True)` on the full
    `Model(inputs=[modeloriginal.input], outputs=logitp)` graph. It does NOT
    need `UTrans_Foundation.h5` at inference time: that file is only used to
    initialize training, and the fine-tuned encoder weights it produced are
    already baked into the saved P/S checkpoint.

    Parameters
    ----------
    input_modelP : str
        Path to the fine-tuned Foundation-EQCCT P-picker weights (.h5).
    input_modelS : str
        Path to the fine-tuned Foundation-EQCCT S-picker weights (.h5).
    input_shape : tuple, default=(6000, 3)
    D1 : int, default=5
        Base U-Net filter width (must match training; 5 in the released
        foundation checkpoint).

    Returns
    -------
    tf.keras.Model
        Compiled model with outputs [logitp, logits].
    """
    inputs, logitp, logits = build_utrans_eqcct_graph(input_shape=input_shape, D1=D1)

    modelP = Model(inputs=[inputs], outputs=[logitp])
    modelS = Model(inputs=[inputs], outputs=[logits])
    model = Model(inputs=[inputs], outputs=[logitp, logits])

    sgd = tf.keras.optimizers.Adam()
    model.compile(optimizer=sgd, loss=["binary_crossentropy", "binary_crossentropy"])

    modelP.load_weights(input_modelP)
    modelS.load_weights(input_modelS)

    return model

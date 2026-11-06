# predictor/utrans_layers.py
#
# Ported from the U-Trans foundation model repository (utrans/layers.py):
#   https://github.com/omarmohamed15/U-Trans-Foundation-For-Seismic-Waveform-Representation
#
# This module rebuilds the exact U-Net + Compact Convolutional Transformer (CCT)
# backbone used to pretrain U-Trans (Saad, Chen & Alkhalifah, 2026, Scientific
# Reports, https://doi.org/10.1038/s41598-026-41454-x). It must stay in sync with
# the architecture used to produce any UTrans_*.h5 checkpoint, since tf.keras
# `load_weights()` matches weighted layers positionally.
#
# `encoder_bottleneck()` is new: it is not part of the upstream repo. It factors
# out the first half of `UNET()` (encoder + CCT bottleneck only, stopping at the
# (B, 75, 80) latent token representation) so downstream picking branches can be
# attached directly to the bottleneck without also building (and then discarding)
# the unused U-Net decoder/upsampling path.

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, UpSampling1D, Dropout, BatchNormalization,
    Activation, Add, concatenate,
)


# -----------------------------------------------------------------------------
# CCT (Transformer) Hyperparameters (must match training)
# -----------------------------------------------------------------------------
stochastic_depth_rate = 0.1
positional_emb = False
conv_layers = 1

projection_dim = 80
num_heads = 4
transformer_units = [projection_dim, projection_dim]
transformer_layers = 4


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class StochasticDepth(layers.Layer):
    """DropPath for residual branches. Identity at inference time."""

    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = float(drop_prop)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_prop": self.drop_prob})
        return cfg

    def call(self, x, training=None):
        if training:
            keep_prob = 1.0 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class CCTTokenizer1(layers.Layer):
    """Conv1D tokenizer used at the U-Net bottleneck (foundation CCT)."""

    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=(1, 1, 1, 1),
        num_conv_layers=conv_layers,
        num_output_channels=None,
        positional_emb=positional_emb,
        **kwargs,
    ):
        super(CCTTokenizer1, self).__init__(**kwargs)

        if num_output_channels is None:
            num_output_channels = [int(projection_dim)] * 8

        self.conv_model = tf.keras.Sequential(name="cct_tokenizer_conv")
        for i in range(num_conv_layers):
            self.conv_model.add(
                layers.Conv1D(
                    num_output_channels[i],
                    kernel_size,
                    stride,
                    padding="same",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )

        self.positional_emb = positional_emb

    def call(self, images):
        outputs = self.conv_model(images)
        return outputs

    def positional_embedding(self, image_size):
        if self.positional_emb:
            dummy_inputs = tf.ones((1, image_size, 1))
            dummy_outputs = self.call(dummy_inputs)
            sequence_length = int(dummy_outputs.shape[1])
            proj_dim = int(dummy_outputs.shape[-1])
            embed_layer = layers.Embedding(input_dim=sequence_length, output_dim=proj_dim)
            return embed_layer, sequence_length
        return None


def create_cct_model1(inputs):
    """Foundation CCT bottleneck: tokenize + `transformer_layers` Transformer blocks."""
    cct_tokenizer = CCTTokenizer1()
    encoded_patches = cct_tokenizer(inputs)

    if positional_emb:
        raise ValueError("positional_emb=True is not supported unless image_size is defined.")

    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    for i in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    return representation


def convF1(inpt, D1, fil_ord, Dr):
    """Residual Conv1D block used throughout U-Trans (encoder and decoder)."""
    filters = int(inpt.shape[-1])

    pre = Conv1D(filters, fil_ord, padding="same", kernel_initializer="he_normal")(inpt)
    pre = BatchNormalization()(pre)
    pre = Activation(tf.nn.gelu)(pre)

    inf = Conv1D(filters, fil_ord, padding="same", kernel_initializer="he_normal")(pre)
    inf = BatchNormalization()(inf)
    inf = Activation(tf.nn.gelu)(inf)
    inf = Add()([inf, inpt])

    inf1 = Conv1D(D1, fil_ord, padding="same", kernel_initializer="he_normal")(inf)
    inf1 = BatchNormalization()(inf1)
    inf1 = Activation(tf.nn.gelu)(inf1)

    return Dropout(Dr)(inf1)


def encoder_bottleneck(inputs, D1=5):
    """
    U-Net encoder + CCT bottleneck only (no decoder/upsampling path).

    This is the exact encoder half of `UNET()` below, stopping at
    `drop5 = create_cct_model1(conv5)`. It reproduces what
    `modeloriginal.layers[63].output` returns in the reference
    `utrans/foundation.py` (verified empirically against the shipped
    `UTrans_Foundation.h5` checkpoint: shape (B, 75, 80) for a (6000, C) input).

    Parameters
    ----------
    inputs : Tensor
        Waveform tensor, shape (B, 6000, C).
    D1 : int
        Base channel width (5 in the released checkpoint).

    Returns
    -------
    Tensor
        CCT bottleneck / latent token representation, shape (B, 75, 80).
    """
    D2 = int(D1 * 2)
    D3 = int(D2 * 2)
    D4 = int(D3 * 2)
    D5 = int(D4 * 2)

    conv1 = Conv1D(D1, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    conv1 = Conv1D(D1, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(D2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    conv2 = Conv1D(D2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(D3, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv3 = Conv1D(D3, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(D4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv4 = Conv1D(D4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv44 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
    conv44 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv44)
    pool44 = MaxPooling1D(pool_size=5)(conv44)

    conv5 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool44)
    conv5 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5)

    bottleneck = create_cct_model1(conv5)
    return bottleneck


def UNET(inputs, D1):
    """
    Full 1D U-Net backbone with CCT bottleneck (encoder + decoder).

    Kept for parity with the upstream U-Trans repository and for anyone who
    needs the full reconstruction decoder (e.g. to load `UTrans_Foundation.h5`
    end-to-end for retraining/inspection). scmlpick's downstream picking
    branches use `encoder_bottleneck()` above instead, since the decoder path
    is not part of the fine-tuned picking checkpoints.
    """
    D2 = int(D1 * 2)
    D3 = int(D2 * 2)
    D4 = int(D3 * 2)
    D5 = int(D4 * 2)

    conv1 = Conv1D(D1, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    conv1 = Conv1D(D1, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(D2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    conv2 = Conv1D(D2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(D3, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv3 = Conv1D(D3, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(D4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv4 = Conv1D(D4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv44 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
    conv44 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv44)
    pool44 = MaxPooling1D(pool_size=5)(conv44)

    conv5 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool44)
    conv5 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5)

    drop5 = create_cct_model1(conv5)

    up66 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling1D(size=5)(drop5))
    merge66 = concatenate([pool4, up66], axis=-1)
    conv66 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge66)
    conv66 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv66)

    up6 = Conv1D(D4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling1D(size=2)(conv66))
    merge6 = concatenate([conv4, up6], axis=-1)
    conv6 = Conv1D(D4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge6)
    conv6 = Conv1D(D4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv6)

    up7 = Conv1D(D3, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling1D(size=2)(conv6))
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv1D(D3, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge7)
    conv7 = Conv1D(D3, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)

    up8 = Conv1D(D2, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling1D(size=2)(conv7))
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv1D(D2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge8)
    conv8 = Conv1D(D2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv8)

    up9 = Conv1D(D1, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling1D(size=2)(conv8))
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv1D(D1, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge9)
    conv9 = Conv1D(D1, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)

    return conv9

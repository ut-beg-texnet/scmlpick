# Inference precision for every scmlpick picker.
#
# RAPID (https://github.com/Costaki33/RAPID) measured bfloat16 as the fastest
# precision that stayed close to float32 picks. Float16 lost: EQTransformer
# overflows, and other pickers lost quality. scmlpick therefore offers bf16
# and fp32.
#
# The pickers here are Keras, not PyTorch. RAPID does model.to(bfloat16).
# The Keras equivalent that keeps the official float32 weights and runs the
# layer math in bfloat16 is the mixed_bfloat16 policy. It has to be selected
# before the network is built. A file opened with load_model() still records
# float32 on each layer, so those layers are pointed at the same policy
# before the first predict().

_PRECISION = "fp32"


def normalize_precision(precision):
    name = str(precision or "bf16").strip().lower()
    if name in ("bf16", "bfloat16"):
        return "bf16"
    if name in ("fp32", "float32"):
        return "fp32"
    raise ValueError(
        "ray.precision must be 'bf16' or 'fp32', got %r." % (precision,)
    )


def begin_precision(precision):
    """Select the compute dtype. Call this before building or loading a model."""
    global _PRECISION
    _PRECISION = normalize_precision(precision)
    import tensorflow as tf

    policy_name = "mixed_bfloat16" if _PRECISION == "bf16" else "float32"
    tf.keras.mixed_precision.set_global_policy(policy_name)
    try:
        import tf_keras
        tf_keras.mixed_precision.set_global_policy(policy_name)
    except Exception:
        pass
    return _PRECISION


def _clone_without_saved_dtype(layer):
    """Rebuild one layer so it picks up the active mixed_bfloat16 policy.

    The official file stores dtype=float32 in every layer config. A layer
    constructed from that config ignores the global policy. Dropping the
    saved dtype lets a new layer follow mixed_bfloat16, which casts the
    float32 weights to bfloat16 for the arithmetic and leaves the stored
    weights unchanged.
    """
    config = layer.get_config()
    config.pop("dtype", None)
    return layer.__class__.from_config(config)


def apply_loaded_policy(model):
    """Return a bfloat16 compute copy of a model loaded from a Keras file.

    Layers created in Python after begin_precision() already compute in
    bfloat16. load_model() does not, so EQTransformer is rebuilt here.
    With precision fp32 this returns the same model.
    """
    if _PRECISION != "bf16" or model is None:
        return model
    import tf_keras

    cloned = tf_keras.models.clone_model(
        model, clone_function=_clone_without_saved_dtype
    )
    cloned.set_weights(model.get_weights())
    return cloned

# Persistent Ray model actor, the RAPID Model-Actor strategy.
#
# The actor loads one picker when the process starts and reuses it for every
# station. That is the execution mode that avoids reloading TensorFlow on
# each station. The other mode (ray.execution = task) still loads the model
# inside each task; that mode recycles the worker with max_calls instead.

import ray


@ray.remote
class PickerModelActor:
    """One process, one loaded picker, many stations."""

    def __init__(self, architecture, p_model, s_model, precision):
        from .precision import begin_precision
        from .predictor import MODEL_LOADERS

        self.architecture = architecture
        self.precision = begin_precision(precision)
        try:
            loader = MODEL_LOADERS[architecture]
        except KeyError as exc:
            raise ValueError(
                "Unknown picker architecture %r. Valid options: %s"
                % (architecture, sorted(MODEL_LOADERS))
            ) from exc
        self.model = loader(p_model, s_model)

    def ready(self):
        return {"architecture": self.architecture, "precision": self.precision}

    def predict_station(self, task, stream_ref, params_ref):
        from .predictor import mseed_predictor

        try:
            if isinstance(stream_ref, ray.ObjectRef):
                stream = ray.get(stream_ref)
            else:
                stream = stream_ref
            if isinstance(params_ref, ray.ObjectRef):
                params = ray.get(params_ref)
            else:
                params = params_ref
            picks = mseed_predictor(
                stream=stream,
                filterShift=params["filterShift"],
                wfinfo=task,
                files=params["files"],
                P_threshold=params["P_threshold"],
                S_threshold=params["S_threshold"],
                normalization_mode=params["normalization_mode"],
                overlap=params["overlap"],
                batch_size=params["batch_size"],
                overwrite=params["overwrite"],
                p_model=params["p_model"],
                s_model=params["s_model"],
                architecture=params["architecture"],
                numCPUs=params["numCPUs"],
                gpu_id=params["gpu_id"],
                gpu_limit=params["gpu_limit"],
                stations_filters=params.get("stations_filters", params.get("df_filters")),
                playback=params["playback"],
                model=self.model,
                precision=self.precision,
                detection_threshold=params.get("detection_threshold", 0.3),
            )
            picks["ok"] = True
            return picks
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def create_picker_actors(n, architecture, p_model, s_model, precision):
    """Start `n` actors and wait until each one has loaded its picker."""
    n = max(1, int(n))
    actors = [
        PickerModelActor.options(num_cpus=1).remote(
            architecture, p_model, s_model, precision
        )
        for _ in range(n)
    ]
    ray.get([actor.ready.remote() for actor in actors])
    return actors

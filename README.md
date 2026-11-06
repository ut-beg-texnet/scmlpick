# scmlpick

scmlpick is a SeisComP module for real-time seismic phase picking. EQCCT is the default architecture. The same program also runs UTrans (`utrans_eqcct`), PhaseNet, and EQTransformer. The architecture is selected with `eqcct.architecture`. One process uses one architecture. Profiles inside that process split stations, messaging groups, and regions.

The full configuration reference is at <https://ut-beg-texnet.github.io/scmlpick/apps/scmlpick.html>.

## Installation

SeisComP is installed separately. This repository adds the module, the predictor, and the four sets of model weights.

### 1. Install SeisComP

Install SeisComP 4.0.0 or newer. This module is run on SeisComP 6.

<https://docs.seiscomp.de/>

### 2. Clone this repository

```bash
git clone https://github.com/ut-beg-texnet/scmlpick.git
cd scmlpick
```

### 3. Copy the module into the SeisComP tree

```bash
rsync -a seiscomp/ "$SEISCOMP_ROOT/"
```

`$SEISCOMP_ROOT` is usually `/opt/seiscomp`. Copy the `seiscomp/` directory, not the repository root.

`seiscomp/etc/scmlpick.cfg` in this repository is the configuration used on one production host. On another computer, edit the messaging connection, the subscriptions, the model paths, and the profiles before starting the module. Station bindings and region polygons are not in this repository.

The weights are under `seiscomp/share/scmlpick/model/` and are included by the copy above.

### 4. Install the Python environment used by the service

`seiscomp/bin/scmlpick` starts `/opt/seiscomp/share/scmlpick/pyenv/bin/python` when that file exists. Python 3.10, 3.11, or 3.12 is required, on glibc Linux.

```bash
python3 -m venv /opt/seiscomp/share/scmlpick/pyenv
/opt/seiscomp/share/scmlpick/pyenv/bin/python -m pip install --upgrade pip
/opt/seiscomp/share/scmlpick/pyenv/bin/python -m pip install .
```

Run the last command from the repository root. It installs the predictor and the libraries in the root `setup.py`, including NumPy 1.26, pandas 2.x, SciPy 1.15, ObsPy, Ray 2.59, TensorFlow 2.17.0, and tf-keras 2.17.0.

If SeisComP is not installed under `/opt/seiscomp`, create the environment at `$SEISCOMP_ROOT/share/scmlpick/pyenv` and point `_PY` at the top of `seiscomp/bin/scmlpick` to that interpreter.

`python3 setup.py --make-env` creates `.venv` inside the repository for work on the predictor. The service does not use it.

`seiscomp/share/scmlpick/tools/scmlpick-predicctor/setup.py` installs the same package when you are already in the deployed SeisComP tree (`pip install -e .` from that directory). A new installation uses the `setup.py` at the repository root.

### 5. Enable and start the module

```bash
seiscomp update-config scmlpick
seiscomp enable scmlpick
seiscomp start scmlpick
```

## Architectures

| `eqcct.architecture` | Weights |
|---|---|
| `eqcct` | `model/ModelPS/test_trainer_024.h5` and `test_trainer_021.h5` |
| `utrans_eqcct` | `model/UTrans/test_trainer_P_FoundationEQCCT_026.h5` and `test_trainer_S_FoundationEQCCT_025.h5` |
| `phasenet` | `model/PhaseNet/190703-214543` as `Pmodel`. `Smodel` is ignored. |
| `eqtransformer` | `model/EQTransformer/EqT_original_model.h5` as `Pmodel`. `Smodel` is ignored. |

Set `eqcct.gpuID = -1` to run on CPU. `ray.precision` is `bf16` by default, or `fp32`.

To run more than one architecture, start one process per architecture. Each process is a symbolic link to `bin/scmlpick` with its own `etc/<name>.cfg` and `etc/key/<name>/` bindings.

## Documentation source

The pages at <https://ut-beg-texnet.github.io/scmlpick/> are built from `docs/` and published on the `gh-pages` branch. `docs/apps/scmlpick.rst` is the module page. `docs/descriptions/scmlpick.xml` is the configuration reference and matches `seiscomp/etc/descriptions/scmlpick.xml`.

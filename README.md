# scmlpick

**scmlpick** is a SeisComP module integrating the deep learning model EQCCT for real-time seismic phase picking. It enables accurate and low-latency detection in operational networks by combining waveform feature extraction with optimized parallel processing.

This version also runs UTrans, PhaseNet, and EQTransformer. The architecture is selected in the module configuration.

For full documentation, configuration guidelines, and usage examples, please refer to the official module documentation:

👉 [Official Documentation](https://ut-beg-texnet.github.io/scmlpick/)

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
<!-- - [Usage](#usage)
- [Data](#data)
- [Reproducibility](#reproducibility)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact) -->

## Project Description

The scmlpick module implements a real-time seismic phase picking solution that integrates deep learning-based detection into operational seismic monitoring workflows. It leverages the EQCCT algorithm — a model based on the Compact Convolutional Transformer architecture — to perform robust and efficient phase picking directly within the SeisComP environment.

The scientific motivation for this development arises from the increasing need to reduce latency and improve pick accuracy in real-time earthquake monitoring systems, particularly for dense regional seismic networks operating under computational constraints. Traditional phase picking methods often struggle with noisy environments or require manual intervention, limiting their applicability for automated, near-real-time applications.

By embedding the EQCCT model into SeisComP, scmlpick allows near-real-time generation of high-quality phase picks with delays typically under 60 seconds after the actual phase arrival, representing a substantial improvement compared to conventional playback-based approaches that can introduce latencies of several minutes. The system has been specifically designed to support operational requirements of networks like TexNet (Texas Seismological Network), where timely detection is critical for rapid response and seismic hazard assessment in areas of induced and natural seismicity.

The scmlpick module supports multiple pipelines within a single execution, enabling independent processing streams for diverse operational needs while optimizing resource utilization through parallelization strategies. The architecture is highly configurable, allowing customization of station/channel subsets, EQCCT model parameters, probability thresholds, and processing intervals.

This tool represents a significant step forward in integrating state-of-the-art machine learning techniques into established seismic monitoring frameworks, enhancing the capacity for real-time earthquake detection, catalog building, and early warning systems.

## Features

- ✅ Seamless integration of the **EQCCT** deep learning phase picker into **SeisComP** for real-time seismic monitoring.
- ✅ Utilizes **Compact Convolutional Transformer (CCT)** architecture, combining convolutional layers and transformer encoders to extract robust time-series features from waveform data.
- ✅ Supports **simultaneous execution** of multiple independent processing pipelines within a single instance, enabling differentiated operational workflows.
- ✅ **Highly configurable** via user-defined station/channel lists, model selection, probability thresholds, and processing frequencies.
- ✅ **Real-time phase picking** with latencies typically below **60 seconds** after phase arrival — significantly faster than playback-based methods.
- ✅ Optimized **parallelization framework** for efficient resource usage on computationally constrained systems.
- ✅ Designed to operate on **regional and dense seismic networks** with mixed sensor types and varying data quality.
- ✅ Fully compatible with **EQCCT pretrained models**, including those trained on global datasets (e.g., **STEAD**) and regional datasets (e.g., **TexNet**).
- ✅ Facilitates integration into **cataloging, early warning, and rapid response systems** via standard SeisComP messaging groups.
- ✅ Reduces manual intervention by improving phase pick **reliability in noisy or complex signal environments**.


## Installation

SeisComP is installed separately. This repository adds the scmlpick module and the Python predictor.

### 1. Install SeisComP

Install SeisComP 4.0.0 or newer. This module is run on SeisComP 6. Follow the official instructions:

https://docs.seiscomp.de/

### 2. Clone this repository

```bash
git clone https://github.com/ut-beg-texnet/scmlpick.git
cd scmlpick
```

### 3. Copy the module into the SeisComP tree

The module files are under `seiscomp/` in this repository. Copy that directory onto the SeisComP root:

```bash
rsync -a seiscomp/ "$SEISCOMP_ROOT/"
```

`$SEISCOMP_ROOT` is usually `/opt/seiscomp`.

Copy the `seiscomp/` directory, not the repository root. Copying the repository root places the binary at `$SEISCOMP_ROOT/seiscomp/bin/scmlpick`, and SeisComP does not look there.

`seiscomp/etc/scmlpick.cfg` in this repository is the configuration used on one production host. On another computer, edit the messaging connection, the subscriptions, the model paths, and the profile groups before starting the module.

Station bindings, filter files, and region polygons are not stored in this repository. Copy those into `$SEISCOMP_ROOT/share/scmlpick/tools/` on the new computer.

The model weights for EQCCT, UTrans, PhaseNet, and EQTransformer are under `seiscomp/share/scmlpick/model/` and are included by the copy above.

### 4. Install the Python environment used by the service

`seiscomp/bin/scmlpick` starts `/opt/seiscomp/share/scmlpick/pyenv/bin/python` when that file exists. Create that environment and install this repository into it. Python 3.10, 3.11, or 3.12 is required, on a glibc Linux system.

```bash
python3 -m venv /opt/seiscomp/share/scmlpick/pyenv
/opt/seiscomp/share/scmlpick/pyenv/bin/python -m pip install --upgrade pip
/opt/seiscomp/share/scmlpick/pyenv/bin/python -m pip install .
```

Run the last command from the repository root. It installs the predictor and the libraries listed in the root `setup.py`, including NumPy 1.26, ObsPy, Ray, TensorFlow 2.17.0, and tf-keras 2.17.0.

If SeisComP is not installed under `/opt/seiscomp`, create the environment at `$SEISCOMP_ROOT/share/scmlpick/pyenv` and point the `_PY` path at the top of `seiscomp/bin/scmlpick` to that interpreter.

`python3 setup.py --make-env`, also run from the repository root, creates `.venv` inside the repository. That environment is for working on the predictor. The service does not use it.

`seiscomp/share/scmlpick/tools/scmlpick-predicctor/setup.py` installs the same package when the current directory is already the deployed SeisComP tree (`pip install -e .` from that directory). A new installation uses the `setup.py` at the repository root.

### 5. Enable and start the module

```bash
seiscomp update-config scmlpick
seiscomp enable scmlpick
seiscomp start scmlpick
```

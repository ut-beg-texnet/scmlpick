# sceqcct

sceqcct is a SeisComP module integrating the deep learning model EQCCT for real-time seismic phase picking. It enables accurate and low-latency detection in operational networks by combining waveform feature extraction with optimized parallel processing.

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

The sceqcct module implements a real-time seismic phase picking solution that integrates deep learning-based detection into operational seismic monitoring workflows. It leverages the EQCCT algorithm — a model based on the Compact Convolutional Transformer architecture — to perform robust and efficient phase picking directly within the SeisComP environment.

The scientific motivation for this development arises from the increasing need to reduce latency and improve pick accuracy in real-time earthquake monitoring systems, particularly for dense regional seismic networks operating under computational constraints. Traditional phase picking methods often struggle with noisy environments or require manual intervention, limiting their applicability for automated, near-real-time applications.

By embedding the EQCCT model into SeisComP, sceqcct allows near-real-time generation of high-quality phase picks with delays typically under 60 seconds after the actual phase arrival, representing a substantial improvement compared to conventional playback-based approaches that can introduce latencies of several minutes. The system has been specifically designed to support operational requirements of networks like TexNet (Texas Seismological Network), where timely detection is critical for rapid response and seismic hazard assessment in areas of induced and natural seismicity.

The sceqcct module supports multiple pipelines within a single execution, enabling independent processing streams for diverse operational needs while optimizing resource utilization through parallelization strategies. The architecture is highly configurable, allowing customization of station/channel subsets, EQCCT model parameters, probability thresholds, and processing intervals.

This tool represents a significant step forward in integrating state-of-the-art machine learning techniques into established seismic monitoring frameworks, enhancing the capacity for real-time earthquake detection, catalog building, and early warning systems.

## Features

- Seamless integration of the EQCCT deep learning phase picker into SeisComP for real-time seismic monitoring.
- Utilizes Compact Convolutional Transformer (CCT) architecture combining convolutional layers and transformer encoders to extract robust time-series features from waveform data.
- Supports simultaneous execution of multiple independent processing pipelines within a single instance, enabling differentiated operational workflows.
- Highly configurable through user-defined station/channel lists, model selection, probability thresholds, and processing frequencies.
- Real-time phase picking with latencies typically below 60 seconds after phase arrival, significantly outperforming conventional playback-based methods.
- Optimized parallelization framework allowing efficient resource usage on computationally constrained systems.
- Designed to operate on regional and dense seismic networks with mixed sensor types and variable data quality.
- Full compatibility with EQCCT pretrained models, including models trained on global datasets (e.g., STEAD) and regional datasets (e.g., TexNet).
- Facilitates integration into cataloging, early warning, and rapid response systems through standard SeisComP messaging groups.
- Reduces manual intervention by improving phase pick reliability in noisy or complex signal environments.

## Installation

### Prerequisites

The following software must be installed and properly configured prior to installing and executing this module:

1️⃣ SeisComP System
SeisComP (required)

Version: ≥ 4.0.0 (strongly recommended: ≥ 6.0.0)

The software is fully compatible with SeisComP releases starting from version 4.0.0. Versions 6.0.0 and higher are recommended for optimal stability and performance.

2️⃣ Operating System
Linux (validated on Ubuntu 22.04 LTS)

3️⃣ Programming Language
Python ≥ 3.10
Note: Python versions prior to 3.10 (e.g., 3.6) are not supported due to incompatibilities with the ray library.

4️⃣ Python Packages
The following Python packages are required:

ray

numpy==1.26.4
Note: Versions >= 2.0 may lead to compatibility issues with SeisComP.

pandas

obspy

tensorflow

silence_tensorflow

It is strongly recommended to install these packages inside a dedicated virtual environment (e.g., Conda) to ensure compatibility.

Predictor Installation

The predictor module must be installed after configuring all prerequisites. The installation procedure depends on whether a virtual environment is being used.

➔ If NOT using a virtual environment:

cd $SEISCOMP_ROOT/share/sceqcct/tools/sceqcct-predicctor

pip3 install -e .

➔ If using a Conda or other virtual environment:

 - Activate your environment

conda activate sceqcct  # Replace 'sceqcct' with your actual environment name

 - Navigate to the predictor directory

cd $SEISCOMP_ROOT/share/sceqcct/tools/sceqcct-predicctor

 - Install the predictor module

pip install -e .

Note: The installation must be performed within the activated environment to ensure that dependencies are correctly registered.

### Installation Steps

Follow the steps below to fully install and configure the software:

Step 1: Install SeisComP

The SeisComP system must be installed prior to installing this module.

Download and install SeisComP version ≥ 4.0.0 (strongly recommended: version ≥ 6.0.0) following the official installation instructions:

https://docs.seiscomp.de/

Verify that SeisComP is properly installed and configured in your system.

Step 2: Clone the Repository

Clone the project repository into any local directory:

git clone https://github.com/your-username/your-project.git

(Replace with the actual repository URL when publicly available.)

Step 3: Deploy the Code into SeisComP Installation Directory

Copy all necessary files into your SeisComP installation using rsync to preserve the directory structure:

rsync -av /path/to/your-cloned-repository/ /path/to/your-seiscomp-installation/

Replace /path/to/your-cloned-repository/ with the absolute path of your cloned repository.

Replace /path/to/your-seiscomp-installation/ with the root directory of your SeisComP installation (typically $SEISCOMP_ROOT/).

⚠ This step will integrate the new module into the SeisComP environment while preserving permissions and directory structure.

Step 4: Install Python Dependencies
It is strongly recommended to install all dependencies inside a dedicated Conda environment to ensure full compatibility.

 - Create a new environment (optional)

conda create -n sceqcct python=3.10

 - Activate the environment

conda activate sceqcct

 - Install required packages

pip install ray

pip install numpy==1.26.4

pip install pandas

pip install obspy

pip install tensorflow

pip install silence_tensorflow

Step 5: Install Predictor Module

The predictor component must be installed manually after configuring all dependencies. The procedure differs slightly depending on whether you are using a virtual environment:

➔ If NOT using a virtual environment:

cd $SEISCOMP_ROOT/share/sceqcct/tools/sceqcct-predicctor

pip3 install -e .

➔ If using a Conda or other virtual environment:

 - Activate your environment

conda activate sceqcct  # Replace 'sceqcct' with your actual environment name

 - Navigate to the predictor directory

cd $SEISCOMP_ROOT/share/sceqcct/tools/sceqcct-predicctor

 - Install the predictor module
pip install -e .
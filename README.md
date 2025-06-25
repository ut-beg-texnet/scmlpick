# sceqcct

**sceqcct** is a SeisComP module integrating the deep learning model EQCCT for real-time seismic phase picking. It enables accurate and low-latency detection in operational networks by combining waveform feature extraction with optimized parallel processing.

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

## 🛠️ Prerequisites

Before installing and running this module, ensure the following components are properly installed and configured:

---

### 1️⃣ SeisComP System

- **Required**: [SeisComP](https://www.seiscomp.de/)
- **Minimum version**: `4.0.0`  
- **Recommended**: `6.0.0` or higher for improved stability and performance.

> ✅ Fully compatible with SeisComP releases ≥ 4.0.0

---

### 2️⃣ Operating System

- **Tested on**: `Linux`  
- **Validated on**: `Ubuntu 22.04 LTS`

---

### 3️⃣ Programming Language

- **Python ≥ 3.10**

> ⚠️ Python versions prior to 3.10 (e.g., 3.6) are **not supported** due to incompatibilities with the `ray` library.

---

### 4️⃣ Required Python Packages

Ensure the following Python packages are installed:

```bash
ray
numpy==1.26.4     # ⚠️ Versions ≥ 2.0 may cause compatibility issues with SeisComP
pandas
obspy
tensorflow
silence_tensorflow
```

> 💡 **Tip**: It is strongly recommended to install these packages inside a dedicated virtual environment (e.g., Conda) to avoid dependency conflicts.

---

## Predictor Installation

The `sceqcct-predicctor` module must be installed **after** all prerequisites have been configured. The installation procedure depends on whether a virtual environment is being used.

---

### ➤ If **NOT** using a virtual environment:

```bash
cd $SEISCOMP_ROOT/share/sceqcct/tools/sceqcct-predicctor
pip3 install -e .
```

---

### ➤ If using a **Conda or other virtual environment**:

1. **Activate your environment**  
   _(Replace `sceqcct` with your actual environment name)_

   ```bash
   conda activate sceqcct
   ```

2. **Navigate to the predictor directory**

   ```bash
   cd $SEISCOMP_ROOT/share/sceqcct/tools/sceqcct-predicctor
   ```

3. **Install the predictor module**

   ```bash
   pip install -e .
   ```

> 🔒 **Note**: Installation must be performed *within the activated environment* to ensure all dependencies are correctly registered.


## Installation Steps

Follow the steps below to fully install and configure the software:

---

### Step 1: Install SeisComP

The SeisComP system must be installed prior to installing this module.

- **Minimum required version**: `4.0.0`  
- **Strongly recommended**: `6.0.0` or higher

📘 Follow the official installation instructions:  
[https://docs.seiscomp.de/](https://docs.seiscomp.de/)

✅ Once installed, verify that SeisComP is properly configured and accessible in your environment.

---

### Step 2: Clone the Repository

Clone the project repository into any local directory:

```bash
git clone https://github.austin.utexas.edu/texnet/eqcct-dev.git
```

> 📝 Replace the URL with the public repository link when available.

---

### Step 3: Deploy the Code into SeisComP Installation Directory

Copy all necessary files into your SeisComP installation using `rsync` to preserve the directory structure:

```bash
rsync -av /path/to/your-cloned-repository/ /path/to/your-seiscomp-installation/
```

- Replace `/path/to/your-cloned-repository/` with the absolute path to your cloned repository.
- Replace `/path/to/your-seiscomp-installation/` with your SeisComP root (typically `$SEISCOMP_ROOT/`).

> ⚠️ This step integrates the module into the SeisComP environment, preserving file structure and permissions.

---

### Step 4: Install Python Dependencies

It is **strongly recommended** to use a dedicated Conda environment for installing dependencies.

1. **Create a new environment (optional)**

   ```bash
   conda create -n sceqcct python=3.10
   ```

2. **Activate the environment**

   ```bash
   conda activate sceqcct
   ```

3. **Install required packages**

   ```bash
   pip install ray
   pip install numpy==1.26.4      # ⚠️ Avoid numpy ≥ 2.0 to maintain compatibility
   pip install pandas
   pip install obspy
   pip install tensorflow
   pip install silence_tensorflow
   ```

---

###  Step 5: Install Predictor Module

The `sceqcct-predicctor` component must be installed **after all dependencies** have been configured.

---

#### ➤ If **NOT** using a virtual environment:

```bash
cd $SEISCOMP_ROOT/share/sceqcct/tools/sceqcct-predicctor
pip3 install -e .
```

---

#### ➤ If using a **Conda or other virtual environment**:

1. **Activate your environment**

   ```bash
   conda activate sceqcct  # Replace 'sceqcct' with your actual environment name
   ```

2. **Navigate to the predictor directory**

   ```bash
   cd $SEISCOMP_ROOT/share/sceqcct/tools/sceqcct-predicctor
   ```

3. **Install the predictor module**

   ```bash
   pip install -e .
   ```

> 🔒 **Note**: Make sure to install the module *within* the activated environment to ensure proper linking of dependencies.

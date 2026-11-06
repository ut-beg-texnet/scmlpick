"""Install the scmlpick predictor and the Python libraries it needs.

This does not install SeisComP. The module still needs a SeisComP tree
(4 or newer; 6 is the comfortable one) and the weight files that already
live under seiscomp/share/scmlpick/model.

Two ways to use this file on a glibc Linux (Ubuntu, Debian, Rocky, RHEL)
with Python 3.10, 3.11, or 3.12:

  # Create scmlpick/.venv and install everything into it.
  python3 setup.py --make-env

  # Or install into the environment that is already active.
  python3 -m pip install .

Alpine and other musl systems are outside the TensorFlow wheels used here.
"""

import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
PREDICTOR_ROOT = "seiscomp/share/scmlpick/tools/scmlpick-predicctor"

# Versions exercised with EQCCT, UTRANS, PhaseNet, EQTransformer,
# Ray task mode, and Ray model-actor, in float32 and bfloat16.
# tf-keras must be the same TensorFlow release: EQTransformer was saved
# with Keras 2 and is opened through that package.
INSTALL_REQUIRES = [
    "numpy>=1.26.4,<2.0",
    "pandas>=1.5",
    "scipy>=1.11",
    "obspy>=1.4",
    "ray>=2.9",
    "tensorflow==2.17.0",
    "tf-keras==2.17.0",
    "silence-tensorflow>=1.2",
    "psutil>=5.9",
]


def _venv_python(venv):
    return venv / "bin" / "python"


def make_env():
    """Create .venv next to this file and install scmlpick into it."""
    venv = ROOT / ".venv"
    if not _venv_python(venv).exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
    py = str(_venv_python(venv))
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([py, "-m", "pip", "install", str(ROOT)])
    print("Environment ready: %s" % venv)
    print("Activate it with: source %s" % (venv / "bin" / "activate"))


if __name__ == "__main__" and "--make-env" in sys.argv:
    make_env()
    raise SystemExit(0)


readme = ROOT / "README.md"
if readme.exists():
    long_description = readme.read_text(encoding="utf-8")
    long_description_content_type = "text/markdown"
else:
    long_description = (
        "SCMLPick predictor for EQCCT, UTRANS, PhaseNet, and EQTransformer "
        "inside SeisComP, with Ray task and model-actor execution."
    )
    long_description_content_type = "text/plain"


setup(
    name="scmlpick",
    version="0.1.0",
    description=(
        "SCMLPick predictor and Python dependencies for EQCCT, UTRANS, "
        "PhaseNet, and EQTransformer."
    ),
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    author="TexNet, Bureau of Economic Geology, The University of Texas at Austin",
    url="https://github.com/ut-beg-texnet/scmlpick",
    project_urls={
        "Documentation": "https://ut-beg-texnet.github.io/scmlpick/",
        "Source": "https://github.com/ut-beg-texnet/scmlpick",
        "Bug Tracker": "https://github.com/ut-beg-texnet/scmlpick/issues",
    },
    license="BSD-3-Clause",
    packages=find_packages(where=str(ROOT / PREDICTOR_ROOT)),
    package_dir={"": str(ROOT / PREDICTOR_ROOT)},
    include_package_data=True,
    python_requires=">=3.10,<3.13",
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "seismology",
        "SeisComP",
        "phase picking",
        "machine learning",
        "EQCCT",
        "PhaseNet",
        "EQTransformer",
        "TexNet",
    ],
    zip_safe=False,
)

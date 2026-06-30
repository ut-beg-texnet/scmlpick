from pathlib import Path
from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent

# The predictor directory is installed from within the SeisComP module tree,
# where a README file is not always present. Keep the long description robust.
readme = ROOT / "README.md"
if readme.exists():
    long_description = readme.read_text(encoding="utf-8")
    long_description_content_type = "text/markdown"
else:
    long_description = (
        "SCMLPick predictor component for running the EQCCT machine-learning "
        "phase picker within SeisComP-based workflows."
    )
    long_description_content_type = "text/plain"


setup(
    name="scmlpick",
    version="0.1.0",
    description="SCMLPick dependencies for SeisComP-based real-time machine-learning phase picking workflows.",
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
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        # Keep NumPy below 2.0 because the SCMLPick documentation notes
        # compatibility issues with SeisComP when using NumPy >= 2.0.
        "numpy>=1.26.4,<2.0",
        "pandas>=1.5",
        "obspy>=1.4",
        "ray>=2.0",
        "tensorflow>=2.10",
        "silence-tensorflow>=1.2",
    ],
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
        "TexNet",
    ],
    zip_safe=False,
)

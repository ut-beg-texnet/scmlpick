from setuptools import find_packages, setup

# Same libraries as the repository setup.py. This file is the one used by
# `pip install -e .` from the installed SeisComP tree.
setup(
    name="predictor",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
    install_requires=[
        "numpy>=1.26.4,<2.0",
        "pandas>=1.5",
        "scipy>=1.11",
        "obspy>=1.4",
        "ray>=2.9",
        "tensorflow==2.17.0",
        "tf-keras==2.17.0",
        "silence-tensorflow>=1.2",
        "psutil>=5.9",
    ],
    author="Camilo Munoz",
    description="Predictor module for SCMLPick",
    zip_safe=False,
)

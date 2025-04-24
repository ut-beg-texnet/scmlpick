from setuptools import setup, find_packages

setup(
    name="predictor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # "numpy>=1.23",
    ],
    author="Camilo Munoz",
    description="Predictor module for SCEQCCT",
    zip_safe=False,
)

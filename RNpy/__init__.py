from setuptools import setup, find_packages

setup(
    name="rnpy",
    version="0.1.0",
    description="A python tool for running resistor network simulations based on simple voxel structures.",
    author="Lukas Ketter",
    author_email="lukas-ketter@t-online.de",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "numba"
    ],
    python_requires=">=3.7",
)

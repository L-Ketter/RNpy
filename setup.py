from setuptools import setup, find_packages

setup(
    name="rnpy",
    version="0.0.1",
    packages=find_packages(),  
    install_requires=[
        "numpy",
        "matplotlib",
        "numba",
        "pyevtk",
        "cupy",
    ],
    author="Lukas Ketter",
    author_email="lukas-ketter@t-online.de",
    url="https://github.com/L-Ketter/RNpy",
    license="MIT", 
)

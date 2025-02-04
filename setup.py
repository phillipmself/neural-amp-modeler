# File: setup.py
# Created Date: 2020-04-08
# Author: Steven Atkinson (steven@atkinson.mn)

from distutils.util import convert_path
from setuptools import setup, find_packages

main_ns = {}
ver_path = convert_path("nam/_version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

requirements = [
    "matplotlib",
    "numpy",
    "pytorch_lightning",
    "scipy",
    "sounddevice",
    "tensorboard",
    "torch",
    "tqdm",
    "wavio<=0.0.4",  # Breaking change in 0.0.5
]

setup(
    name="nam",
    version=main_ns["__version__"],
    description="Neural amp modeler",
    author="Steven Atkinson",
    author_email="steven@atkinson.mn",
    url="https://github.com/sdatkinson/",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'nam = nam.train.gui:run',
        ]
    }
)

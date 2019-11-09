#!/usr/bin/env python
"""
Code to access, create and edit SAGA Survey data catalogs.
Project website: http://sagasurvey.org/
The MIT License (MIT)
Copyright (c) 2018-2019 The SAGA Survey
http://opensource.org/licenses/MIT
"""

import os

from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "SAGA", "version.py")) as f:
    exec(f.read())  # pylint: disable=W0122

setup(
    name="SAGA",
    version=__version__,  # pylint: disable=E0602 # noqa: F821
    description="Code to access, create and edit SAGA Survey data catalogs.",
    url="https://github.com/sagasurvey/saga",
    author="The SAGA Survey",
    author_email="saga@sagasurvey.org",
    maintainer="Yao-Yuan Mao",
    maintainer_email="yymao.astro@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="SAGA",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.11",
        "scipy>=0.17" "numexpr>=2.0",
        "astropy>=2.0",
        "easyquery>=0.1.5",
        "requests>=2.0",
        "fast3tree>=0.3.1",
        "healpy>=1.11",
    ],
)

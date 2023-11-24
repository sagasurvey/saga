#!/usr/bin/env python
"""
Code to access, create and edit SAGA Survey data catalogs.
Project website: http://sagasurvey.org/
The MIT License (MIT)
Copyright (c) 2018-2022 The SAGA Survey
http://opensource.org/licenses/MIT
"""

import os

from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "SAGA", "version.py")) as f:
    exec(f.read())

setup(
    name="SAGA",
    version=__version__,  # noqa: F821
    description="Code to access, create and edit SAGA Survey data catalogs.",
    url="https://github.com/sagasurvey/saga",
    download_url="https://github.com/sagasurvey/saga/archive/master.tar.gz",
    author="The SAGA Survey",
    author_email="saga@sagasurvey.org",
    maintainer="Yao-Yuan Mao",
    maintainer_email="yymao.astro@gmail.com",
    license="MIT",
    license_file="LICENSE",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="SAGA",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.16",
        "numexpr>=2.7",
        "astropy>=3.0",
        "easyquery>=0.2",
        "requests>=2.20",
    ],
    extras_require={
        "full": ["healpy>=1.12", "fast3tree>=0.3.1", "ipython>=7.0", "scikit-learn>=0.20", "pyperclip>=1.7"],
    },
)

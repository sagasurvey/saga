#!/usr/bin/env python
"""
Code to access, create and edit SAGA Survey data catalogs.
Project website: http://sagasurvey.org/
The MIT License (MIT)
Copyright (c) 2018-2021 The SAGA Survey
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="SAGA",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.15.3",
        "numexpr>=2.6",
        "astropy>=2.0",
        "easyquery>=0.1.5",
        "requests>=2.18",
    ],
    extras_require={
        "full": ["healpy>=1.12", "fast3tree>=0.3.1", "ipython", "scikit-learn", "pyperclip"],
    },
)

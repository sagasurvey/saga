#!/usr/bin/env python
"""
Code to access, create and edit SAGA Survey data catalogs.
Project website: http://sagasurvey.org/
The MIT License (MIT)
Copyright (c) 2018 The SAGA Survey
http://opensource.org/licenses/MIT
"""

import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), 'SAGA', 'version.py')) as f:
    exec(f.read()) # pylint: disable=W0122

setup(
    name='SAGA',
    version=__version__, # pylint: disable=E0602
    description='Code to access, create and edit SAGA Survey data catalogs.',
    url='https://github.com/sagasurvey/saga',
    author='The SAGA Survey',
    author_email='saga@sagasurvey.org',
    maintainer='Yao-Yuan Mao',
    maintainer_email='yymao.astro@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='SAGA',
    packages=find_packages(),
    install_requires=['numpy', 'numexpr', 'astropy', 'easyquery', 'requests'],
)

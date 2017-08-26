#!/usr/bin/env python
"""
Code to access, create and edit SAGA Survey data catalogs.
Project website: http://sagasurvey.org/
The MIT License (MIT)
Copyright (c) 2017 The SAGA Survey
http://opensource.org/licenses/MIT
"""

from setuptools import setup, find_packages

setup(
    name='SAGA',
    version='0.1.2',
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
    keywords='easyquery query numpy',
    packages=find_packages(),
    install_requires=['numpy', 'numexpr', 'astropy', 'easyquery', 'scipy', 'requests', 'casjobs'],
)

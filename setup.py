#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from os.path import join as pjoin
from glob import glob

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'dask[array] >= 1.1.4',
    'numpy >= 1.14.0',
    'numba >= 0.42.0',
    'ruamel.yaml >= 0.15.37',
    'scipy >= 1.2.0',
    'xarray-ms >= 0.1.3',
    'zarr >= 2.3.1'
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', 'nose', 'scipy']

setup(
    author="Simon Perkins",
    author_email='sperkins@ska.ac.za',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Science Data Processing flagging code, wrapped in dask",
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='tricolour',
    name='tricolour',
    packages=find_packages(),
    scripts=glob(pjoin('tricolour', 'scripts', '*')),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ska-sa/tricolour',
    version='0.1.0',
    zip_safe=False,
)

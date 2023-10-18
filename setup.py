#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'donfig >= 0.4.0',
    'numba >= 0.43.0',
    'scipy >= 1.2.0',
    'threadpoolctl >= 1.0.0',
    'dask-ms[xarray,zarr,s3]'
]

extras_require = {'testing': ['pytest',
                              'pytest-flake8',
                              'flake8 >= 4.0.0, <5.0.0',
                              'requests', 'gdown']}

setup(
    author="Simon Perkins",
    author_email='sperkins@ska.ac.za',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Science Data Processing flagging code, wrapped in dask",
    entry_points={
        'console_scripts': ['tricolour=tricolour.apps.tricolour.app:main'],
    },
    extras_require=extras_require,
    install_requires=requirements,
    long_description=readme,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='tricolour',
    name='tricolour',
    packages=find_packages(),
    python_requires=">=3.6",
    url='https://github.com/ska-sa/tricolour',
    version='0.1.8.1',
    zip_safe=False,
)

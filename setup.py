# -*- coding: utf-8 -*-
import os
import re
import subprocess

from setuptools import setup, find_packages, Command

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])

with open('README.md', 'r') as fp:
    readme = fp.read()

reqs = ['numpy',
        'pandas',
        'torch',
        'scikit-bio']

setup_args = {
    'name': 'gtnet',
    'description': 'A package for running Genome Taxonomy Network predictions',
    'long_description': readme,
    'long_description_content_type': 'text/x-rst; charset=UTF-8',
    'author': 'Andrew Tritt',
    'author_email': 'ajtritt@lbl.gov',
    'url': 'https://github.com/exabiome/gtnet',
    'license': "BSD",
    'install_requires': reqs,
    'packages': ['gtnet'],
    'package_data': {'gtnet': ["models/*",
                              "data/*.fna"]},
    'classifiers': [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    'keywords': 'python '
                'microbiome '
                'microbial-taxonomy '
                'cross-platform '
                'open-data '
                'data-format '
                'open-source '
                'open-science '
                'reproducible-research ',
    'zip_safe': False,
    'entry_points':{
        'console_scripts': [
            'gtnet = gtnet.main:run'
        ]
    },
}



if __name__ == '__main__':
    setup(**setup_args)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
from setuptools import setup

# Use the VERSION file to get version
version_file = os.path.join(os.path.dirname(__file__), 'underthesea', 'VERSION')
with open(version_file) as fh:
    version = fh.read().strip()

with io.open('README.rst', encoding="utf-8") as readme_file:
    readme = readme_file.read()

with io.open('HISTORY.rst') as history_file:
    history = history_file.read()

install_requires = [
    'Click>=6.0',
    'python-crfsuite>=0.9.6',
    'nltk',
    'tabulate',
    'tqdm',
    'requests',
    'joblib',
    'scikit-learn>=0.20,<0.22',
    'unidecode',
    'seqeval',
    'PyYAML'
]

tests_require = [
    'nose==1.3.7'
]

setup_requires = [
]

setup(
    name='underthesea',
    version=version,
    description="Vietnamese NLP Toolkit",
    long_description=readme + '\n\n' + history,
    author="Vu Anh",
    author_email='anhv.ict91@gmail.com',
    url='https://github.com/undertheseanlp/underthesea',
    packages=[
        'underthesea',
    ],
    package_dir={'underthesea': 'underthesea'},
    entry_points={
        'console_scripts': [
            'underthesea=underthesea.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=install_requires,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords='underthesea',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=tests_require,
    setup_requires=setup_requires
)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
from setuptools import setup

with io.open('README.rst', encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

install_requires = [
    'Click>=6.0',
    'python-crfsuite==0.9.5',
    'languageflow==1.1.6rc2'
]

tests_require = [
    'nose==1.3.7'
]

setup_requires = [

]

setup(
    name='underthesea',
    version='1.1.6',
    description="Vietnamese NLP Toolkit",
    long_description=readme + '\n\n' + history,
    author="Vu Anh",
    author_email='brother.rain.1024@gmail.com',
    url='https://github.com/magizbox/underthesea',
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=tests_require,
    setup_requires=setup_requires
)

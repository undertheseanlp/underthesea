import os
from setuptools import setup

# Use the VERSION file to get version
version_file = os.path.join(os.path.dirname(__file__), 'underthesea', 'VERSION')
with open(version_file) as fh:
    version = fh.read().strip()

install_requires = [
    'Click>=6.0',
    'python-crfsuite>=0.9.6',
    'nltk',
    'tqdm',
    'requests',
    'joblib',
    'scikit-learn',
    'PyYAML',
    'underthesea_core==0.0.5a2'
]

tests_require = [
    'nose==1.3.7'
]

setup_requires = [
]

extras_require = {
    'train': [
        'seqeval'
    ],
    'deep': [
        'torch>=1.1.0,<1.13',
        'transformers>=3.5.0'
    ]
}
setup(
    name='underthesea',
    version=version,
    description="Vietnamese NLP Toolkit",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
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
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    test_suite='tests',
    tests_require=tests_require,
    extras_require=extras_require,
    setup_requires=setup_requires
)

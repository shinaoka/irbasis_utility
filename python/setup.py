# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='irbasis utility',

    version='0.12',

    description='Utility python libraries for irbasis',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='https://github.com/shinaoka/irbasis_utility',

    author='Hiroshi Shinaoka, Markus Wallerberger, Dominique Geffroy',

    author_email='h.shinaoka@gmail.com', 

    classifiers=[ 
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='quantum many-body theory',

    packages = find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=['scipy', 'irbasis>=2.1.1', 'unittest2', 'chainmap', 'opt_einsum'],
)

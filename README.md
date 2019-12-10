[![Build Status](https://travis-ci.org/shinaoka/irbasis_utility.svg?branch=master)](https://travis-ci.org/shinaoka/irbasis_utility)

irbasis utility
======
Utility library for irbasis: support for two-point Green's function, three-point Green's function.<br>
This software is released under the MIT License, see LICENSE.txt. 
The files in the opt_einsum directory are originally from the opt_einsum packakge,
and they are licensed under the MIT License as well.

## Requirements
* irbasis >= 1.0.2
* scipy

## How to build tests & install via pip

```
> mkdir build && cd build
> cmake path_to_source_directory
> make
> make test
> python setup.py bdist_wheel # make a universal binary package
> cd dist
> pip install irbasis_utility-*.whl
```

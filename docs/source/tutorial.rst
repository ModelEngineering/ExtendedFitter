Tutorial
=========

This section describes the basic usage of ``fitterpp``.

To install the package use:
    ``pip install fitterpp``

To use the package in your code, include the following statement
at the top of your python module:

    ''import fitterpp as fpp''

The package assumes that you have previously writtern a function
that inputs
[``lmfit.Parameters``](https://lmfit.github.io/lmfit-py/parameters.html)
and outputs
a list (or list-like) of floats that are the difference between
what the function computed for these parameter values and observational
data.

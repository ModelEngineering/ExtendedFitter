Tutorial
=========

This section describes the basic usage of ``fitterpp``.

To install the package use:
    ``pip install fitterpp``

To use the package in your code, include the following statement
at the top of your python module:

    ``import fitterpp as fpp``

To do fitting, you must first write a parameterized function.
For example, consider the following function for a parabola
the has two parameters:
* where the parabola is centered on the x-axis
* a multiplier for how quickly the y-value increases

.. code-block:: python

    def calcParabola(center=0, mult=1, xvalues=XVALUES, is_noise=False):
        estimates = np.array([mult*(n - center)**2 + for n in xvalues])
        return pd.DataFrame({"x": xvalues, "y": estimates})

Note that all arguments to ``calcParabola`` are specified using keywords.
The output from the function is a ``pandas`` ``DataFrame``.

You will also need to describe the parameters to be fitted.
In our example, these are ``center`` and ``mult``.
You use
`lmfit.Parameters <(https://lmfit.github.io/lmfit-py/parameters.html>`_
to describe these parameters, as shown below.

.. code-block:: python

    parameters = lmfit.Parameters()
    parameters.add("center", value=0, min=0, max=100)
    parameters.add("mult", value=0, min=0, max=100)

Last, you must provide data that is used to fit the parameters.
The data should be a ``pandas`` ``DataFrame`` that has some (or all)
of the columns present in the output of the function to be fit.

and outputs
a list (or list-like) of floats that are the difference between
what the function computed for these parameter values and observational
data.

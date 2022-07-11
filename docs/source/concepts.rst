Core Concepts
=============

Many times we want to fit a *parameterized* function to data.
For example, suppose that we have an array of data
``y[n]`` that we want to fit as a linear function of the variables
``x[n]``,
where the n-th element of each array.
That is, we want to find the slope ``a`` and the y-intercept ``b``
such that
``a*x[n] + b`` is as close as possible to ``y[n]``.
We define "as close as possible" to mean that the
sum of the squared difference between ``y[n]`` and
``a*x[n] + b`` is as small as possible.

Fitting is the process of finding parameters ``a`` and ``b``
that make the **fitting function** as close as possible to the observational
data.
Thus, to perform fitting, we must specify:

* the fitting function;

* the parameters of the function that are to be adjusted;

* observational data;

* outputs of the function that are used in fitting (e.g., ``y``).


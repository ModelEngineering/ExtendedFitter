# ExtendedFitter
Extended fitter for fitting parameters.

The ``Parameter`` class describes parameter constraints.
* Constructor: parameterName, lower_bound, upper_bound
* value: value of the parameter
* name: name of the parameter

The ``ParameterManager`` class is used to manage parameters in multiple models.

The ``ExtendedFitter`` class extends lmfit by:
1. Ensuring that the parameters chosen have the lowest residuals sum of squares
2. Providing for a sequence of optimization methods
3. Providing an option to repeat a method sequence with different randomly
   chosen initial parameter values (numRandomRestart).

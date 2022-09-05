
# fitterpp: Simplified Parameter Fitting With Advanced Capabilities

fitterpp (pronounced "fitter plus plus") extends the capabilities of ``lmfit`` in several ways:

1. simplifies parameter fitting by automating the calculation of residuals;

2. ensures that the parameters reported have the smallest residual sum of squares;

3. provides for running several fitting algorithm in succession;

4. reports statistics on the runtime and quality of parameter fits.

For more details, see 
[readthedocs](https://fitterpp.readthedocs.io/en/latest/
).

# Version History
* 0.0.3 
  * dictToParameter can set initial value to random in [min, max]
  * method_names can be strings or a FitterppMethod
  * latin cube index to compute across a latin cube
* 0.0.2 7/22/2022.
    User function must be self-describing, optionally returning a dataframe
* 0.0.1 7/19/2022. First release.

"""Constants for Project."""

# Numerical thresholds
PARAMETER_LOWER_BOUND = 0
PARAMETER_UPPER_BOUND =10

# Constants
ALL = "#all#"  # Model for all parameters

# Fitting methods
#  Minimizer methods
METHOD_DIFFERENTIAL_EVOLUTION = "differential_evolution"
METHOD_BOTH = "both"
METHOD_LEASTSQ = "leastsq"
METHOD_FITTER_DEFAULTS = [METHOD_DIFFERENTIAL_EVOLUTION, METHOD_LEASTSQ]
METHOD_BOOTSTRAP_DEFAULTS = [METHOD_LEASTSQ]
#
MAX_NFEV_DFT = 10
MAX_NFEV = "max_nfev"

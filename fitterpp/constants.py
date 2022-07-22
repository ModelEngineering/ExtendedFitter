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
METHOD_BOOTSTRAP_DEFAULTS = [METHOD_DIFFERENTIAL_EVOLUTION]
ROW_KEY = "row_key"
#
MAX_NFEV_DFT = 1000
MAX_NFEV = "max_nfev"
#
SEC_TO_MS = 1000

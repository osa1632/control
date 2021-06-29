import numpy as np

# PARAM_AND_CONTROLSIG_options = list(range(2**2))
# SIMULATE_DIFFRENTIATOR_THIRD_options = list(range(2**4))
# PARAM_AND_CONTROLSIG_options = list(range(2**2))

SIMULATE_NOISEnFILTER_THIRD_DIFFRENTIATOR = 0|4|2|1
PARAM_CONTROLSIG = 3

d=lambda x, ii: (x >> (ii-1)) &1

useSecondThanFirstA = d(PARAM_CONTROLSIG, 1)
useSecondThanFirstU_c = d(PARAM_CONTROLSIG, 2) & d(~SIMULATE_NOISEnFILTER_THIRD_DIFFRENTIATOR, 1)


use_differentiator = d(SIMULATE_NOISEnFILTER_THIRD_DIFFRENTIATOR, 1)
use_secondOrThird = d(SIMULATE_NOISEnFILTER_THIRD_DIFFRENTIATOR, 2)
simulateSignalThanDiffrientiator = d(~SIMULATE_NOISEnFILTER_THIRD_DIFFRENTIATOR, 4)

use_filter=d(SIMULATE_NOISEnFILTER_THIRD_DIFFRENTIATOR, 1) & d(SIMULATE_NOISEnFILTER_THIRD_DIFFRENTIATOR, 3)
add_noise=d(SIMULATE_NOISEnFILTER_THIRD_DIFFRENTIATOR, 3)

if use_secondOrThird:
    x0 = np.array([0,0,.0])
else:
    x0 = np.array([-3., -4, -7])

if use_secondOrThird:
    alpha = -2.; L=1000
    # alpha = -50.; L=5001
else:
    alpha = -5.; L = 501

# alpha = -10.; L = 1200
# alpha = -1.9
# L = 300

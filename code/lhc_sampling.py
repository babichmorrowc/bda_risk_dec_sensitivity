# Code to create Latin hypercube samples for the decision-related parameters

import safepython.sampling as sampling
import scipy.stats as st
import numpy as np

##################################################################################################################
# Latin Hypercube sampling

# Set distribution parameters for each input variable
# Parameters loc and scale give the uniform distribution over [loc, loc + scale]

# Cost per day of work lost
cost_per_day_locscale = [100, 200]
# Cost per person per year of d2
d2_1_locscale = [150, 200]
# Effectiveness of d2
d2_2_locscale = [0.3, 0.2]
# Cost per person per year of d3
d3_1_locscale = [500, 200]
# Effectiveness of d3
d3_2_locscale = [0.7, 0.2]
# Relative importance of financial cost
fin_weight_locscale = [0.7, 0.2]

input_locscales = [cost_per_day_locscale, d2_1_locscale, d2_2_locscale, d3_1_locscale, d3_2_locscale, fin_weight_locscale]

# Number of Latin hypercube samples
n_samples = 200

# Generate Latin hypercube samples
X_lat_hyp = sampling.AAT_sampling(samp_strat = 'lhs',
                                  M = len(input_locscales),
                                  distr_fun=st.uniform,
                                  distr_par=input_locscales,
                                  N=n_samples
)
# Round the financial columns to 2 decimal places
# Columns 1, 2, and 4 are financial costs
X_lat_hyp[:,0] = np.round(X_lat_hyp[:,0], 2)
X_lat_hyp[:,1] = np.round(X_lat_hyp[:,1], 2)
X_lat_hyp[:,3] = np.round(X_lat_hyp[:,3], 2)

# Check max and min of each column
# for i in range(X_lat_hyp.shape[1]):
#     print('Column ' + str(i) + ':')
#     print('Min: ' + str(min(X_lat_hyp[:,i])))
#     print('Max: ' + str(max(X_lat_hyp[:,i])))

# Save the Latin Hypercube samples
np.savetxt('./data/lat_hyp_samples_' + str(n_samples) + '.csv', X_lat_hyp, delimiter=',')



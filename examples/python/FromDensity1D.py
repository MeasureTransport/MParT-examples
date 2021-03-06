import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

from mpart import *

# Make target samples
num_points = 5000
mu = 2
sigma = .5
x = np.random.randn(num_points)[None,:]

# for plotting
rv = norm(loc=mu,scale=sigma)
t = np.linspace(-3,6,100)
rho_t = rv.pdf(t)

num_bins = 50
# Before optimization plot
plt.figure()
plt.hist(x.flatten(), num_bins, facecolor='blue', alpha=0.5, density=True, label='Reference samples')
plt.plot(t,rho_t,label="Target density")
plt.legend()
plt.show()

# Create multi-index set
multis = np.array([[0], [1]])  # affine transform enough to capture Gaussian target
mset = MultiIndexSet(multis)
fixed_mset = mset.fix(True)

# Set MapOptions and make map
opts = MapOptions()
monotoneMap = CreateComponent(fixed_mset, opts)

# KL divergence objective
def objective(coeffs, monotoneMap, x, num_points):
    monotoneMap.SetCoeffs(coeffs)
    map_of_x = monotoneMap.Evaluate(x)
    pi_of_map_of_x = rv.logpdf(map_of_x)
    log_det = monotoneMap.LogDeterminant(x)
    return -np.sum(pi_of_map_of_x + log_det)/num_points

# Optimize
print('Starting coeffs')
print(monotoneMap.CoeffMap())
print('and error: {:.2E}'.format(objective(monotoneMap.CoeffMap(), monotoneMap, x, num_points)))
res = minimize(objective, monotoneMap.CoeffMap(), args=(monotoneMap, x, num_points), method="Nelder-Mead")
print('Final coeffs')
print(monotoneMap.CoeffMap())
print('and error: {:.2E}'.format(objective(monotoneMap.CoeffMap(), monotoneMap, x, num_points)))

# After optimization plot
map_of_x = monotoneMap.Evaluate(x)
plt.figure()
plt.hist(map_of_x.flatten(), num_bins, facecolor='blue', alpha=0.5, density=True, label='Mapped samples')
plt.plot(t,rho_t,label="Target density")
plt.legend()
plt.show()

assert math.isclose(monotoneMap.CoeffMap()[0], 2, abs_tol=1e-1)
assert math.isclose(monotoneMap.CoeffMap()[1], -0.68, abs_tol=1e-1)
assert math.isclose(objective(monotoneMap.CoeffMap(), monotoneMap, x, num_points), 1.41, abs_tol=1e-1)

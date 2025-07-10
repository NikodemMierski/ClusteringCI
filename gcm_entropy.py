#!/usr/bin/env python3

"""
CI_dbscan: gcm_entropy
Version 1.0, July 2, 2025.
Copyright (C) 2025 Nikodem Mierski

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt

# GCM 
def f(a,epsilon,x):
	"""Computation of the next vector in the GCM system (Kaneko 1990)."""
	fx = 1 - a * x * x
	return (1 - epsilon) * fx + epsilon / len(x) * fx.sum()

# function for computing local entropy values
def shannon_entropy(series, window_size):
    entropy_values = []
    half_window = window_size // 2  

    for i in range(len(series)):
        if i < half_window or i > len(series) - half_window - 1:
            entropy_values.append(np.nan)
        else:
            window = series[i - half_window:i + half_window + 1]
            
            prob_dist, _ = np.histogram(window, bins=100, range=(np.min(series), np.max(series)), density=True)
            
            prob_dist = prob_dist / np.sum(prob_dist)
            epsilon = 1e-10 
            entropy = -np.sum(prob_dist * np.log2(prob_dist + epsilon))
            entropy_values.append(entropy)

    return np.array(entropy_values)

# function for computing sum of variances of local entropy
def entropy_variance(data):
    sum = 0
    window_size = 200
    for i in range(0, data.shape[1]):
        entropy = np.nanvar(shannon_entropy(data[:,i], window_size))
        sum = sum + entropy

    return sum

# parameters 
N = 5
a = 2
eps_values = np.linspace(0.1,0.3, 2001)
eps_values = np.round(eps_values,4)
t_start = 10000
t_end = 40000
variances = []

for eps in eps_values:
    np.random.seed (7)
    x = np.random.random ((N,))

    for i in range(t_start):
        x = f(a,eps,x)

    data = np.empty ((t_end - t_start,N,))

    for i in range(t_end - t_start):
        data[i,:] = x
        x = f(a,eps,x)

    variances.append(entropy_variance(data))

# list of epsilon-variance pairs
pairs = list(zip(variances, eps_values))
sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

# print 10 largest variances
top_10 = sorted_pairs[:10]
for i, (w, e) in enumerate(top_10, 1):
    print(f"eps: {e}, variance of entropy: {w}")

# plot of local entropy variance as a function of epsilon
plt.scatter(eps_values[:len(variances)], variances, s=5, c="r")
plt.grid()
plt.xlabel("ε")
plt.ylabel("Sum of variances of local entropy for all dimensions")
plt.tight_layout()
plt.show()

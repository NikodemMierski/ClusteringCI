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
import antropy as ant

# GCM 
def f(a,epsilon,x):
	"""Computation of the next vector in the GCM system (Kaneko 1990)."""
	fx = 1 - a * x * x
	return (1 - epsilon) * fx + epsilon / len(x) * fx.sum()

# function for computing local Shannon entropy values
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

# function for computing local permutation entropy values
def permutation_entropy(series, window_size):
    entropy_values = []
    half_window = window_size // 2  

    for i in range(len(series)):
        if i < half_window or i > len(series) - half_window - 1:
            entropy_values.append(np.nan)
        else:
            window = series[i - half_window:i + half_window + 1]
            entropy = ant.perm_entropy(window, order = 3, delay = 1)
            entropy_values.append(entropy)

    return np.array(entropy_values)

# function for computing sum of variances of local Shannon entropy
def shannon_entropy_variance(data):
    sum = 0
    window_size = 200
    for i in range(0, data.shape[1]):
        entropy = np.nanvar(shannon_entropy(data[:,i], window_size))
        sum = sum + entropy

    return sum

# function for computing sum of variances of local permutation entropy
def permutation_entropy_variance(data):
    sum = 0
    window_size = 200
    for i in range(0, data.shape[1]):
        entropy = np.nanvar(permutation_entropy(data[:,i], window_size))
        sum = sum + entropy

    return sum

# parameters 
N = 3
a = 2
eps_values = np.linspace(0.1,0.28, 1801)
eps_values = np.round(eps_values,4)
t_start = 10000
t_end = 40000
shannon_variances = []
permutation_variances = []

for eps in eps_values:
    np.random.seed (2)
    x = np.random.random ((N,))

    for i in range(t_start):
        x = f(a,eps,x)

    data = np.empty ((t_end - t_start,N,))

    for i in range(t_end - t_start):
        data[i,:] = x
        x = f(a,eps,x)

    shannon_variances.append(shannon_entropy_variance(data))
    permutation_variances.append(permutation_entropy_variance(data))

# list of epsilon-variance pairs for local Shannon entropy
pairs_shannon = list(zip(shannon_variances, eps_values))
sorted_pairs_shannon = sorted(pairs_shannon, key=lambda x: x[0], reverse=True)

# print 10 largest variances of local Shannon entropy
top_10_shannon = sorted_pairs_shannon[:10]
for i, (w, e) in enumerate(top_10_shannon, 1):
    print(f"eps: {e}, variance of Shannon entropy: {w}\n")

# list of epsilon-variance pairs for local permutation entropy
pairs_permutation = list(zip(permutation_variances, eps_values))
sorted_pairs_permutation = sorted(pairs_permutation, key=lambda x: x[0], reverse=True)

# print 10 largest variances of local permutation entropy
top_10_permutation = sorted_pairs_permutation[:10]
for i, (w, e) in enumerate(top_10_permutation, 1):
    print(f"eps: {e}, variance of permutation entropy: {w}\n")

# plot of local Shannon entropy variance as a function of epsilon
plt.scatter(eps_values, shannon_variances, s=5, c="r")
plt.grid()
plt.xlabel("ε")
plt.ylabel("Sum of variances of local Shannon entropy for all dimensions")
plt.tight_layout()
plt.show()

# plot of local permutation entropy variance as a function of epsilon
plt.scatter(eps_values, permutation_variances, s=5, c="r")
plt.grid()
plt.xlabel("ε")
plt.ylabel("Sum of variances of local permutation entropy for all dimensions")
plt.tight_layout()
plt.show()
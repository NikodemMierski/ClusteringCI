#!/usr/bin/env python3

"""
CI_dbscan: gaussian_entropy
Version 1.0, July 29, 2025.
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
import pandas as pd
import matplotlib.pyplot as plt
import antropy as ant

# Mutually Coupled Gaussian Maps
def gaussian_map(x, x_coupled, alpha, beta, epsilon):
    return np.exp(-alpha * x**2) + beta + epsilon * (x_coupled - x)

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
alpha = 12
beta = -0.504
t_start = 50000
t_end = 80000

eps_values = np.linspace(-0.2,0.2, 4001)
eps_values = np.round(eps_values,4)
shannon_variances = []
permutation_variances= []

for epsilon in eps_values:
    np.random.seed(11)
    x = np.random.random ((2,))
    x1=x[0]
    x2=x[1]
    x1_series, x2_series = [], []

    for _ in range(t_start):
        x1_new = gaussian_map(x1, x2, alpha, beta, epsilon)
        x2_new = gaussian_map(x2, x1, alpha, beta, epsilon)
        x1, x2 = x1_new, x2_new
    for _ in range(t_end-t_start):
        x1_new = gaussian_map(x1, x2, alpha, beta, epsilon)
        x2_new = gaussian_map(x2, x1, alpha, beta, epsilon)
        x1, x2 = x1_new, x2_new
        x1_series.append(x1)
        x2_series.append(x2)

    data = np.column_stack((x1_series, x2_series)) 

    shannon_variances.append(shannon_entropy_variance(data))
    permutation_variances.append(permutation_entropy_variance(data))

# list of epsilon-variance pairs for local Shannon entropy
pairs_shannon = list(zip(shannon_variances, eps_values))
sorted_pairs_shannon = sorted(pairs_shannon, key=lambda x: x[0], reverse=True)

# print 10 largest variances of local Shannon entropy
top_10_shannon = sorted_pairs_shannon[:10]
for i, (w, e) in enumerate(top_10_shannon, 1):
    print(f"eps: {e}, variance of local Shannon entropy: {w}\n")

# list of epsilon-variance pairs for local permutation entropy
pairs_permutation = list(zip(permutation_variances, eps_values))
sorted_pairs_permutation = sorted(pairs_permutation, key=lambda x: x[0], reverse=True)

# print 10 largest variances of local permutation entropy
top_10_permutation = sorted_pairs_permutation[:10]
for i, (w, e) in enumerate(top_10_permutation, 1):
    print(f"eps: {e}, variance of local permutation entropy: {w}\n")


# plot of local Shannon entropy variance and local permutation entropy variance as a function of epsilon
shannon_variances = np.array(shannon_variances)
permutation_variances = np.array(permutation_variances)

# normalized variances
shannon_variances_normed = (shannon_variances - shannon_variances.min()) / (shannon_variances.max() - shannon_variances.min())
permutation_variances_normed= (permutation_variances - permutation_variances.min()) / (permutation_variances.max() - permutation_variances.min())

plt.scatter(eps_values, shannon_variances_normed, s=5)
plt.scatter(eps_values, permutation_variances_normed, s=5)
plt.grid()
plt.xlabel("ε", fontsize=10)
plt.ylabel("Sum of variances of local entropy for all dimensions", fontsize=10)
plt.tight_layout()
plt.show()

#!/usr/bin/env python3

"""
CI_dbscan: gcm_analysis
Version 2.0, September 23, 2025.
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
import hdbscan
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import math
from scipy.stats import chi2

# GCM
def f(a,epsilon,x):
	"""Computation of the next vector in the GCM system (Kaneko 1990)."""
	fx = 1 - a * x * x
	return (1 - epsilon) * fx + epsilon / len(x) * fx.sum()

# function for cluster merging
def merge(labels):
    valid_labels = [label for label in labels if label != -1]
    unique_labels = sorted(set(valid_labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    # build adjacency matrix only for valid labels
    adj_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    for i in range(len(labels) - 1):
        current = labels[i]
        next_ = labels[i + 1]
        
        if current == -1 or next_ == -1:
            continue  # skip transitions involving -1

        adj_matrix[label_to_index[current], label_to_index[next_]] += 1

    transitions = pd.DataFrame(adj_matrix, index=unique_labels, columns=unique_labels)

    # check for dominant transitions (over 80% in a row)
    def check_dominant_rows(df):
        return [any(row > 0.8 * row.sum()) for _, row in df.iterrows()]

    should_merge = np.any(check_dominant_rows(transitions))

    merged_groups = []

    if should_merge:
        def find_merge_chains(df):
            chains = []
            n_rows = df.shape[0]

            def explore_chain(start_idx, current_chain):
                row = df.iloc[start_idx]
                row_sum = row.sum()

                found = False
                for col_idx, value in enumerate(row):
                    if value > 0.8 * row_sum and col_idx not in current_chain:
                        found = True
                        new_chain = current_chain + (col_idx,)
                        explore_chain(col_idx, new_chain)

                if not found and len(current_chain) > 1:
                    group = tuple(sorted(current_chain))
                    if group not in chains:
                        chains.append(group)

            for i in range(n_rows):
                explore_chain(i, (i,))

            return chains

        index_groups = find_merge_chains(transitions)

        # convert index groups back to label groups
        merged_groups = [
            tuple(sorted([unique_labels[idx] for idx in group]))
            for group in index_groups
        ]

        # merge overlapping tuples
        def merge_overlapping_groups(group_list):
            if not group_list:
                return []

            groups = {}
            for group in group_list:
                elements = set(group)
                overlapping_keys = [k for k, v in groups.items() if elements & v]

                merged = elements
                for key in overlapping_keys:
                    merged |= groups[key]
                    del groups[key]

                groups[id(merged)] = merged

            return [tuple(sorted(g)) for g in groups.values()]

        merged_groups = merge_overlapping_groups(merged_groups)

        # update labels - assign minimum label in group (without -1)
        for group in merged_groups:
            new_label = min(group)
            labels = np.array([
                new_label if lbl in group else lbl
                for lbl in labels
            ])

        # normalize labels (without -1) to consecutive numbers starting from 0
        unique_final_labels = sorted(set([l for l in labels if l != -1]))
        label_remap = {old: new for new, old in enumerate(unique_final_labels)}

        labels = np.array([
            label_remap[lbl] if lbl != -1 else -1
            for lbl in labels
        ])
    else:
        pass
    return labels

def include_isolated_points(lst):
    lst = list(lst)  
    
    i = 0
    while i < len(lst) - 1:
        j = i + 1
        while j < len(lst) and lst[j] == lst[i]:
            j += 1
        
        sequence_length = j - i
        
        if sequence_length == 1 and i > 0 and j < len(lst) and lst[i - 1] == lst[j]:
            for k in range(i, j):
                if lst[k] != lst[i - 1]:  
                    lst[k] = lst[i - 1]
        
        i = j  
    
    return np.array(lst)

def time_in_clusters(sequence):
    result = []
    previous = None
    count = 0
    
    for x in sequence:
        if x == previous:
            count += 1
        else:
            if previous is not None:
                result.append((previous, count))
            previous = x
            count = 1
    
    if previous is not None:
        result.append((previous, count))
    
    return result

def average_noise_time(sequence):
    clusters = time_in_clusters(sequence)
    if not clusters:
        return 0

    times = [t for x, t in clusters if x == -1]
    average_time = sum(times) / len(times)

    return average_time

def var_pca(df):
    clusters = df['Cluster'].unique()
    clusters = [x for x in clusters if x != -1]

    pc1_variances = []

    for x in clusters:
        cluster = df[df['Cluster'] == x].drop(columns=['Cluster'])
        cluster = cluster.select_dtypes(include='number')

        # omit clusters that are too small
        if min(cluster.shape) < 3:
            continue  

        pca = PCA(n_components=3)
        pca.fit(cluster)

        pc1_var = pca.explained_variance_ratio_[0]
        pc1_variances.append(pc1_var)

    if pc1_variances:
        return min(pc1_variances)
    
# Ljung-Box Test
def ljung_box_test(sequence):
    result = acorr_ljungbox(sequence, lags=[10], return_df=True)
    pval = result['lb_pvalue'].iloc[0]

    return pval

# Augmented Dickey-Fuller Test 
def adf(sequence):
    result = adfuller(sequence)
    p_value = result[1]

    return p_value

# O’Brien-Dyck Runs Test - code from https://github.com/psinger/RunsTest described in file runs_test.py
def weighted_variance(counts):
    avg = 0
    for length, count in counts.items():
        avg += count * length

    counts_only = list(counts.values())
    avg /= sum(counts_only)

    var = 0
    for length, count in counts.items():
        var += count * math.pow((length - avg), 2)

    try:
        var /= sum(counts_only) - 1
    except ZeroDivisionError:
        raise Exception("Division by zero due to too few counts!")

    return var

# Runs test 
def runs_WW(sequence):
    a, p = runstest_1samp(sequence)
    return p

def runs_OBD(input_data, path=True):
    if path:
        counter = 1
        cats = defaultdict(lambda: defaultdict(int))

        for i, elem in enumerate(input_data):
            if i == len(input_data) - 1:
                cats[elem][counter] += 1
                break

            if input_data[i + 1] == elem:
                counter += 1
            else:
                cats[elem][counter] += 1
                counter = 1
    else:
        cats = input_data

    x2 = 0
    df = 0
    nr_elem = len(cats.keys())
    fail_cnt = 0

    for elem in cats.keys():
        ns = sum([x * y for x, y in cats[elem].items()])
        rs = sum(cats[elem].values())

        if len(cats[elem].keys()) == 1 or rs == 1 or (ns - rs) == 1:
            fail_cnt += 1
            continue

        ss = weighted_variance(cats[elem])
        cs = ((rs ** 2) - 1) * (rs + 2) * (rs + 3) / (2 * rs * (ns - rs - 1) * (ns + 1))
        vs = cs * ns * (ns - rs) / (rs * (rs + 1))

        x2 += ss * cs
        df += vs

    if nr_elem - fail_cnt < 2:
        raise Exception("Too many categories were ignored. Cannot perform the test.")

    if x2 == 0 or df == 0:
        raise Exception("x2 or df is zero, cannot compute p-value.")

    pval = chi2.sf(x2, df)
    return pval

# sequence of consecutively visited clusters
def sequence_of_clusters(sequence):
    sequence_clusters = sequence[np.append(True, sequence[1:] != sequence[:-1])]
    return sequence_clusters

# time in clusters
def time_in_cluster(sequence):
    result = []
    previous = None
    count = 0
    
    for x in sequence:
        if x == previous:
            count += 1
        else:
            if previous is not None:
                result.append((previous, count))
            previous = x
            count = 1
    
    if previous is not None:
        result.append((previous, count))
    
    return result
    
# parameters
N = 3
a_values = np.arange(1.4, 2.0001, 0.01)
eps_values = np.arange(0.005, 0.4001, 0.01)
t_start = 20000
t_end = 40000

noise_times = []
variances_PCA = []
tests = []

for a in a_values:
    for eps in eps_values:
        score = 0
        np.random.seed(1)
        x = np.random.random((N,))

        for i in range(t_start):
            x = f(a, eps, x)

        data = np.empty((t_end - t_start, N))

        for i in range(t_end - t_start):
            data[i, :] = x
            x = f(a, eps, x)

        df = pd.DataFrame(data)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=300)
        labels = clusterer.fit_predict(data)
        labels=merge(labels)
        labels = include_isolated_points(labels)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.sum(labels == -1) / len(labels)
        if n_clusters < 2 or noise_ratio > 0.9 or noise_ratio < 0.1:
            variances_PCA.append(0)
            noise_times.append(0)
            tests.append(-1)
        else:
            df["Cluster"] = labels
            variances_PCA.append(var_pca(df))
            noise_times.append(average_noise_time(labels))
            consecutive_clusters = sequence_of_clusters(np.array(labels))
            consecutive_clusters_without_noise = consecutive_clusters[consecutive_clusters != -1]
            clusters_times = time_in_cluster(labels)
            times = [length for _, length in clusters_times if _ != -1]

            if n_clusters > 2:
                try:
                    run_inne_value = runs_OBD(consecutive_clusters_without_noise)
                except Exception as e:
                    run_inne_value = 0
                test1 = run_inne_value
            else:
                test1 = runs_WW(consecutive_clusters_without_noise)
            try:
                test2 = adf(times)
            except Exception as e:
                test2 = 1
            test3 = ljung_box_test(times)
            if test1 >= 0.05:
                score +=1
            if test2 <= 0.05:
                score +=1
            if test3 >= 0.05:
                score+=1
            tests.append(score)

# plot of average noise time
heatmap = np.array(noise_times).reshape(len(a_values), len(eps_values))

heatmap_T = heatmap.T 

heatmap_T_2 = heatmap_T.copy()
heatmap_T_2[heatmap_T_2 == 0] = np.nan

colors = [
    "#e6f2e6",
    "#cce6cc",
    "#99cc99",
    "#66b266",
    "#339933",
    "#228b22",
    "#006400"
]

positions = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 1.0]

cmap = mcolors.LinearSegmentedColormap.from_list("custom_greens", list(zip(positions, colors)))
cmap.set_bad("white")

plt.figure(figsize=(10, 7))
im = plt.imshow(heatmap_T_2, extent=[a_values[0], a_values[-1], eps_values[0], eps_values[-1]], aspect='auto', cmap=cmap, origin='lower')

plt.xlabel("a", fontsize=10)
plt.ylabel("ε", fontsize=10)
plt.xticks(np.arange(1.4, 2.01, 0.1))

plt.xlim(1.4, 2.0)
plt.ylim(0, 0.4)
plt.grid(True, linestyle='--', alpha=0.5)
#plt.text(0.23, 0.91, "(a)", transform=plt.gca().transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color="black")
#plt.text(0.68, 0.45, "(b)", transform=plt.gca().transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color="black")
cbar = plt.colorbar(im)
cbar.set_label('Average time of wandering between clusters', fontsize=10)
plt.tight_layout()
plt.show()

# plot of variance of the first principal component in PCA
heatmap = np.array(variances_PCA).reshape(len(a_values), len(eps_values))

heatmap_T = heatmap.T 

heatmap_T_2 = heatmap_T.copy()
heatmap_T_2[heatmap_T_2 == 0] = np.nan

colors = [
    "powderblue",
    "lightblue",
    "skyblue",
    "deepskyblue",
    "royalblue",
    "darkblue"
]

positions = [0,0.1, 0.35, 0.65, 0.95, 1.0]

cmap = mcolors.LinearSegmentedColormap.from_list("custom_blues", list(zip(positions, colors)))
cmap.set_bad("white")

plt.figure(figsize=(10, 7))
im = plt.imshow(heatmap_T_2, extent=[a_values[0], a_values[-1], eps_values[0], eps_values[-1]], aspect='auto', cmap=cmap, origin='lower')

plt.xlabel("a", fontsize=10)
plt.ylabel("ε", fontsize=10)
plt.xticks(np.arange(1.4, 2.01, 0.1))
plt.xlim(1.4, 2.0)
plt.ylim(0, 0.4)
plt.grid(True, linestyle='--', alpha=0.5)
#plt.text(0.23, 0.91, "(a)", transform=plt.gca().transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color="yellow")
#plt.text(0.68, 0.45, "(b)", transform=plt.gca().transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color="yellow")
cbar = plt.colorbar(im)
cbar.set_label('Minimum variance captured by the first principal\ncomponent among all identified clusters', fontsize=10)
plt.show()

# plot of number of passed randomness tests
heatmap = np.array(tests).reshape(len(a_values), len(eps_values))
heatmap_T = heatmap.T
heatmap_T = np.where(heatmap_T == -1, np.nan, heatmap_T)

cmap = mcolors.ListedColormap([
    "#d9b38c",
    "#a66d4a",
    "#7a320f",
    "#3b1e0c"
])
cmap.set_bad(color='white')

plt.figure(figsize=(10, 7))
im = plt.imshow(heatmap_T, extent=[a_values[0], a_values[-1], eps_values[0], eps_values[-1]], aspect='auto', cmap=cmap, origin='lower')

plt.ylabel('ε', fontsize=10)
plt.xlabel('a', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
#plt.text(0.23, 0.91, "(a)", transform=plt.gca().transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color="yellow")
#plt.text(0.68, 0.45, "(b)", transform=plt.gca().transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color="yellow")

cbar = plt.colorbar(im, ticks=[0.4, 1.15, 1.9, 2.6])
cbar.set_ticklabels([0, 1, 2, 3])
cbar.set_label('Number of passed randomness tests', fontsize=10)

plt.xlim(1.4, 2.0)
plt.ylim(0, 0.4)
plt.xticks(np.arange(1.4, 2.01, 0.1))
plt.show()
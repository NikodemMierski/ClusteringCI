#!/usr/bin/env python3

"""
CI_dbscan: gcm_hdbscan
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
import pandas as pd
import hdbscan
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

# parameters
N = 3
a_values = np.arange(1.4, 2.0001, 0.1)
eps_values = np.arange(0.005, 0.4001, 0.1)
t_start = 20000
t_end = 40000

noise_times = []
variances_PCA = []

for a in a_values:
    for eps in eps_values:
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
        else:
            df["Cluster"] = labels
            variances_PCA.append(var_pca(df))
            noise_times.append(average_noise_time(labels))



# plot of average noise time
bifdiagY = [0, 1]
heatmap = np.array(noise_times).reshape(len(a_values), len(eps_values))

heatmap_T = heatmap.T 

cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_blues",
    [
        (0.0, "white"),
        (0.01, "lightblue"),
        (0.1, "skyblue"),
        (0.2, "deepskyblue"),
        (0.3, "dodgerblue"),    
        (0.5, "royalblue"), 
        (0.75, "mediumblue"), 
        (1.0, "darkblue")     
    ]
)

plt.figure(figsize=(10, 7))
im = plt.imshow(heatmap_T, extent=[a_values[0], a_values[-1], eps_values[0], eps_values[-1]], aspect='auto', cmap=cmap, origin='lower')

plt.xlabel("a", fontsize=10)
plt.ylabel("ε", fontsize=10)
plt.xticks(np.arange(1.4, 2.01, 0.1))
plt.xlim(1.4, 2.0)
plt.ylim(0, 0.4)
plt.grid(True, linestyle='--', alpha=0.5)
cbar = plt.colorbar(im)
cbar.set_label('Average time of wandering between clusters', fontsize=10)
plt.tight_layout()
plt.show()

# plot of variance of the first principal component in PCA
heatmap = np.array(variances_PCA).reshape(len(a_values), len(eps_values))

heatmap_T = heatmap.T 

cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_blues",
    [
        (0.0, "white"),
        (0.03, "lightblue"),
        (0.35, "skyblue"),
        (0.65, "deepskyblue"),
        (0.95, "royalblue"),
        (1.0, "darkblue")
    ]
)

# Utwórz figurę z precyzyjnym układem
plt.figure(figsize=(10, 7))
im = plt.imshow(heatmap_T, extent=[a_values[0], a_values[-1], eps_values[0], eps_values[-1]], aspect='auto', cmap=cmap, origin='lower')

plt.xlabel("a", fontsize=10)
plt.ylabel("ε", fontsize=10)
plt.xticks(np.arange(1.4, 2.01, 0.1))
plt.xlim(1.4, 2.0)
plt.ylim(0, 0.4)
plt.grid(True, linestyle='--', alpha=0.5)
plt.text(0.23, 0.91, "(a)", transform=plt.gca().transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color="yellow")
plt.text(0.68, 0.45, "(b)", transform=plt.gca().transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color="yellow")
cbar = plt.colorbar(im)
cbar.set_label('Minimum variance captured by the first principal\ncomponent among all identified clusters', fontsize=10)
plt.show()
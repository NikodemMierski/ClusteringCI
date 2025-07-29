#!/usr/bin/env python3

"""
CI_dbscan: gaussian_dbscan
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from collections import defaultdict
import matplotlib.lines as mlines
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

# parameters 
alpha = 12
beta = -0.504
epsilon = -0.0855

# time start and end
t_start = 50000
t_end = 70000

# begin with pseudo-random initial conditions
np.random.seed(11)
x = np.random.random ((2,))
x1=x[0]
x2=x[1]
x1_series, x2_series = [], []

# Mutually Coupled Gaussian Maps
def gaussian_map(x, x_coupled, alpha, beta, epsilon):
    return np.exp(-alpha * x**2) + beta + epsilon * (x_coupled - x)

# iterate the model
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

# plot the graph in the first two dimensions
plt.figure(figsize=(8, 8))
plt.scatter(x1_series, x2_series,color="#0000FF55", s=1)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()

# time series
plt.figure(figsize=(10, 5))
plt.plot(range(t_start, t_end), x1_series, label='$x_1$', alpha=0.8, linewidth=1.5, color='crimson')
plt.plot(range(t_start, t_end), x2_series, label='$x_2$', alpha=0.8, linewidth=1.5, color='dodgerblue')
plt.xlabel("Time")
plt.ylabel("$x_1,x_2$")
plt.legend()
plt.show()

df = pd.DataFrame(data)

# DBSCAN
dbscan = DBSCAN(eps=0.035, min_samples=200)
labels = dbscan.fit_predict(df)

# add a cluster column 
df['Cluster'] = labels

# number of clusters without noise
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Number of the clusters (without noise): {n_clusters}")

# calculate Silhouette Score
silhouette_avg = silhouette_score(df[df['Cluster'] != -1].iloc[:, :-1], labels[labels != -1])
print(f"Silhouette Score: {silhouette_avg:.2f}")

# calculate noise ratio 
noise_ratio = np.sum(labels == -1) / len(labels)
print(f"Noise ratio: {noise_ratio:.2%}\n")

# number of points in every cluster
clusters  = {
    'Cluster': list(set(list(labels))),
    'Number of points': [list(labels).count(x) for x in list(set(list(labels)))]
}

clusters = pd.DataFrame(clusters)
print(clusters.to_string(index = False))

# plot with found clusters
plt.figure(figsize=(8, 8))

red = [0.8, 0.0, 0.0, 1.0]
navy = [0.0, 0.0, 0.5, 1.0]
gray = [0.3, 0.3, 0.3, 0.7]

colors = np.zeros((len(labels), 4))

for i, label in enumerate(labels):
    if label == -1:
        colors[i] = gray
    elif label == 0:
        colors[i] = red
    elif label == 1:
        colors[i] = navy

plt.scatter(df[0], df[1], c=colors, s=3)

plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)

legend_elements = [
    mlines.Line2D([], [], color=red, marker='o', linestyle='None', markersize=6, label='Cluster 0'),
    mlines.Line2D([], [], color=navy, marker='o', linestyle='None', markersize=6, label='Cluster 1'),
    mlines.Line2D([], [], color=[0.5, 0.5, 0.5, 1], marker='o', linestyle='None', markersize=6, label='Noise')
]

plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
plt.show()

# transition matrix
unique_labels = sorted(set(labels))  
letter_to_index = {letter: idx for idx, letter in enumerate(unique_labels)} 

adjacency_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

for i in range(len(labels) - 1):
    current_letter = labels[i]
    next_letter = labels[i + 1]
    adjacency_matrix[letter_to_index[current_letter], letter_to_index[next_letter]] += 1

transition_matrix = pd.DataFrame(adjacency_matrix, index=unique_labels, columns=unique_labels)

print("\n",transition_matrix)


print("\nPotential merging")

# filtered list of labels excluding -1
valid_labels = [label for label in labels if label != -1]
unique_labels = sorted(set(valid_labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

# build transition matrix only for valid labels (without -1)
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
    result = []
    for i, row in df.iterrows():
        row_wo_self = row.copy()
        row_wo_self[i] = 0
        result.append(any(row_wo_self > 0.8 * row.sum()))
    return result

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

    # update labels: assign minimum label in group (without -1)
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

    print("\nMerged clusters:", merged_groups, ". Number of clusters after merging:", len(set(labels)) - (1 if -1 in labels else 0),"\n")

else:
    print("No clusters merged.\n")


clusters = {
    'Cluster': list(set(list(labels))),
    'Number of points': [list(labels).count(x) for x in list(set(list(labels)))]
}

clusters = pd.DataFrame(clusters)
print(clusters.to_string(index = False))

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

# average time in clusters
def average_time(times):
    lengths = defaultdict(list)
    
    for cluster, time in times:
        lengths[cluster].append(time)
    
    average_times = {cluster: np.mean(lengths[cluster]) for cluster in lengths}
    return average_times

print("\nAverage time in clusters:\n")
for cluster, average in average_time(time_in_cluster(labels)).items():
    print(f"Cluster: {cluster:2d};  average time:  {average:.2f}")

# median time in clusters
def median_time(times):
    lengths = defaultdict(list)
    
    for cluster, time in times:
        lengths[cluster].append(time)
    
    median_times = {cluster: np.median(lengths[cluster]) for cluster in lengths}
    return median_times

print("\nMedian time in clusters:\n")
for cluster, median in median_time(time_in_cluster(labels)).items():
    print(f"Cluster: {cluster:2d};  median time:  {median:.2f}")

# standard deviation of time in clusters
def standard_deviation_of_times(sequence):
    times = time_in_cluster(sequence)

    groups = {}
    for cluster, length in times:
        groups.setdefault(cluster, []).append(length)

    result = {cluster: np.std(lengths, ddof=0) for cluster, lengths in groups.items()}
    return result

print("\nStandard deviation of time in clusters:\n")
for cluster, deviation in standard_deviation_of_times(labels).items():
    print(f"Cluster: {cluster:2d}; standard deviation of time in cluster: {deviation:.2f}")

# number of visits 
def count_visits(sequence):
    visits = {}
    previous = None

    for number in sequence:
        if number != previous:
            if number not in visits:
                visits[number] = 0
            visits[number] += 1
        previous = number

    return visits

print("\nNumber of visits in clusters:\n")
for cluster, count in count_visits(labels).items():
    print(f"Cluster: {cluster:2d}; number of visits: {count}")

# sequence of consecutively visited clusters
def sequence_of_clusters(sequence):
    sequence_clusters = sequence[np.append(True, sequence[1:] != sequence[:-1])]
    return sequence_clusters

# plot of cluster membership
a, b = 4700, 5200
x_values = np.arange(t_start, t_end)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), sharex=True)

axes[0].plot(x_values[a:b], data[a:b, 0], label='$x_1$', linewidth=1, color='#2ca02c', alpha=0.8)
axes[0].plot(x_values[a:b], data[a:b, 1], label='$x_2$', linewidth=1, color='#f4a300', alpha=0.8)
axes[0].set_ylabel('$x_1, x_2$', fontsize=11)
axes[0].legend(fontsize=11, loc='upper right')
axes[0].grid(zorder=0)

red = [0.8, 0.0, 0.0, 1.0]
blue = [0.0, 0.0, 0.5, 1.0]
gray = [0.5, 0.5, 0.5, 1]

label_colors = {
    -1: gray,
     0: blue,
     1: red
}

for label_val, color in label_colors.items():
    mask = labels[a:b] == label_val
    axes[1].scatter(x_values[a:b][mask], labels[a:b][mask], color=color, s=12, zorder=3)

axes[1].set_yticks([-1, 0, 1])
axes[1].set_yticklabels(["noise", "cluster 0", "cluster 1"], fontsize=11)
axes[1].set_xlabel('Time', fontsize=11)
axes[1].grid(zorder=0)
axes[1].set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()

# adding isolated points in the cluster membership plot to the cluster to which the surrounding points belong
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
    
    return lst

labels = include_isolated_points(labels)

consecutive_clusters = sequence_of_clusters(np.array(labels))

consecutive_clusters_without_noise = consecutive_clusters[consecutive_clusters != -1]

print("\nAverage time in clusters after adding isolated points in the cluster membership plot:\n")
for cluster, average in average_time(time_in_cluster(labels)).items():
    print(f"Cluster: {cluster:2d};  average time:  {average:.2f}")


print("\nMedian time in clusters after adding isolated points in the cluster membership plot:\n")
for cluster, median in median_time(time_in_cluster(labels)).items():
    print(f"Cluster: {cluster:2d};  median time:  {median:.2f}")

print("\nStandard deviation of time in clusters after adding isolated points in the cluster membership plot:\n")
for cluster, deviation in standard_deviation_of_times(labels).items():
    print(f"Cluster: {cluster:2d}; standard deviation of time in cluster: {deviation:.2f}")


print("\nNumber of visits in clusters after adding isolated points in the cluster membership plot:\n")
for cluster, count in count_visits(labels).items():
    print(f"Cluster: {cluster:2d}; number of visits: {count}")


# Ljung-Box Test
def ljung_box_test(sequence):
    clusters = time_in_cluster(sequence)
    times = [length for _, length in clusters if _ != -1]

    result = acorr_ljungbox(times, lags=[10], return_df=True)
    pval = result['lb_pvalue'].iloc[0]

    return pval

print("\nLjung-Box Test p-value:", ljung_box_test(labels))

# Augmented Dickey-Fuller Test 
def adf(sequence):
    clusters = time_in_cluster(sequence)
    times = [length for _, length in clusters if _ != -1]

    result = adfuller(times)
    p_value = result[1]

    return p_value

print("\nAugmented Dickey-Fuller Test p-value:", adf(labels))

# Runs Test
def runs(sequence):
    a, p = runstest_1samp(sequence)
    return p

print("\nRuns Test p-value:", runs(consecutive_clusters_without_noise))
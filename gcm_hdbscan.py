#!/usr/bin/env python3

"""
CI_dbscan: gcm_hdbscan
Version 2.1, October 17, 2025.
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
import hdbscan
from sklearn.metrics import silhouette_score
from collections import defaultdict
import matplotlib.lines as mlines
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import math
from scipy.stats import chi2
from statsmodels.sandbox.stats.runs import runstest_1samp
from sklearn.decomposition import PCA

def f(a,epsilon,x):
	"""Computation of the next vector in the GCM system (Kaneko 1990)."""
	fx = 1 - a * x * x
	return (1 - epsilon) * fx + epsilon / len(x) * fx.sum()

# the number of globally coupled maps
N = 3
# the coupling term (should be between 0 and 1)
epsilon = 0.2574

# the parameter of the logistic map
a = 2

# time start and end
t_start = 20000
t_end = 40000

# begin with pseudo-random initial conditions
np.random.seed (3)
x = np.random.random ((N,))

# iterate the initial number of times
for i in range(t_start):
    x = f(a,epsilon,x)

# prepare an array for the points to plot
data = np.empty ((t_end - t_start,N,))

# continue iterating and store the vectors in the array
for i in range(t_end - t_start):
    data[i,:] = x
    x = f(a,epsilon,x)

# plot the graph in the first two dimensions
plt.figure(figsize=(8,8))
plt.scatter(data[:,0],data[:,1],color="#0000FF55", s=1)
plt.xlabel('$x_n(1)$')
plt.ylabel('$x_n(2)$')
plt.tight_layout ()

# plot the time series of the first variable
x_values = np.arange(t_start, t_end)
plt.figure(figsize=(10, 6))  
plt.plot(x_values, data[:,0], label='x[0]', linewidth = 0.3, alpha=0.7)  
plt.xlabel('Time')
plt.ylabel('$x_n(1)$')
plt.grid()
plt.tight_layout()

df = pd.DataFrame(data)

# HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=300)
labels = clusterer.fit_predict(data)

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

custom_colors = [
    (230/255, 25/255, 75/255, 1.0),
    (60/255, 180/255, 75/255, 1.0),
    (67/255,  99/255, 216/255, 1.0),
    (245/255,130/255, 49/255, 1.0),
    (145/255, 30/255, 180/255, 1.0),
    (240/255,50/255, 230/255, 1.0),
    (0/255, 128/255, 128/255, 1.0),
    (67/255, 99/255, 150/255, 1.0),
    (0/255, 0/255, 128/255, 1.0),
    (128/255, 0/255, 128/255, 1.0),
    (150/255, 75/255, 0/255, 1.0),
    (204/255, 153/255, 0/255, 1.0)
]

unique_labels = sorted(set(labels) - {-1})
label_color_map = {label: custom_colors[i] for i, label in enumerate(unique_labels)}

colors = np.zeros((len(labels), 4))
for i, label in enumerate(labels):
    if label == -1:
        colors[i] = [0.3, 0.3, 0.3, 0.4]  
    else:
        colors[i] = label_color_map[label]

plt.scatter(df[0], df[1], c=colors, s=3)
plt.xlabel('$x_n(1)$', fontsize=10)
plt.ylabel('$x_n(2)$', fontsize=10)

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

    print("\nMerged clusters:", merged_groups, ". Number of clusters after merging:", len(set(labels)) - (1 if -1 in labels else 0),"\n")

else:
    print("No clusters merged.\n")

# plot with clusters after merging
plt.figure(figsize=(8, 8))
red = [1, 0, 0, 1]
navy = [0, 0, 0.5, 1]
yellow = [1, 0.598, 0.155, 1]
gray = [0.3, 0.3, 0.3, 0.4]

colors = np.zeros((len(labels), 4))

for i, label in enumerate(labels):
    if label == -1:
        colors[i] = gray
    elif label == 0:
        colors[i] = red
    elif label == 1:
        colors[i] = navy
    elif label == 2:
        colors[i] = yellow

scatter = plt.scatter(df[0], df[1], c=colors, s=3)

plt.xlabel('$x_n(1)$', fontsize=10)
plt.ylabel('$x_n(2)$', fontsize=10)

legend_elements = [
    mlines.Line2D([], [], color=red, marker='o', linestyle='None', markersize=6, label='Attractor ruin 0'),
    mlines.Line2D([], [], color=navy, marker='o', linestyle='None', markersize=6, label='Attractor ruin 1'),
    mlines.Line2D([], [], color=yellow, marker='o', linestyle='None', markersize=6, label='Attractor ruin 2'),
    mlines.Line2D([], [], color=[0.5, 0.5, 0.5, 1], marker='o', linestyle='None', markersize=6, label='Noise')
]

plt.legend(handles=legend_elements, loc='best', fontsize=10)

plt.scatter(df[0], df[1], c=colors, s=1)
plt.xlabel('$x_n(1)$')
plt.ylabel('$x_n(2)$')
plt.show()

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
a, b = 13700, 14700
x_values = np.arange(t_start, t_end)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 7), sharex=True)

axes[0].plot(x_values[a:b], data[a:b, 0], label='x[0]', linewidth=1)
axes[0].set_ylabel('$x_n(1)$')
axes[0].grid()

red = [1, 0, 0, 1]
blue = [0, 0, 0.5, 1]
orange = [1, 0.598, 0.155, 1]
gray = [0.5, 0.5, 0.5, 1]

label_colors = {
    -1: gray,
     0: red,
     1: blue,
     2: orange
}

for label_val, color in label_colors.items():
    mask = labels[a:b] == label_val
    axes[1].scatter(x_values[a:b][mask], labels[a:b][mask], color=color, s=12, zorder=3)

axes[1].set_yticks([-1, 0, 1, 2])
axes[1].set_yticklabels(["noise", "attractor ruin 0", "attractor ruin 1", "attractor ruin 2"], fontsize=10)
axes[1].set_xlabel('Time', fontsize=10)
axes[1].grid()
axes[1].set_ylim(-1.5, 2.5)
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

if n_clusters < 2 or noise_ratio < 0.1 or noise_ratio > 0.9:
    print("Lack of chaotic itinerancy")

else:
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

    if n_clusters == 2:
        # Runs Test
        def runs_WW(sequence):
            a, p = runstest_1samp(sequence)
            return p

        print("\nRuns Test p-value:", runs_WW(consecutive_clusters_without_noise))

    else:
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

        print("\nO\’Brien-Dyck Runs Test p-value:", runs_OBD(consecutive_clusters_without_noise))

    df['Cluster'] = labels

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

    print("\nMinimal variance captured by the first principal component among the attractor ruins:", var_pca(df))
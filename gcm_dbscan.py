#!/usr/bin/env python3

"""
CI_dbscan: gcm_dbscan
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from collections import Counter
from collections import defaultdict
from scipy.stats import entropy

def f(a,epsilon,x):
	"""Computation of the next vector in the GCM system (Kaneko 1990)."""
	fx = 1 - a * x * x
	return (1 - epsilon) * fx + epsilon / len(x) * fx.sum()

# the number of globally coupled maps
N = 5
# the coupling term (should be between 0 and 1)
epsilon = 0.234

# the parameter of the logistic map
a = 2

# time start and end
t_start = 10000
t_end = 30000

# begin with pseudo-random initial conditions
np.random.seed (7)
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

# DBSCAN
dbscan = DBSCAN(eps=0.04, min_samples=185)
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
cmap = plt.cm.plasma

# create a color array
colors = np.zeros((len(labels), 4)) 
for i, label in enumerate(labels):
    if label == -1:
        colors[i] = [0, 0, 1, 0.33]  
    else:
        normalized_label = (label - labels.min()) / (labels.max() - labels.min())
        colors[i] = cmap(normalized_label) 

plt.scatter(df[0], df[1], c=colors, s=1)
plt.xlabel('$x_n(1)$')
plt.ylabel('$x_n(2)$')
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

# plot with clusters after merging
plt.figure(figsize=(8, 8))
cmap = plt.cm.plasma

# create a color array
colors = np.zeros((len(labels), 4))  
for i, label in enumerate(labels):
    if label == -1:
        colors[i] = [0, 0, 1, 0.33]  
    else:
        normalized_label = (label - labels.min()) / (labels.max() - labels.min())
        colors[i] = cmap(normalized_label) 

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

# sequence of consecutively visited clusters
def sequence_of_clusters(sequence):
    sequence_clusters = sequence[np.append(True, sequence[1:] != sequence[:-1])]
    return sequence_clusters

consecutive_clusters = sequence_of_clusters(np.array(labels))

consecutive_clusters_without_noise = consecutive_clusters[consecutive_clusters != -1]

# entropy 
def calculate_entropy(sequence):
    symbol_counts = Counter(sequence)
    probabilities = [count / len(sequence) for count in symbol_counts.values()]
    entropy_value = entropy(probabilities, base=2)
    
    return entropy_value

print("\nEntropy of the sequence of consecutively visited clusters:", calculate_entropy(consecutive_clusters_without_noise),"\n")

# average time in noise after leaving a given cluster
def average_time_noise(sequnce):
    results = {}
    for i in range(len(sequnce)-1):
        if sequnce[i] != -1:  
            count_minus_1 = 0
            j = i + 1
            while j < len(sequnce) and sequnce[j] == -1:
                count_minus_1 += 1
                j += 1
            
            if count_minus_1 > 0:
                if sequnce[i] not in results:
                    results[sequnce[i]] = []
                results[sequnce[i]].append(count_minus_1)

    for x in results:
        average = sum(results[x]) / len(results[x])
        print(f"Cluster: {x}, average number of -1 after leaving: {average:.2f}")

average_time_noise(labels)

# plot of cluster membership
a, b = 12600, 13800
x_values = np.arange(t_start, t_end)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 7), sharex=True)

axes[0].plot(x_values[a:b], data[a:b, 0], label='x[0]', linewidth=1, color='red')
axes[0].set_ylabel('$x_n(1)$')
axes[0].grid()

axes[1].scatter(x_values[a:b], labels[a:b], marker='o', s=30, color='orange')
axes[1].set_yticks([-1, 0, 1])
axes[1].set_yticklabels(["noise", "cluster 0", "cluster 1"])
axes[1].set_xlabel('Time')
axes[1].grid()

plt.ylim(-2,2)
plt.tight_layout()
plt.show()

# adding isolated points in the cluster membership plot to the cluster to which the surrounding points belong
def replace_inner_sequences(lst):
    lst = list(lst)  
    changed = np.zeros(len(lst), dtype=int) 
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
                    changed[k] = 1 
        
        i = j  
    
    return lst, changed.tolist() 

labels, changes = replace_inner_sequences(labels)

print("\nAverage time in clusters after adding isolated points in the cluster membership plot:\n")
for cluster, average in average_time(time_in_cluster(labels)).items():
    print(f"Cluster: {cluster:2d};  average time:  {average:.2f}")
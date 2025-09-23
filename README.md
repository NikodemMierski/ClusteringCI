# Analysis of the Chaotic Itinerancy Phenomenon using Entropy and Clustering

This repository contains code for analyzing chaotic itinerancy for globally coupled logistic maps using entropy-based measures and the HDBSCAN clustering algorithm. The program identifies attractor ruins visited by the system and analyzes their structure and transitions.

This code accompanies the article (in preparation) titled:  
**"Analysis of the Chaotic Itinerancy Phenomenon using Entropy and Clustering"**

### Files:

- gcm_hdbscan.py
- gcm_entropy.py
- gcm_analysis.py

### The file gcm_hdbscan.py:

- simulates the dynamics of the system for given parameters,
- applies the HDBSCAN clustering algorithm to identify dense clusters,
- analyzes transitions between identified clusters and merge two clusters into a single cluster if at least 80% of the points from the first cluster transition into the second,
- generates visualizations of time series, phase portraits, and cluster membership over time,
- computes the average time in each cluster, median time in each cluster, standard deviation of time in each cluster and number of visits in each cluster and the same measures after including isolated points shown on the cluster membership plot,
- performs test to check whether the observed itinerancy is chaotic.
  

### The file gcm_entropy.py:

- finds the values of epsilon for which the variance of local Shannon entropy is the highest,
- finds the values of epsilon for which the variance of local permutation entropy is the highest,
- plots the variance of local Shannon entropy and the variance of local permutation entropy as a function of epsilon.

### The file gcm_analysis.py:

- performs an automated method for detecting chaotic itinerancy on grid of parameters in the GCM model,
- computes the average time in noise, the fraction of the total variance captured by the first principal component obtained through PCA and numer of randomness testes thare are passed,
- visualizes these measures as heatmaps on the parameter grid.

### Remarks:

- In gcm_hdbscan.py, the DBSCAN parameters are predefined for the considered parameter values of these models.

- In gcm_hdbscan.py, the considered epsilon values correspond to those with one of the highest variances of local entropy identified in gcm_entropy.py.


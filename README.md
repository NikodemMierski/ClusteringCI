# DBSCAN and Entropy for Chaotic Itinerancy Analysis

This repository contains code for analyzing chaotic itinerancy for globally coupled maps and mutually coupled Gaussian maps using entropy-based measures and the DBSCAN clustering algorithm. The program identifies attractor ruins visited by the system and analyzes their structure and transitions.

This code accompanies the article (in preparation) titled:  
**"Analysis of the Chaotic Itinerancy Phenomenon using Entropy and Clustering"**

### Files:

- gcm_dbscan.py
- gaussian_dbscan.py
- gcm_entropy.py
- gaussian_entropy.py
- gcm_hdbscan.py


### The files gcm_dbscan.py and gaussian_dbscan.py:

- simulate the dynamics of the system for given parameters,
- apply the DBSCAN clustering algorithm to identify dense clusters,
- analyze transitions between identified clusters and merge two clusters into a single cluster if at least 80% of the points from the first cluster transition into the second,
- generate visualizations of time series, phase portraits, and cluster membership over time,
- compute the average time in each cluster and the average time in each cluster after including isolated points shown on the cluster membership plot.

### The files gcm_entropy.py and gaussian_entropy.py:

- find the values of epsilon for which the variance of local Shannon entropy is the highest,
- plot the variance of local Shannon entropy as a function of epsilon for constant N = 5 and constant a = 2.

### The file gcm_hdbscan.py:

- performs an automated method for detecting chaotic itinerancy on grid of parameters in the GCM model,
- computes the average time in noise and the fraction of the total variance captured by the first principal component obtained through PCA,
- visualizes both measures as heatmaps on the parameter grid.

### Remarks:

- In gcm_dbscan.py and gauss_dbscan.py, the DBSCAN parameters are predefined for the considered parameter values of these models.

- In gcm_dbscan.py and gaussian_dbscan.py, the considered epsilon values correspond to those with one of the highest variances of local entropy identified in gcm_entropy.py and gaussian_entropy.py.


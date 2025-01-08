import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import networkx as nx
import random
import math
import matplotlib.pyplot as plt



# Step 1: Generate random positions for the nodes in a plane, separated into clusters
np.random.seed(42)

# Define cluster centers
cluster_centers = np.array([[-1, -1], [0.7, 0.3], [1, 1]])
# 50%, 18%, 32%
# Generate points around the cluster centers
# points_per_cluster = [10, 4, 6]  # Assuming 20 total nodes
points_per_cluster = [25, 9, 16]  # Assuming 50 total nodes
points_per_cluster = [20, 7, 13]  # Assuming 40 total nodes
points_per_cluster = [30, 11, 19]  # Assuming 60 total nodes
points_per_cluster = [35, 13, 22]  # Assuming 70 total nodes
points_per_cluster = [40, 14, 26]  # Assuming 70 total nodes

# points_per_cluster = [10, 4, 6]  # Assuming 100 total nodes

positions = []
for i, center in enumerate(cluster_centers):
    noise = np.random.rand(points_per_cluster[i], 1)  # No noise added
    points = center + 0.1*noise
    positions.append(points)

positions = np.concatenate(positions)
plt.scatter(positions[:25,0], positions[:25,1])
# plt.scatter(positions[25:34,0], positions[25:34,1])
# plt.scatter(positions[34:,0], positions[34:,1])
# plt.show()
#%%
# Step 2: Calculate the pairwise distances between nodes
distances = squareform(pdist(positions))
distances = np.exp(-distances ** 2)

# Step 3: Define the adjacency matrix based on a threshold
threshold = 0.4
adjacency = np.where(distances < threshold, 0, distances)

# Step 4: Calculate the degree matrix
degree = np.diag(np.sum(adjacency, axis=1))

# Step 5: Calculate the Laplacian matrix as an estimation of the precision matrix
laplacian = degree - adjacency

# Step 6: Calculate the covariance matrix as the pseudoinverse of the precision matrix
covariance_matrix = np.linalg.pinv(laplacian)

print("Adjacency matrix:")
print(pd.DataFrame(adjacency))
print("\nDegree matrix:")
print(pd.DataFrame(degree))
print("\nLaplacian matrix (Estimation of Precision matrix):")
print(pd.DataFrame(laplacian))
print(pd.DataFrame(np.round(laplacian,2)))
print("\nCovariance matrix:")
print(pd.DataFrame(covariance_matrix))

#%%
np.random.seed(42)
num_sensors = 80
num_years = 6
num_measurements_per_day = 24
total_measurements = num_years * 365 * num_measurements_per_day

# Generate random mean and covariance for M0
mean_M0 = np.random.uniform(low=55, high=65, size=num_sensors)

# Generate random mean and covariance for A0
mean_A0 = np.random.uniform(low=10, high=35, size=num_sensors)
mean_phi = np.zeros(num_sensors)

# Generate random samples for M0, A0, and phi
M0_samples = np.random.multivariate_normal(mean_M0, covariance_matrix, total_measurements)
A0_samples = np.random.multivariate_normal(mean_A0, covariance_matrix, total_measurements)
phi_samples = np.random.multivariate_normal(mean_phi, covariance_matrix, total_measurements)

# Generate time series data
time_series_data = np.zeros((num_sensors, total_measurements))
D = 0

for i in range(num_sensors):
    for j in range(total_measurements):
        t = j % num_measurements_per_day  # Get the time within a day (0-23)
        D += 1 if t == 0 else 0  # Increase D by 1 when a new day starts
        M0 = M0_samples[j, i] * (1 + 0.8* np.sin(2 * np.pi * D / 365))
        A0 = A0_samples[j,i] * (1 + 0.7 * np.sin(2 * np.pi * D / 365))
        phi = phi_samples[j, i]
        noise = 0.5 * np.random.normal(0, 1)  # Generate white noise
        time_series_data[i, j] = M0 + A0 * np.sin((2 * np.pi * t / num_measurements_per_day) + 0.1*phi) + noise
a = pd.DataFrame(time_series_data.T)
plt.plot(a.iloc[6000:6048, 0])
plt.show()

plt.plot(a.iloc[:, 0])
plt.plot(a.iloc[:48, 5])
plt.plot(a.iloc[:48, 7])
plt.plot(a.iloc[:48, 8])
plt.plot(a.iloc[:48, 10])
plt.plot(a.iloc[:48, 13])
plt.plot(a.iloc[:48, 1])
plt.plot(a.iloc[:48, 9])
plt.plot(a.iloc[:48, 11])
plt.plot(a.iloc[:48, 12])
plt.plot(a.iloc[:48, 2])
plt.plot(a.iloc[:48, 3])
plt.plot(a.iloc[:48, 4])
plt.plot(a.iloc[:48, 6])
plt.plot(a.iloc[:48, 16])
plt.plot(a.iloc[:48, 19])
plt.plot(a.iloc[:48, 14])
plt.plot(a.iloc[:48, 15])
plt.plot(a.iloc[:48, 18])
plt.plot(a.iloc[:48, 17])
plt.show()

start_date = '2017-01-01'
end_date = '2022-12-31'
date_range = pd.date_range(start=start_date, end=end_date, freq='H')
date_range = date_range[:-1]

# Create a new DataFrame with the data and the date index
df = pd.DataFrame(time_series_data.T, index=date_range)

# Assuming your DataFrame is called 'df'
old_column_names = df.columns.tolist()  # Get the current column names

# Generate new column names
new_column_names = ['date']
new_column_names += [f'sensor{i}' for i in range(1, 101)]

# Rename the columns
df.rename(columns=dict(zip(old_column_names, new_column_names)), inplace=True)

# Verify the new column names
print(df.columns)
# df.to_csv('SyntheticDataSezonality_80Sensors.csv')
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

def generate_gaussian_blobs_with_labels(num_clusters, points_per_cluster):
    data = []
    labels = []
    for cluster_id in range(num_clusters):
        # Randomly choose cluster center
        center_x, center_y = np.random.uniform(-10, 10, 2)
        # Randomly choose standard deviation
        std_dev = np.random.uniform(0.5, 2.5)
        # Generate points for the cluster
        points = np.random.normal(loc=(center_x, center_y), scale=std_dev, size=(points_per_cluster, 2))
        data.append(points)
        labels.extend([cluster_id] * points_per_cluster)  # Assign cluster labels
    return np.vstack(data), np.array(labels)



# Set a fixed random seed for reproducibility
np.random.seed(42)

# Parameters
num_clusters = 6
points_per_cluster = 200

# Generate data and labels with the same seed for consistent clusters
data, labels = generate_gaussian_blobs_with_labels(num_clusters, points_per_cluster)

# Plot the data with different colors for each cluster
plt.figure(figsize=(8, 8))
for cluster_id in range(num_clusters):
    cluster_points = data[labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, alpha=0.7, label=f'Cluster {cluster_id + 1}')

plt.title("2D Gaussian Blobs with Varying Cluster Density (Fixed Random Seed)", fontsize=14)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend(title="Clusters", loc="upper right")
plt.grid(True)
plt.axis('equal')
plt.savefig('synthetic.png')

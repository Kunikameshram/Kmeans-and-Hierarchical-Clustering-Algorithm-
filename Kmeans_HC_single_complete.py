import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns

# Fix the random seed for reproducibility
random.seed(0)

class KMeansClustering:
    def __init__(self, n_clusters=3, max_iterations=100, random_state=None):
        self.k = n_clusters
        self.max_iterations = max_iterations
        self.clusters = {}
        self.centroids = {}
        self.cluster_prefix = "cluster_"
        self.centroid_prefix = "centroid_"
        self.error_history = []
        self.labels_ = None
        if random_state is not None:
            np.random.seed(random_state)
            
    # Calculates the Euclidean distance between two points
    def _calculate_euclidean_distance(self, point1, point2):
        
        return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))
    
    # Calculate the distance from each point to all centers
    def _get_distances_to_centers(self, data, centers):
        distances = np.zeros((len(data), len(centers)))
        for i, point in enumerate(data):
            for j, center in enumerate(centers):
                distances[i, j] = self._calculate_euclidean_distance(point, center)
        return distances
    
    # initial centroid
    def _initialize_centroids(self, data):
        
        data = np.array(data)
        n_samples = len(data)
        centroids = []

        # Choose first centroid randomly
        first_centroid_idx = np.random.randint(n_samples)
        centroids.append(data[first_centroid_idx])

        # Choose remaining k-1 centroids
        for _ in range(1, self.k):
            distances = self._get_distances_to_centers(data, centroids)
            min_distances = np.min(distances, axis=1)
            squared_distances = min_distances ** 2
            # Choose next centroid based on probability distribution
            probs = squared_distances / squared_distances.sum()
            next_centroid_idx = np.random.choice(n_samples, p=probs)
            centroids.append(data[next_centroid_idx])

        self.centroids.clear()
        for i, centroid in enumerate(centroids, 1):
            self.centroids[self.centroid_prefix + str(i)] = centroid.tolist()
    
    def _assign_clusters(self, data):
        self.clusters.clear()
        cluster_assignments = []
        
        for dp in data:
            min_distance = float('inf')
            nearest_cluster = None
            for centroid_key, centroid in self.centroids.items():
                distance = self._calculate_euclidean_distance(dp, centroid)
                if distance < min_distance:
                    min_distance = distance
                    nearest_cluster = self.cluster_prefix + centroid_key.split("_")[1]
            
            if nearest_cluster not in self.clusters:
                self.clusters[nearest_cluster] = []
            self.clusters[nearest_cluster].append(dp)
            cluster_assignments.append(int(nearest_cluster.split("_")[1]) - 1)
        
        self.labels_ = np.array(cluster_assignments)
    
    def _update_centroids(self):
        self.centroids.clear()
        for cluster in self.clusters:
            cluster_points = np.array(self.clusters[cluster])
            centroid = cluster_points.mean(axis=0)
            self.centroids[self.centroid_prefix + cluster.split("_")[1]] = centroid.tolist()
    
    def _calculate_sse(self):
        total_error = 0.0
        for cluster in self.clusters:
            for dp in self.clusters[cluster]:
                centroid = self.centroids[self.centroid_prefix + cluster.split("_")[1]]
                total_error += self._calculate_euclidean_distance(dp, centroid) ** 2
        return total_error
    
    def fit(self, X):
        self._initialize_centroids(X)
        
        for _ in range(self.max_iterations):
            old_centroids = dict(self.centroids)
            self._assign_clusters(X)
            self._update_centroids()
            
            # Calculate error and check convergence
            error = self._calculate_sse()
            self.error_history.append(error)
            
            # Check if centroids have stabilized
            if old_centroids == self.centroids:
                break
        
        return self
    
    def predict(self, X):
        predictions = []
        for point in X:
            min_distance = float('inf')
            closest_cluster = None
            for centroid_key, centroid in self.centroids.items():
                distance = self._calculate_euclidean_distance(point, centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = int(centroid_key.split("_")[1]) - 1
            predictions.append(closest_cluster)
        return np.array(predictions)

class HierarchicalClustering:
    def __init__(self, linkage_type='single'):
        if linkage_type not in ['single', 'complete']:
            raise ValueError("linkage_type must be 'single' or 'complete'")
        self.linkage_type = linkage_type
        self.labels_ = None
        self.linkage_matrix = []
        self.n_samples = None
        
    # Calculate Euclidean distance between all points in two clusters
    def _calculate_distance(self, cluster1, cluster2):
        cluster1 = np.array(cluster1)
        cluster2 = np.array(cluster2)
        
        if cluster1.ndim == 1:
            cluster1 = cluster1.reshape(1, -1)
        if cluster2.ndim == 1:
            cluster2 = cluster2.reshape(1, -1)
            
        distances = np.zeros((len(cluster1), len(cluster2)))
        for i, point1 in enumerate(cluster1):
            for j, point2 in enumerate(cluster2):
                distances[i, j] = np.sqrt(np.sum((point1 - point2) ** 2))
        
        if self.linkage_type == 'single':
            return np.min(distances)
        else:  # complete linkage
            return np.max(distances)
    
    def fit(self, X):
        self.n_samples = X.shape[0]
        self.linkage_matrix = []
        
        # Initialize clusters
        clusters = [np.array([X[i]]) for i in range(self.n_samples)]
        active_clusters = list(range(self.n_samples))
        next_cluster_id = self.n_samples
        
        while len(active_clusters) > 1:
            n_clusters = len(active_clusters)
            distances = np.full((n_clusters, n_clusters), np.inf)
            
            # Calculate distances between all active clusters
            for i in range(n_clusters - 1):
                for j in range(i + 1, n_clusters):
                    distances[i, j] = self._calculate_distance(
                        clusters[active_clusters[i]], 
                        clusters[active_clusters[j]]
                    )
            
            # Find minimum distance pair
            min_i, min_j = np.unravel_index(np.argmin(distances), distances.shape)
            min_dist = distances[min_i, min_j]
            
            cluster1_idx = active_clusters[min_i]
            cluster2_idx = active_clusters[min_j]
            
            # Record merge
            self.linkage_matrix.append([
                float(cluster1_idx),
                float(cluster2_idx),
                float(min_dist),
                float(len(clusters[cluster1_idx]) + len(clusters[cluster2_idx]))
            ])
            
            # Merge clusters
            new_cluster = np.vstack((clusters[cluster1_idx], clusters[cluster2_idx]))
            clusters.append(new_cluster)
            
            # Update active clusters
            active_clusters = [idx for idx in active_clusters 
                             if idx not in [cluster1_idx, cluster2_idx]]
            active_clusters.append(next_cluster_id)
            next_cluster_id += 1
        
        self.linkage_matrix = np.array(self.linkage_matrix)
        return self
    
    def cut_tree(self, n_clusters):
        if len(self.linkage_matrix) == 0:
            raise Exception("Fit the model first")
        
        if n_clusters < 1 or n_clusters > self.n_samples:
            raise ValueError("Invalid number of clusters")
            
        labels = np.arange(self.n_samples)
        
        for i in range(self.n_samples - n_clusters):
            cluster1, cluster2 = int(self.linkage_matrix[i][0]), int(self.linkage_matrix[i][1])
            labels[labels == cluster2] = cluster1
        
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[label] for label in labels])
            
        self.labels_ = labels
        return self.labels_

# Plot elbow graph for Kmeans
def plot_elbow_method(k_values, errors):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, errors, marker='o', color='red', linestyle='-')
    plt.title("Elbow Method: SSE vs. Number of Clusters (k)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Sum of Squared Errors (SSE)")
    plt.grid(True)
    plt.show()

# Plot dendogram for Hierarchical Clustering
def plot_dendrogram(model, title, X):
    colors = sns.color_palette("tab10", 3)
    plt.figure(figsize=(10, 7))
    dendrogram(
        model.linkage_matrix,
        color_threshold=model.linkage_matrix[-2, 2],
        above_threshold_color=colors[0],
        labels=range(1, len(X) + 1)
    )
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

def main():
    # Load and preprocess data
    df = pd.read_csv('wine.csv')
    X = df.iloc[:, 1:].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means clustering
    print("\nRunning K-means clustering...")
    k_errors = []
    k_range = range(2, 9)
    
    for k in k_range:
        kmeans = KMeansClustering(n_clusters=k, max_iterations=100)
        kmeans.fit(X)
        error = kmeans.error_history[-1]
        k_errors.append(error)
        print(f"K = {k}, Final SSE = {error:.2f}")
    
    plot_elbow_method(list(k_range), k_errors)
    
    # Hierarchical clustering
    print("\nRunning Hierarchical clustering:")
    # Reduce dimensions for hierarchical clustering
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_scaled)
    
    # Single Link
    print("Performing Single Link clustering:")
    single_link = HierarchicalClustering(linkage_type='single')
    single_link.fit(X_reduced)
    single_labels = single_link.cut_tree(n_clusters=3)
    single_silhouette = silhouette_score(X_reduced, single_labels)
    plot_dendrogram(single_link, 'Single Link Hierarchical Clustering Dendrogram', X_reduced)
    print(f"Single Link Silhouette Score: {single_silhouette:.3f}")
    
    # Complete Link
    print("\nPerforming Complete Link clustering:")
    complete_link = HierarchicalClustering(linkage_type='complete')
    complete_link.fit(X_reduced)
    complete_labels = complete_link.cut_tree(n_clusters=3)
    complete_silhouette = silhouette_score(X_reduced, complete_labels)
    plot_dendrogram(complete_link, 'Complete Link Hierarchical Clustering Dendrogram', X_reduced)
    print(f"Complete Link Silhouette Score: {complete_silhouette:.3f}")

if __name__ == "__main__":
    main()
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def fit(self, X):
        # Inicializar centróides aleatórios
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        prev_centroids = np.zeros(self.centroids.shape)

        while np.linalg.norm(self.centroids - prev_centroids) > 1e-4:
            prev_centroids = self.centroids.copy()
            # Atribuir pontos aos clusters
            self.labels = self._assign_clusters(X)
            # Recalcular os centróides
            self.centroids = self._update_centroids(X)

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids

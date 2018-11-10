from typing import *

from sklearn.cluster import AgglomerativeClustering, k_means
from scipy.spatial.distance import cosine
from spherecluster import SphericalKMeans

from utility_functions import get_clus_config


# ----------------------------------------------------------------------
# type definitions
data_type = List[Iterator[float]] # a list of vectors

# ----------------------------------------------------------------------


class Clustering:
    """Clustering interface for different clustering methods."""

    def __init__(self):
        clus_configs = get_clus_config()
        self.clus_type = clus_configs['type']
        self.affinity = clus_configs['affinity']
        self.linkage = clus_configs['linkage']
        self.n_clusters = clus_configs['n_clusters']
        self.cluster = None
        self.compactness = None

    def cluster(self,
                data: List[Iterator[float]],
                find_n: False
                ) -> Dict[str, Union[List[int],  Union[float, None]]]:
        """Cluster the input data into n clusters.

        Args:
            data: A list of vectors.
            find_n: If True, don't use self.n_cluster but find n using
                elbow analysis instead
        Return:
            A list of integers as class labels. The order of the list
            corresponds to the order of the input data.
        """
        if not n:
            n = self._get_n()
        if self.clus_type == 'kmeans':
            self.cluster = k_means(n_clusters=self.n_clusters)
        elif self.clus_type == 'sphericalkmeans':
            self.cluster = SphericalKMeans(n_clusters=self.n_clusters)
        elif self.clus_type == 'agglomerative':
            self.cluster = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                affinity=self.affinity,
                linkage=self.linkage)

        self.cluster.fit(data)
        self._calc_compactness()

        return {'labels': cluster.labels_, 'compactness': self.compactness}

    def _get_n(self):
        raise NotImplementedError

    def _calc_compactness(self):
        if self.clus_type == 'kmeans' or self.clus_type == 'sphericalkmeans':
            return self.cluster.inertia
        else:
            return None
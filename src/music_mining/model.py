"""Clustering, dimensionality reduction, evaluation, and recommendation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .config import AUDIO_FEATURES


@dataclass
class EvaluationResult:
    """Model-selection metrics for a range of cluster counts."""

    scores: pd.DataFrame


class MusicDiscoveryModel:
    """Fit a reusable content-based music discovery model."""

    def __init__(self, n_clusters: int = 5, random_state: int = 42) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.clusterer = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        )
        self.pca = PCA(n_components=2, random_state=random_state)
        self.tracks: pd.DataFrame | None = None
        self.matrix: np.ndarray | None = None

    def fit(self, tracks: pd.DataFrame) -> "MusicDiscoveryModel":
        """Fit scaling, K-Means, and PCA on a track table."""
        missing = [column for column in AUDIO_FEATURES if column not in tracks]
        if missing:
            raise ValueError(f"Missing model features: {', '.join(missing)}")
        if len(tracks) < self.n_clusters:
            raise ValueError("The dataset must contain at least n_clusters tracks.")

        self.tracks = tracks.reset_index(drop=True).copy()
        self.matrix = self.scaler.fit_transform(self.tracks[AUDIO_FEATURES])
        self.tracks["cluster"] = self.clusterer.fit_predict(self.matrix)
        coordinates = self.pca.fit_transform(self.matrix)
        self.tracks["pca_1"] = coordinates[:, 0]
        self.tracks["pca_2"] = coordinates[:, 1]
        return self

    def evaluate(
        self, tracks: pd.DataFrame, cluster_counts: range = range(2, 11)
    ) -> EvaluationResult:
        """Calculate inertia and silhouette scores for candidate cluster counts."""
        matrix = StandardScaler().fit_transform(tracks[AUDIO_FEATURES])
        rows = []
        for count in cluster_counts:
            if count >= len(tracks):
                continue
            clusterer = KMeans(
                n_clusters=count, random_state=self.random_state, n_init=10
            )
            labels = clusterer.fit_predict(matrix)
            cluster_sizes = np.bincount(labels, minlength=count)
            rows.append(
                {
                    "n_clusters": count,
                    "inertia": clusterer.inertia_,
                    "silhouette_score": silhouette_score(matrix, labels),
                    "calinski_harabasz_score": calinski_harabasz_score(matrix, labels),
                    "davies_bouldin_score": davies_bouldin_score(matrix, labels),
                    "smallest_cluster_pct": cluster_sizes.min() / len(labels) * 100,
                }
            )
        if not rows:
            raise ValueError("No valid cluster counts for the supplied dataset.")
        return EvaluationResult(pd.DataFrame(rows))

    def recommend(self, query: str, n: int = 10) -> pd.DataFrame:
        """Return the most similar tracks by name, ID, or artist text."""
        if self.tracks is None or self.matrix is None:
            raise RuntimeError("Fit the model before requesting recommendations.")
        if n < 1:
            raise ValueError("n must be positive.")

        matches = self._find_query_matches(query)
        if matches.empty:
            raise ValueError(f"No track matched query: {query}")
        query_index = matches.index[0]
        similarities = cosine_similarity(
            self.matrix[query_index : query_index + 1], self.matrix
        )[0]
        similarities[query_index] = -1
        count = min(n, len(self.tracks) - 1)
        indexes = np.argsort(similarities)[::-1][:count]
        result = self.tracks.iloc[indexes].copy()
        result.insert(0, "similarity", similarities[indexes])
        return result

    def _find_query_matches(self, query: str) -> pd.DataFrame:
        assert self.tracks is not None
        text = self.tracks.fillna("").astype(str)
        query_lower = query.casefold()
        mask = pd.Series(False, index=text.index)
        for column in ("track_id", "track_name", "artists"):
            if column in text:
                mask |= text[column].str.casefold().str.contains(
                    query_lower, regex=False, na=False
                )
        return self.tracks.loc[mask]

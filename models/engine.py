# -*- coding: utf-8 -*-
"""
ClusteringModelEngine — Algoritmos de clustering y reducción de dimensionalidad.

Responsabilidades:
  - K-Means: método del codo, silhouette, ajuste final.
  - DBSCAN: ajuste de eps/min_samples, detección de outliers.
  - PCA: reducción lineal a 2D, varianza explicada, loadings.
  - t-SNE: reducción no lineal a 2D para visualización.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


class ClusteringModelEngine:
    """Motor de modelos de clustering y reducción de dimensionalidad."""

    def __init__(self, scaled_data: np.ndarray, feature_names: list[str]):
        self.X = scaled_data
        self.feature_names = feature_names

        # Resultados K-Means
        self.kmeans_model: Optional[KMeans] = None
        self.kmeans_labels: np.ndarray | None = None
        self.kmeans_inertias: list[float] = []
        self.kmeans_silhouettes: list[float] = []
        self.optimal_k: int = 0

        # Resultados DBSCAN
        self.dbscan_model: Optional[DBSCAN] = None
        self.dbscan_labels: np.ndarray | None = None
        self.dbscan_n_clusters: int = 0
        self.dbscan_n_noise: int = 0

        # Resultados PCA
        self.pca_model: Optional[PCA] = None
        self.pca_2d: np.ndarray | None = None
        self.pca_variance_ratio: np.ndarray | None = None
        self.pca_loadings: pd.DataFrame | None = None

        # Resultados t-SNE
        self.tsne_2d: np.ndarray | None = None

    # ==================================================================
    # K-MEANS
    # ==================================================================
    def run_elbow_analysis(self, k_range: range = range(1, 10)) -> tuple[list, list]:
        """Ejecuta K-Means para rango de k, calcula inercia y silhouette."""
        print("=" * 60)
        print("FASE 3 — K-MEANS: MÉTODO DEL CODO + SILHOUETTE")
        print("=" * 60)

        self.kmeans_inertias = []
        self.kmeans_silhouettes = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = km.fit_predict(self.X)
            self.kmeans_inertias.append(km.inertia_)

            if k >= 2:
                sil = silhouette_score(self.X, labels)
                self.kmeans_silhouettes.append(sil)
                print(f"  K={k:2d} | Inercia: {km.inertia_:10.2f} | Silhouette: {sil:.4f}")
            else:
                self.kmeans_silhouettes.append(0)
                print(f"  K={k:2d} | Inercia: {km.inertia_:10.2f} | Silhouette: N/A")

        # Determinar K óptimo por silhouette
        valid_sils = [(k, s) for k, s in zip(k_range, self.kmeans_silhouettes) if k >= 2]
        self.optimal_k = max(valid_sils, key=lambda x: x[1])[0]
        print(f"\n  [OK] K óptimo por Silhouette Score: {self.optimal_k}\n")

        return self.kmeans_inertias, self.kmeans_silhouettes

    def fit_kmeans(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """Ajusta K-Means final con el K óptimo (o especificado)."""
        k = n_clusters or self.optimal_k
        print(f"  Ajustando K-Means final con K={k}...")

        self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.kmeans_labels = self.kmeans_model.fit_predict(self.X)

        sil = silhouette_score(self.X, self.kmeans_labels)
        print(f"  [OK] Silhouette Score final: {sil:.4f}")
        print(f"  [OK] Distribución de clusters: {np.bincount(self.kmeans_labels).tolist()}\n")

        return self.kmeans_labels

    def get_centroids_original_scale(self, scaler) -> pd.DataFrame:
        """Devuelve centroides en escala original para interpretación."""
        centroids_scaled = self.kmeans_model.cluster_centers_
        centroids_original = scaler.inverse_transform(centroids_scaled)
        df = pd.DataFrame(centroids_original, columns=self.feature_names)
        df.index.name = "Cluster"
        return df.round(2)

    # ==================================================================
    # DBSCAN
    # ==================================================================
    def estimate_eps(self, min_samples: int = 5) -> float:
        """Estima eps óptimo usando el método k-distance graph."""
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(self.X)
        distances, _ = nn.kneighbors(self.X)
        k_distances = np.sort(distances[:, -1])

        # Buscar el punto de inflexión (codo) en la curva de distancias
        diffs = np.diff(k_distances)
        diffs2 = np.diff(diffs)
        knee_idx = np.argmax(diffs2) + 2 if len(diffs2) > 0 else len(k_distances) // 2
        estimated_eps = k_distances[min(knee_idx, len(k_distances) - 1)]

        return round(float(estimated_eps), 2)

    def fit_dbscan(self, eps: Optional[float] = None, min_samples: int = 5) -> np.ndarray:
        """Ajusta DBSCAN. Si eps es None, lo estima automáticamente."""
        print("=" * 60)
        print("FASE 4 — DBSCAN: CLUSTERING POR DENSIDAD")
        print("=" * 60)

        if eps is None:
            eps = self.estimate_eps(min_samples)
            print(f"\n  eps estimado automáticamente: {eps}")

        print(f"  Parámetros: eps={eps}, min_samples={min_samples}")

        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        self.dbscan_labels = self.dbscan_model.fit_predict(self.X)

        unique_labels = set(self.dbscan_labels)
        self.dbscan_n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        self.dbscan_n_noise = int(np.sum(self.dbscan_labels == -1))

        print(f"\n  Clusters encontrados:  {self.dbscan_n_clusters}")
        print(f"  Puntos de ruido:       {self.dbscan_n_noise}")

        if self.dbscan_n_clusters > 0:
            valid_mask = self.dbscan_labels != -1
            if np.sum(valid_mask) > 1 and len(set(self.dbscan_labels[valid_mask])) > 1:
                sil = silhouette_score(self.X[valid_mask], self.dbscan_labels[valid_mask])
                print(f"  Silhouette (sin ruido): {sil:.4f}")

        # Distribución
        labels_series = pd.Series(self.dbscan_labels)
        print(f"\n  Distribución:")
        for label, count in labels_series.value_counts().sort_index().items():
            tag = "RUIDO" if label == -1 else f"Cluster {label}"
            print(f"    {tag}: {count} registros ({count/len(self.dbscan_labels)*100:.1f}%)")

        print()
        return self.dbscan_labels

    # ==================================================================
    # PCA
    # ==================================================================
    def fit_pca(self, n_components: int = 2) -> np.ndarray:
        """Reducción de dimensionalidad lineal con PCA."""
        print("=" * 60)
        print("FASE 5 — PCA: REDUCCIÓN DE DIMENSIONALIDAD LINEAL")
        print("=" * 60)

        self.pca_model = PCA(n_components=n_components, random_state=42)
        self.pca_2d = self.pca_model.fit_transform(self.X)
        self.pca_variance_ratio = self.pca_model.explained_variance_ratio_

        print(f"\n  Varianza explicada por componente:")
        for i, var in enumerate(self.pca_variance_ratio):
            print(f"    PC{i+1}: {var:.4f} ({var*100:.1f}%)")
        print(f"    Total acumulado: {self.pca_variance_ratio.sum()*100:.1f}%")

        # Loadings (pesos de cada variable en cada componente)
        self.pca_loadings = pd.DataFrame(
            self.pca_model.components_,
            columns=self.feature_names,
            index=[f"PC{i+1}" for i in range(n_components)],
        ).round(4)
        print(f"\n  Loadings (pesos):")
        print(f"  {self.pca_loadings.to_string()}\n")

        return self.pca_2d

    # ==================================================================
    # t-SNE
    # ==================================================================
    def fit_tsne(self, perplexity: float = 30, learning_rate: float = 200) -> np.ndarray:
        """Reducción de dimensionalidad no lineal con t-SNE."""
        print("=" * 60)
        print("FASE 6 — t-SNE: REDUCCIÓN NO LINEAL")
        print("=" * 60)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=42,
            max_iter=1000,
        )
        self.tsne_2d = tsne.fit_transform(self.X)

        print(f"\n  Parámetros: perplexity={perplexity}, learning_rate={learning_rate}")
        print(f"  [OK] Proyección 2D generada: {self.tsne_2d.shape}\n")

        return self.tsne_2d

# -*- coding: utf-8 -*-
"""
ClusteringVisualizer — Generación y persistencia de visualizaciones.

Responsabilidades:
  - Gráficos exploratorios: correlación, distribución, pairplot.
  - Método del codo y silhouette.
  - Scatter plots de K-Means y DBSCAN.
  - Visualizaciones 2D de PCA y t-SNE coloreadas por cluster.
  - Heatmap de loadings PCA.
  - Tabla resumen de centroides.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo global
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = sns.color_palette("Set2", 10)
FIGSIZE_STANDARD = (10, 7)
FIGSIZE_WIDE = (14, 6)
DPI = 150


class ClusteringVisualizer:
    """Genera todas las visualizaciones del análisis de clustering."""

    def __init__(self, assets_dir: str = "assets"):
        self.assets_dir = assets_dir
        os.makedirs(assets_dir, exist_ok=True)

    def _save(self, fig, filename: str):
        path = os.path.join(self.assets_dir, filename)
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  [SAVED] {path}")

    # ------------------------------------------------------------------
    # EXPLORATORIOS
    # ------------------------------------------------------------------
    def plot_correlation_matrix(self, df: pd.DataFrame, numeric_cols: list[str]):
        """Heatmap de correlación de Pearson."""
        fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Matriz de Correlación de Pearson", fontsize=14, fontweight="bold", pad=15)
        self._save(fig, "correlacion_pearson.png")

    def plot_distributions(self, df: pd.DataFrame, numeric_cols: list[str]):
        """Histogramas + KDE para variables numéricas."""
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, ax=axes[i], color=PALETTE[i % len(PALETTE)],
                         edgecolor="white", alpha=0.7)
            axes[i].set_title(col, fontsize=11, fontweight="bold")
            axes[i].set_xlabel("")

        # Ocultar ejes sobrantes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Distribución de Variables Numéricas", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        self._save(fig, "distribuciones_variables.png")

    def plot_pairplot(self, df: pd.DataFrame, features: list[str]):
        """Pairplot exploratorio de las features seleccionadas."""
        g = sns.pairplot(df[features], diag_kind="kde", corner=True,
                         plot_kws={"alpha": 0.5, "s": 20, "edgecolor": "none"})
        g.figure.suptitle("Pairplot — Features Seleccionadas", fontsize=14, fontweight="bold", y=1.02)
        self._save(g.figure, "pairplot_features.png")

    # ------------------------------------------------------------------
    # K-MEANS
    # ------------------------------------------------------------------
    def plot_elbow_silhouette(self, k_range, inertias, silhouettes):
        """Gráfico dual: método del codo + silhouette score."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

        # Codo
        ax1.plot(list(k_range), inertias, "o-", color="#2196F3", linewidth=2, markersize=8)
        ax1.set_xlabel("Número de Clusters (K)", fontsize=11)
        ax1.set_ylabel("Inercia", fontsize=11)
        ax1.set_title("Método del Codo", fontsize=13, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Silhouette
        sil_values = [s for k, s in zip(k_range, silhouettes) if k >= 2]
        sil_ks = [k for k in k_range if k >= 2]
        ax2.plot(sil_ks, sil_values, "o-", color="#4CAF50", linewidth=2, markersize=8)
        ax2.set_xlabel("Número de Clusters (K)", fontsize=11)
        ax2.set_ylabel("Silhouette Score", fontsize=11)
        ax2.set_title("Silhouette Score por K", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Marcar óptimo
        best_k = sil_ks[np.argmax(sil_values)]
        best_sil = max(sil_values)
        ax2.axvline(x=best_k, color="red", linestyle="--", alpha=0.7, label=f"Óptimo: K={best_k}")
        ax2.scatter([best_k], [best_sil], color="red", s=150, zorder=5, edgecolors="black")
        ax2.legend(fontsize=10)

        fig.suptitle("Determinación del Número Óptimo de Clusters", fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "metodo_codo_silhouette.png")

    def plot_kmeans_clusters(self, data_2d, labels, title_suffix="", filename="segmentacion_kmeans.png"):
        """Scatter 2D de clusters K-Means."""
        fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
        unique_labels = sorted(set(labels))

        for label in unique_labels:
            mask = labels == label
            ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                       c=[PALETTE[label % len(PALETTE)]], label=f"Cluster {label}",
                       alpha=0.6, s=50, edgecolors="white", linewidth=0.5)

        ax.set_xlabel("Componente 1", fontsize=11)
        ax.set_ylabel("Componente 2", fontsize=11)
        ax.set_title(f"Segmentación K-Means {title_suffix}", fontsize=13, fontweight="bold")
        ax.legend(title="Cluster", fontsize=10, title_fontsize=11)
        ax.grid(True, alpha=0.3)
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # DBSCAN
    # ------------------------------------------------------------------
    def plot_dbscan_clusters(self, data_2d, labels):
        """Scatter 2D de clusters DBSCAN (incluyendo ruido)."""
        fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
        unique_labels = sorted(set(labels))

        for label in unique_labels:
            mask = labels == label
            if label == -1:
                ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                           c="gray", label="Ruido (outliers)", alpha=0.4, s=30, marker="x")
            else:
                ax.scatter(data_2d[mask, 0], data_2d[mask, 1],
                           c=[PALETTE[label % len(PALETTE)]], label=f"Cluster {label}",
                           alpha=0.6, s=50, edgecolors="white", linewidth=0.5)

        ax.set_xlabel("Componente 1", fontsize=11)
        ax.set_ylabel("Componente 2", fontsize=11)
        ax.set_title("Segmentación DBSCAN", fontsize=13, fontweight="bold")
        ax.legend(title="Cluster", fontsize=10, title_fontsize=11)
        ax.grid(True, alpha=0.3)
        self._save(fig, "segmentacion_dbscan.png")

    # ------------------------------------------------------------------
    # COMPARATIVO K-MEANS vs DBSCAN
    # ------------------------------------------------------------------
    def plot_comparison(self, data_2d, kmeans_labels, dbscan_labels):
        """Gráfico side-by-side comparando K-Means vs DBSCAN."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # K-Means
        for label in sorted(set(kmeans_labels)):
            mask = kmeans_labels == label
            ax1.scatter(data_2d[mask, 0], data_2d[mask, 1],
                        c=[PALETTE[label % len(PALETTE)]], label=f"Cluster {label}",
                        alpha=0.6, s=40, edgecolors="white", linewidth=0.3)
        ax1.set_title("K-Means", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # DBSCAN
        for label in sorted(set(dbscan_labels)):
            mask = dbscan_labels == label
            if label == -1:
                ax2.scatter(data_2d[mask, 0], data_2d[mask, 1],
                            c="gray", label="Ruido", alpha=0.4, s=30, marker="x")
            else:
                ax2.scatter(data_2d[mask, 0], data_2d[mask, 1],
                            c=[PALETTE[label % len(PALETTE)]], label=f"Cluster {label}",
                            alpha=0.6, s=40, edgecolors="white", linewidth=0.3)
        ax2.set_title("DBSCAN", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Comparación: K-Means vs DBSCAN (proyección PCA 2D)",
                      fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "comparacion_kmeans_vs_dbscan.png")

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------
    def plot_pca_variance(self, variance_ratio):
        """Barras de varianza explicada por componente."""
        fig, ax = plt.subplots(figsize=(8, 5))
        components = [f"PC{i+1}" for i in range(len(variance_ratio))]
        cumulative = np.cumsum(variance_ratio)

        bars = ax.bar(components, variance_ratio * 100, color="#2196F3", alpha=0.7,
                       edgecolor="white", label="Individual")
        ax.plot(components, cumulative * 100, "o-", color="#FF5722", linewidth=2,
                markersize=8, label="Acumulada")

        for bar, val in zip(bars, variance_ratio * 100):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

        ax.set_ylabel("Varianza Explicada (%)", fontsize=11)
        ax.set_title("Varianza Explicada por Componente Principal", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis="y")
        self._save(fig, "varianza_explicada_pca.png")

    def plot_pca_loadings_heatmap(self, loadings: pd.DataFrame):
        """Heatmap de pesos (loadings) de cada variable en cada componente."""
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(loadings, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Heatmap de Componentes PCA (Loadings)", fontsize=13, fontweight="bold", pad=15)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        self._save(fig, "heatmap_componentes_pca.png")

    # ------------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------------
    def plot_tsne(self, tsne_2d, labels, title="Visualización t-SNE coloreada por Cluster (K-Means)"):
        """Scatter t-SNE coloreado por labels de cluster."""
        fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

        for label in sorted(set(labels)):
            mask = labels == label
            ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
                       c=[PALETTE[label % len(PALETTE)]], label=f"Cluster {label}",
                       alpha=0.6, s=50, edgecolors="white", linewidth=0.5)

        ax.set_xlabel("t-SNE Dim 1", fontsize=11)
        ax.set_ylabel("t-SNE Dim 2", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(title="Cluster", fontsize=10, title_fontsize=11)
        ax.grid(True, alpha=0.3)
        self._save(fig, "visualizacion_tsne_kmeans.png")

    # ------------------------------------------------------------------
    # TABLA RESUMEN
    # ------------------------------------------------------------------
    def plot_centroids_table(self, centroids_df: pd.DataFrame):
        """Genera una imagen con la tabla de centroides en escala original."""
        fig, ax = plt.subplots(figsize=(12, 2 + 0.5 * len(centroids_df)))
        ax.axis("off")

        table = ax.table(
            cellText=centroids_df.values,
            colLabels=centroids_df.columns,
            rowLabels=[f"Cluster {i}" for i in centroids_df.index],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Estilizar header
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2196F3")
                cell.set_text_props(color="white", fontweight="bold")
            elif col == -1:
                cell.set_facecolor("#E3F2FD")
                cell.set_text_props(fontweight="bold")

        ax.set_title("Características Medias por Cluster (Escala Original)",
                      fontsize=13, fontweight="bold", pad=20)
        self._save(fig, "tabla_centroides.png")

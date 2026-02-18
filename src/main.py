# -*- coding: utf-8 -*-
"""
main.py — Orquestador Principal: Segmentación de Clientes E-commerce

Ejecuta las 6 fases del pipeline de aprendizaje no supervisado:
  1. Carga y validación del dataset
  2. Preprocesamiento y selección de features
  3. K-Means (codo + silhouette + ajuste)
  4. DBSCAN (clustering por densidad)
  5. PCA + t-SNE (reducción de dimensionalidad)
  6. Visualizaciones y tabla resumen

Autor: Jefferson Velasquez, Frank Macias, Jorge Murillo
Fecha: Febrero 2026
Asignatura: Machine Learning — Maestría en Inteligencia Artificial
"""

import os
import sys
import numpy as np

# Asegurar que los imports funcionen desde la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.processor import EcommerceDataProcessor
from models.engine import ClusteringModelEngine
from utils.visualizer import ClusteringVisualizer


def main():
    # ==================================================================
    # CONFIGURACIÓN
    # ==================================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "kaggle", "ecommerce_customer_behavior.csv")
    ASSETS_DIR = os.path.join(BASE_DIR, "..", "assets")

    FEATURES = [
        "Age",
        "Total Spend",
        "Items Purchased",
        "Average Rating",
        "Days Since Last Purchase",
        "Membership_encoded",
    ]

    K_RANGE = range(1, 10)

    # ==================================================================
    # FASE 1-2: CARGA, VALIDACIÓN Y PREPROCESAMIENTO
    # ==================================================================
    processor = EcommerceDataProcessor(DATA_PATH)
    processor.load_and_validate()
    scaled_data = processor.preprocess(features=FEATURES)

    # ==================================================================
    # INICIALIZAR MOTOR Y VISUALIZADOR
    # ==================================================================
    engine = ClusteringModelEngine(scaled_data, FEATURES)
    viz = ClusteringVisualizer(assets_dir=ASSETS_DIR)

    # ==================================================================
    # FASE 2.5: ANÁLISIS EXPLORATORIO — VISUALIZACIONES
    # ==================================================================
    print("=" * 60)
    print("VISUALIZACIONES EXPLORATORIAS")
    print("=" * 60)

    numeric_cols = ["Age", "Total Spend", "Items Purchased", "Average Rating",
                    "Days Since Last Purchase"]

    viz.plot_correlation_matrix(processor.processed_dataframe, numeric_cols)
    viz.plot_distributions(processor.processed_dataframe, numeric_cols)
    viz.plot_pairplot(processor.processed_dataframe, FEATURES)

    # ==================================================================
    # FASE 3: K-MEANS
    # ==================================================================
    inertias, silhouettes = engine.run_elbow_analysis(k_range=K_RANGE)
    viz.plot_elbow_silhouette(K_RANGE, inertias, silhouettes)

    # Nota: Silhouette sugiere K=2 como máximo global, pero el codo muestra
    # inflexión clara en K=3 con ganancia marginal mínima después.
    # K=3 ofrece mayor granularidad de negocio (3 perfiles accionables)
    # con un silhouette aceptable (>0.25), por lo que se selecciona K=3.
    kmeans_labels = engine.fit_kmeans(n_clusters=3)

    # Centroides en escala original
    centroids_df = engine.get_centroids_original_scale(processor.scaler)
    print("\n  Centroides en escala original:")
    print(f"  {centroids_df.to_string()}\n")
    viz.plot_centroids_table(centroids_df)

    # ==================================================================
    # FASE 4: DBSCAN
    # ==================================================================
    dbscan_labels = engine.fit_dbscan(min_samples=5)

    # ==================================================================
    # FASE 5: PCA + t-SNE
    # ==================================================================
    pca_2d = engine.fit_pca(n_components=2)
    tsne_2d = engine.fit_tsne(perplexity=30, learning_rate=200)

    # ==================================================================
    # FASE 6: VISUALIZACIONES DE RESULTADOS
    # ==================================================================
    print("=" * 60)
    print("FASE 7 — VISUALIZACIONES FINALES")
    print("=" * 60)

    # K-Means sobre PCA 2D
    viz.plot_kmeans_clusters(pca_2d, kmeans_labels,
                             title_suffix="(PCA 2D)", filename="segmentacion_kmeans_pca.png")

    # DBSCAN sobre PCA 2D
    viz.plot_dbscan_clusters(pca_2d, dbscan_labels)

    # Comparativo side-by-side
    viz.plot_comparison(pca_2d, kmeans_labels, dbscan_labels)

    # PCA: varianza explicada + heatmap de loadings
    viz.plot_pca_variance(engine.pca_variance_ratio)
    viz.plot_pca_loadings_heatmap(engine.pca_loadings)

    # t-SNE coloreado por K-Means
    viz.plot_tsne(tsne_2d, kmeans_labels)

    # ==================================================================
    # RESUMEN FINAL
    # ==================================================================
    print("\n" + "=" * 60)
    print("RESUMEN DE EJECUCIÓN")
    print("=" * 60)
    print(f"  Dataset:              {processor.df_raw.shape[0]} registros, {processor.df_raw.shape[1]} columnas")
    print(f"  Features utilizadas:  {len(FEATURES)} → {FEATURES}")
    print(f"  K óptimo (K-Means):   {engine.optimal_k}")
    print(f"  DBSCAN clusters:      {engine.dbscan_n_clusters} cluster(s) + {engine.dbscan_n_noise} outlier(s)")
    print(f"  PCA varianza 2D:      {engine.pca_variance_ratio.sum()*100:.1f}%")
    print(f"  Visualizaciones:      {ASSETS_DIR}/")
    print("=" * 60)
    print("\n  [DONE] Pipeline completado exitosamente.\n")


if __name__ == "__main__":
    main()

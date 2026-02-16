# -*- coding: utf-8 -*-
"""
EcommerceDataProcessor — Carga, validación y preprocesamiento del dataset.

Responsabilidades:
  - Cargar CSV y validar integridad (nulos, duplicados).
  - Codificar variables categóricas para clustering.
  - Seleccionar y escalar features numéricas.
  - Exponer datos crudos y escalados para los modelos.
"""

import os
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class EcommerceDataProcessor:
    """Pipeline de datos para segmentación de clientes e-commerce."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df_raw: pd.DataFrame | None = None
        self.df_processed: pd.DataFrame | None = None
        self.feature_names: list[str] = []
        self.scaler = StandardScaler()
        self._scaled_data: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 1. Carga y validación
    # ------------------------------------------------------------------
    def load_and_validate(self) -> pd.DataFrame:
        """Carga el CSV y ejecuta validaciones básicas de calidad."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset no encontrado: {self.csv_path}")

        self.df_raw = pd.read_csv(self.csv_path)
        print("=" * 60)
        print("FASE 1 — CARGA Y VALIDACIÓN DEL DATASET")
        print("=" * 60)
        print(f"\nRegistros: {self.df_raw.shape[0]} | Columnas: {self.df_raw.shape[1]}")
        print(f"\nPrimeras 5 filas:")
        print(self.df_raw.head().to_string())
        print(f"\nTipos de datos:")
        print(self.df_raw.dtypes.to_string())
        print(f"\nEstadísticas descriptivas:")
        print(self.df_raw.describe().round(2).to_string())

        # Validaciones
        nulls = self.df_raw.isnull().sum()
        total_nulls = nulls.sum()
        duplicates = self.df_raw.duplicated(subset=["Customer ID"]).sum()

        print(f"\n--- Validación de calidad ---")
        print(f"  Valores nulos totales:    {total_nulls}")
        print(f"  Registros duplicados:     {duplicates}")

        if total_nulls > 0:
            print(f"  [WARN] Columnas con nulos: {nulls[nulls > 0].to_dict()}")
            self.df_raw.dropna(inplace=True)
            print(f"  [INFO] Registros tras eliminar nulos: {len(self.df_raw)}")

        if duplicates > 0:
            self.df_raw.drop_duplicates(subset=["Customer ID"], inplace=True)
            print(f"  [INFO] Registros tras eliminar duplicados: {len(self.df_raw)}")

        print(f"  [OK] Dataset limpio: {self.df_raw.shape[0]} registros válidos.\n")
        return self.df_raw

    # ------------------------------------------------------------------
    # 2. Preprocesamiento: encoding + selección de features
    # ------------------------------------------------------------------
    def preprocess(self, features: Optional[list[str]] = None) -> np.ndarray:
        """
        Codifica categóricas, selecciona features y escala con StandardScaler.

        Args:
            features: Lista de columnas a usar para clustering.
                      Si es None, usa un conjunto por defecto.
        Returns:
            np.ndarray escalado listo para modelos.
        """
        if self.df_raw is None:
            raise RuntimeError("Ejecutar load_and_validate() primero.")

        print("=" * 60)
        print("FASE 2 — PREPROCESAMIENTO Y SELECCIÓN DE FEATURES")
        print("=" * 60)

        self.df_processed = self.df_raw.copy()

        # --- Encoding de variables categóricas ---
        # Gender: binario
        self.df_processed["Gender_encoded"] = LabelEncoder().fit_transform(
            self.df_processed["Gender"]
        )
        # Membership Type: ordinal (Bronze < Silver < Gold)
        membership_map = {"Bronze": 0, "Silver": 1, "Gold": 2}
        self.df_processed["Membership_encoded"] = self.df_processed[
            "Membership Type"
        ].map(membership_map)

        # Discount Applied: binario
        self.df_processed["Discount_encoded"] = self.df_processed[
            "Discount Applied"
        ].astype(int)

        # Satisfaction Level: ordinal (Unsatisfied < Neutral < Satisfied)
        satisfaction_map = {"Unsatisfied": 0, "Neutral": 1, "Satisfied": 2}
        self.df_processed["Satisfaction_encoded"] = self.df_processed[
            "Satisfaction Level"
        ].map(satisfaction_map)

        # --- Selección de features ---
        if features is None:
            features = [
                "Age",
                "Total Spend",
                "Items Purchased",
                "Average Rating",
                "Days Since Last Purchase",
                "Membership_encoded",
            ]

        self.feature_names = features
        print(f"\nFeatures seleccionadas ({len(features)}):")
        for f in features:
            col = self.df_processed[f]
            print(f"  - {f:30s} | rango: [{col.min():.1f}, {col.max():.1f}] | media: {col.mean():.2f}")

        # --- Estandarización ---
        self._scaled_data = self.scaler.fit_transform(
            self.df_processed[features].values
        )
        print(f"\n  [OK] StandardScaler aplicado (media=0, std=1).")
        print(f"  [OK] Shape final: {self._scaled_data.shape}\n")
        return self._scaled_data

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def scaled_data(self) -> np.ndarray:
        if self._scaled_data is None:
            raise RuntimeError("Ejecutar preprocess() primero.")
        return self._scaled_data

    @property
    def raw_dataframe(self) -> pd.DataFrame:
        if self.df_raw is None:
            raise RuntimeError("Ejecutar load_and_validate() primero.")
        return self.df_raw

    @property
    def processed_dataframe(self) -> pd.DataFrame:
        if self.df_processed is None:
            raise RuntimeError("Ejecutar preprocess() primero.")
        return self.df_processed

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """Revierte la estandarización para interpretar centroides."""
        return pd.DataFrame(
            self.scaler.inverse_transform(data),
            columns=self.feature_names,
        )

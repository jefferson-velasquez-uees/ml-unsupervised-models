# -*- coding: utf-8 -*-
"""
main.py — Orquestador Principal: Segmentación de Clientes E-commerce

Autor: Jefferson Velasquez, Frank Macias, Jorge Delgado
Fecha: Febrero 2026
Asignatura: Machine Learning — Maestría en Inteligencia Artificial
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.processor import EcommerceDataProcessor

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

    # ==================================================================
    # CARGA, VALIDACIÓN Y PREPROCESAMIENTO
    # ==================================================================
    processor = EcommerceDataProcessor(DATA_PATH)
    processor.load_and_validate()

    print("\n" + "=" * 60)
    print(f"  Dataset:              {processor.df_raw.shape[0]} registros, {processor.df_raw.shape[1]} columnas")
    print(f"  Features utilizadas:  {len(FEATURES)} → {FEATURES}")



if __name__ == "__main__":
    main()

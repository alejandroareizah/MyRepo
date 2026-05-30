import pandas as pd
import numpy as np

def resumen_ventas_por_region(df):

    df = df.copy()

    df["ingreso_neto"] = (
        df["cantidad"] *
        df["precio_unitario"] *
        (1 - df["descuento"])
    )

    summary = df.groupby("region").agg(
        total_ingresos=("ingreso_neto", "sum"),
        promedio_descuento=("descuento", "mean"),
        num_transacciones=("region", "count")
    ).reset_index()

    total_general = summary["total_ingresos"].sum()

    summary["porcentaje_ingresos"] = np.round(
        summary["total_ingresos"] / total_general * 100,
        2
    )

    return summary.sort_values(
        "total_ingresos",
        ascending=False
    ).reset_index(drop=True)
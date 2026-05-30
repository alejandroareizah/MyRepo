import pandas as pd
import numpy as np

def segmentar_pacientes(df):

    df_clean = df.drop_duplicates().dropna().copy()

    conditions = [
        (df_clean["glucosa"] >= 140) |
        (df_clean["presion_arterial"] >= 140),

        (df_clean["glucosa"] >= 100) |
        (df_clean["presion_arterial"] >= 120)
    ]

    choices = ["alto", "medio"]

    df_clean["grupo_riesgo"] = np.select(
        conditions,
        choices,
        default="bajo"
    )

    return (
        df_clean
        .sort_values("edad")
        .reset_index(drop=True)
    )
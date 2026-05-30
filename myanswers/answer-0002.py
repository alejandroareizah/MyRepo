import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def resumir_ventas_y_tendencia(df):

    temp = df.copy()

    temp["fecha"] = pd.to_datetime(temp["fecha"])
    temp["mes"] = temp["fecha"].dt.to_period("M").astype(str)

    resumen = (
        temp.groupby("mes", as_index=False)["ventas"]
        .sum()
        .reset_index(drop=True)
    )

    X = np.arange(len(resumen)).reshape(-1, 1)
    y = resumen["ventas"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    pendiente = float(model.coef_[0])

    return resumen, pendiente
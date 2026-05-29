from sklearn.ensemble import IsolationForest

def detectar_fugas_energia(X, contaminacion):
    modelo = IsolationForest(
        contamination=contaminacion,
        random_state=42
    )

    return modelo.fit_predict(X)
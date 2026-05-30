from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier

def entrenar_clasificador_paneles(X, y):

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )

    clf.fit(X_scaled, y)

    predicciones = clf.predict(X_scaled)

    return clf, predicciones
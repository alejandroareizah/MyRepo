from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def entrenar_clasificador_paneles(X, y):
    
    # 1. Escalar datos
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Crear y entrenar modelo
    modelo = RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )
    
    modelo.fit(X_scaled, y)
    
    # 3. Predicciones
    predicciones = modelo.predict(X_scaled)
    
    # 4. Retornar modelo y predicciones
    return modelo, predicciones
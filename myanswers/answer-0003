from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import numpy as np

def detectar_anomalias_red(X, y):

    modelo = DecisionTreeClassifier(random_state=42)

    modelo.fit(X, y)

    predicciones = modelo.predict(X)

    balanced_acc = balanced_accuracy_score(y, predicciones)

    matriz = confusion_matrix(y, predicciones)

    return float(balanced_acc), matriz
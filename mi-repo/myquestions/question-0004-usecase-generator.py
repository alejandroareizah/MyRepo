import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
import random


def generar_caso_de_uso_clasificacion_multietiqueta():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función clasificacion_multietiqueta(X_train, y_train,
                                                X_test, y_test).

    Retorna
    -------
    input_data : dict — {
        'X_train': np.ndarray,
        'y_train': np.ndarray,
        'X_test' : np.ndarray,
        'y_test' : np.ndarray,
    }
    output_data: dict — {
        'f1_por_etiqueta': list[float],
        'f1_promedio'    : float,
        'n_etiquetas'    : int,
    }
    """
    n_samples  = random.randint(200, 500)
    n_features = random.randint(8, 20)
    n_labels   = random.randint(3, 6)
    test_size  = random.choice([0.2, 0.25, 0.3])

    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_labels=n_labels,
        n_classes=n_labels,
        allow_unlabeled=False,
        random_state=random.randint(0, 999),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Ground truth
    modelo = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    f1_etiquetas = f1_score(y_test, y_pred, average=None, zero_division=0)
    output = {
        'f1_por_etiqueta': [float(v) for v in f1_etiquetas],
        'f1_promedio':     float(np.mean(f1_etiquetas)),
        'n_etiquetas':     int(n_labels),
    }

    input_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test':  X_test,
        'y_test':  y_test,
    }
    return input_data, output


if __name__ == '__main__':
    inp, out = generar_caso_de_uso_clasificacion_multietiqueta()
    print("=== INPUT ===")
    print("X_train shape:", inp['X_train'].shape)
    print("y_train shape:", inp['y_train'].shape)
    print("X_test  shape:", inp['X_test'].shape)
    print("y_test  shape:", inp['y_test'].shape)
    print("\n=== OUTPUT ESPERADO ===")
    print("n_etiquetas   :", out['n_etiquetas'])
    print("f1_por_etiqueta:", [f"{v:.4f}" for v in out['f1_por_etiqueta']])
    print("f1_promedio   :", round(out['f1_promedio'], 4))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc
import random


def generar_caso_de_uso_curva_precision_recall():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función curva_precision_recall(y_true, y_proba).

    Retorna
    -------
    input_data : dict — {'y_true': np.ndarray, 'y_proba': np.ndarray}
    output_data: dict — {
        'mejor_umbral'       : float,
        'mejor_f1'           : float,
        'precision_en_umbral': float,
        'recall_en_umbral'   : float,
        'auc_pr'             : float,
    }
    """
    n_samples    = random.randint(150, 400)
    weights      = random.choice([[0.85, 0.15], [0.80, 0.20], [0.75, 0.25]])
    n_features   = random.randint(4, 10)
    random_state = random.randint(0, 999)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 2),
        n_redundant=min(2, n_features // 3),
        weights=weights,
        random_state=random_state,
    )

    modelo = random.choice([
        LogisticRegression(max_iter=300, random_state=42),
        RandomForestClassifier(n_estimators=50, random_state=42),
    ])
    modelo.fit(X, y)
    y_proba = modelo.predict_proba(X)[:, 1]

    # Ground truth
    precision_arr, recall_arr, thresholds = precision_recall_curve(y, y_proba)

    with np.errstate(invalid='ignore'):
        f1_arr = np.where(
            (precision_arr[:-1] + recall_arr[:-1]) == 0,
            0.0,
            2 * precision_arr[:-1] * recall_arr[:-1]
            / (precision_arr[:-1] + recall_arr[:-1])
        )

    best_idx = int(np.argmax(f1_arr))
    output = {
        'mejor_umbral':         float(thresholds[best_idx]),
        'mejor_f1':             float(f1_arr[best_idx]),
        'precision_en_umbral':  float(precision_arr[best_idx]),
        'recall_en_umbral':     float(recall_arr[best_idx]),
        'auc_pr':               float(auc(recall_arr, precision_arr)),
    }

    input_data = {'y_true': y, 'y_proba': y_proba}
    return input_data, output


if __name__ == '__main__':
    inp, out = generar_caso_de_uso_curva_precision_recall()
    print("=== INPUT ===")
    print("Positivos en y_true:", inp['y_true'].sum())
    print("Shape y_proba      :", inp['y_proba'].shape)
    print("\n=== OUTPUT ESPERADO ===")
    for k, v in out.items():
        print(f"  {k}: {v:.4f}")

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import random


def generar_caso_de_uso_evaluar_acuerdo_anotadores():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función evaluar_acuerdo_anotadores(df, col_anotador_1,
                                               col_anotador_2,
                                               col_referencia).

    Retorna
    -------
    input_data : dict — {
        'df'            : pd.DataFrame,
        'col_anotador_1': str,
        'col_anotador_2': str,
        'col_referencia': str,
    }
    output_data: dict — {
        'kappa_entre_anotadores' : float,
        'kappa_anotador1_vs_ref' : float,
        'kappa_anotador2_vs_ref' : float,
        'mejor_anotador'         : str,
    }
    """
    dominios = [
        {
            'clases':         ['positivo', 'negativo', 'neutral'],
            'col_anotador_1': 'anotador_medico_1',
            'col_anotador_2': 'anotador_medico_2',
            'col_referencia': 'diagnostico_oficial',
        },
        {
            'clases':         ['spam', 'no_spam'],
            'col_anotador_1': 'revisor_A',
            'col_anotador_2': 'revisor_B',
            'col_referencia': 'etiqueta_gold',
        },
        {
            'clases':         ['bajo', 'medio', 'alto', 'critico'],
            'col_anotador_1': 'experto_1',
            'col_anotador_2': 'experto_2',
            'col_referencia': 'referencia',
        },
        {
            'clases':         ['gato', 'perro', 'pajaro', 'otro'],
            'col_anotador_1': 'clasificador_humano_1',
            'col_anotador_2': 'clasificador_humano_2',
            'col_referencia': 'clase_real',
        },
    ]

    dominio = random.choice(dominios)
    clases  = dominio['clases']
    n_rows  = random.randint(50, 150)

    col_a1  = dominio['col_anotador_1']
    col_a2  = dominio['col_anotador_2']
    col_ref = dominio['col_referencia']

    # Referencia oficial (ground truth)
    referencia = np.random.choice(clases, size=n_rows)

    # Anotador 1: mayormente de acuerdo con referencia, algo de ruido
    ruido_a1 = random.uniform(0.05, 0.30)
    anotador_1 = np.where(
        np.random.rand(n_rows) > ruido_a1,
        referencia,
        np.random.choice(clases, size=n_rows)
    )

    # Anotador 2: diferente nivel de ruido
    ruido_a2 = random.uniform(0.05, 0.40)
    anotador_2 = np.where(
        np.random.rand(n_rows) > ruido_a2,
        referencia,
        np.random.choice(clases, size=n_rows)
    )

    df = pd.DataFrame({
        col_a1:  anotador_1,
        col_a2:  anotador_2,
        col_ref: referencia,
    })

    # Ground truth
    k_entre    = float(cohen_kappa_score(df[col_a1], df[col_a2]))
    k_a1_ref   = float(cohen_kappa_score(df[col_a1], df[col_ref]))
    k_a2_ref   = float(cohen_kappa_score(df[col_a2], df[col_ref]))

    if k_a1_ref >= k_a2_ref:
        mejor = 'anotador_1'
    else:
        mejor = 'anotador_2'

    output = {
        'kappa_entre_anotadores':  k_entre,
        'kappa_anotador1_vs_ref':  k_a1_ref,
        'kappa_anotador2_vs_ref':  k_a2_ref,
        'mejor_anotador':          mejor,
    }

    input_data = {
        'df':             df.copy(),
        'col_anotador_1': col_a1,
        'col_anotador_2': col_a2,
        'col_referencia': col_ref,
    }
    return input_data, output


if __name__ == '__main__':
    inp, out = generar_caso_de_uso_evaluar_acuerdo_anotadores()
    print("=== INPUT ===")
    print("col_anotador_1:", inp['col_anotador_1'])
    print("col_anotador_2:", inp['col_anotador_2'])
    print("col_referencia:", inp['col_referencia'])
    print(inp['df'].head())
    print("\n=== OUTPUT ESPERADO ===")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

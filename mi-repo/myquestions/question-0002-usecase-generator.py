import pandas as pd
import numpy as np
import random


def generar_caso_de_uso_resumir_frecuencias():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función resumir_frecuencias(df, grupo_col, evento_col).

    Retorna
    -------
    input_data : dict       — {'df': pd.DataFrame,
                                'grupo_col': str,
                                'evento_col': str}
    output_data: pd.DataFrame — resumen de frecuencias por grupo.
    """
    dominios = [
        {
            'grupos':     ['usuario_1', 'usuario_2', 'usuario_3',
                           'usuario_4', 'usuario_5'],
            'eventos':    ['click', 'scroll', 'compra', 'abandono', 'login'],
            'grupo_col':  'usuario_id',
            'evento_col': 'accion',
        },
        {
            'grupos':     ['sensor_A', 'sensor_B', 'sensor_C', 'sensor_D'],
            'eventos':    ['alerta', 'normal', 'critico', 'mantenimiento'],
            'grupo_col':  'sensor_id',
            'evento_col': 'estado',
        },
        {
            'grupos':     ['tienda_norte', 'tienda_sur',
                           'tienda_centro', 'tienda_este'],
            'eventos':    ['devolucion', 'venta', 'cambio', 'consulta'],
            'grupo_col':  'tienda',
            'evento_col': 'tipo_transaccion',
        },
    ]

    dominio    = random.choice(dominios)
    n_rows     = random.randint(40, 100)
    grupo_col  = dominio['grupo_col']
    evento_col = dominio['evento_col']

    grupos  = random.choices(dominio['grupos'],  k=n_rows)
    eventos = random.choices(dominio['eventos'], k=n_rows)

    df = pd.DataFrame({grupo_col: grupos, evento_col: eventos})

    # Ground truth
    def agg_group(g):
        vc = g[evento_col].value_counts()
        max_count  = vc.max()
        dominante  = sorted(vc[vc == max_count].index)[0]
        return pd.Series({
            'total_eventos':    len(g),
            'tipos_distintos':  g[evento_col].nunique(),
            'evento_dominante': dominante,
            'pct_dominante':    round(max_count / len(g) * 100, 2),
        })

    output = (
        df.groupby(grupo_col)
          .apply(agg_group)
          .reset_index()
          .sort_values('total_eventos', ascending=False)
          .reset_index(drop=True)
    )
    output['total_eventos']   = output['total_eventos'].astype(int)
    output['tipos_distintos'] = output['tipos_distintos'].astype(int)

    input_data = {'df': df.copy(), 'grupo_col': grupo_col, 'evento_col': evento_col}
    return input_data, output


if __name__ == '__main__':
    inp, out = generar_caso_de_uso_resumir_frecuencias()
    print("=== INPUT ===")
    print("grupo_col :", inp['grupo_col'])
    print("evento_col:", inp['evento_col'])
    print(inp['df'].head())
    print("\n=== OUTPUT ESPERADO ===")
    print(out)

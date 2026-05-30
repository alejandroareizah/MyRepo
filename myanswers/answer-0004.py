from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score

def comparar_regresores(X, y, n_folds):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr_scores = cross_val_score(
        LinearRegression(),
        X_scaled,
        y,
        cv=n_folds,
        scoring="r2"
    )

    ridge_scores = cross_val_score(
        Ridge(alpha=1.0),
        X_scaled,
        y,
        cv=n_folds,
        scoring="r2"
    )

    lr_mean = lr_scores.mean()
    ridge_mean = ridge_scores.mean()

    return {
        "linear_mean_r2": lr_mean,
        "linear_std_r2": lr_scores.std(),
        "ridge_mean_r2": ridge_mean,
        "ridge_std_r2": ridge_scores.std(),
        "mejor_modelo": (
            "Ridge" if ridge_mean > lr_mean
            else "LinearRegression"
        )
    }
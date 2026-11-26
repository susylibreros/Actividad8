import os
import urllib.request
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

os.makedirs("datos", exist_ok=True)

def preparar_y_entrenar():
    print("Descargando dataset público UCI Default of Credit Card Clients...")
    url_uci = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    ruta_xls = "datos/default_credit_card.xls"
    
    if not os.path.exists(ruta_xls):
        urllib.request.urlretrieve(url_uci, ruta_xls)

    # Leer archivo Excel oficial de UCI (la fila 1 contiene los encabezados reales)
    df = pd.read_excel(ruta_xls, header=1)

    y = df["default payment next month"].astype(int)
    cols_clave = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "BILL_AMT1", "BILL_AMT2", "PAY_AMT1", "PAY_AMT2"]
    X = df[cols_clave].astype(float)

    X_train, X_prod_pool, y_train, y_prod_pool = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=cols_clave)
    X_prod_scaled = pd.DataFrame(scaler.transform(X_prod_pool), columns=cols_clave)

    # Guardar particiones de referencia y flujo de producción
    train_df = X_train_scaled.copy()
    train_df["target"] = y_train.values
    train_df.to_csv("datos/referencia_train.csv", index=False)

    prod_df = X_prod_scaled.copy()
    prod_df["target"] = y_prod_pool.values
    prod_df.to_csv("datos/flujo_produccion.csv", index=False)

    mlflow.set_experiment("Scoring_Crediticio_Observabilidad")
    with mlflow.start_run(run_name="Baseline_Produccion_v1"):
        modelo = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42, n_jobs=1)
        modelo.fit(X_train_scaled, y_train)

        preds = modelo.predict(X_prod_scaled.iloc[:3000])
        probs = modelo.predict_proba(X_prod_scaled.iloc[:3000])[:, 1]
        f1_base = f1_score(y_prod_pool.iloc[:3000], preds)
        auc_base = roc_auc_score(y_prod_pool.iloc[:3000], probs)

        mlflow.log_params({"n_estimators": 80, "max_depth": 8, "dataset": "UCI_Credit_Default_30k"})
        mlflow.log_metrics({"f1_baseline": f1_base, "roc_auc_baseline": auc_base})
        mlflow.sklearn.log_model(
    modelo,
    artifact_path="modelo_crediticio",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
        print(f"Modelo Baseline registrado en MLflow | F1: {f1_base:.4f} | ROC-AUC: {auc_base:.4f}")

if __name__ == "__main__":
    preparar_y_entrenar()
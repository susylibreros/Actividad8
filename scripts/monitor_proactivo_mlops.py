cat << 'EOF' > scripts/monitor_proactivo_mlops.py
import os
import json
import uuid
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

try:
    import mlflow
except ImportError:
    class DummyMLflow:
        def set_tracking_uri(self, uri): pass
        def set_experiment(self, exp): pass
        def start_run(self, run_name=None):
            class _Ctx:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return _Ctx()
        def log_param(self, k, v): pass
        def log_metric(self, k, v, step=None): pass
    mlflow = DummyMLflow()

os.makedirs("evidencias", exist_ok=True)

# 1. Formateador de Logs Estructurados JSON con Trace ID y Span ID
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": "churn-prediction-service",
            "trace_id": getattr(record, "trace_id", "N/A"),
            "span_id": getattr(record, "span_id", "N/A"),
            "model_version": getattr(record, "model_version", "v1.2-prod"),
            "event_type": getattr(record, "event_type", "general_telemetry"),
            "message": record.getMessage()
        }
        return json.dumps(log_entry, ensure_ascii=False)

logger = logging.getLogger("MLOpsMonitor")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler("evidencias/structured_telemetry.log", mode="w", encoding="utf-8")
fh.setFormatter(JSONFormatter())
logger.addHandler(fh)

# 2. Función matemática para calcular Data Drift (PSI)
def calcular_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] -= 1e-5
    breakpoints[-1] += 1e-5
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    expected_percents = np.where(expected_percents == 0, 1e-4, expected_percents)
    actual_percents = np.where(actual_percents == 0, 1e-4, actual_percents)
    return float(np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents)))

# 3. Runbook Automatizado RB-001: Rollback por Data Drift y Caída de Recall
def ejecutar_runbook_rb001_rollback(psi_val: float, recall_val: float, trace_id: str):
    extra = {"trace_id": trace_id, "span_id": "rb-001-exec", "model_version": "v1.2-prod", "event_type": "INCIDENT_P1_TRIGGER"}
    logger.error(f"ALERTA CRÍTICA ALR-001: PSI={psi_val:.3f} (>0.25) y Recall={recall_val:.3f} (<0.70). Iniciando RB-001.", extra=extra)
    extra["event_type"] = "RUNBOOK_ACTION"
    logger.info("Paso 1 [RB-001]: Congelando tráfico a 'v1.2-prod' y aislando lote con drift.", extra=extra)
    logger.info("Paso 2 [RB-001]: Rollback en MLflow Registry a versión estable 'v1.1-robust-fallback'.", extra=extra)
    logger.info("Paso 3 [RB-001]: Disparando pipeline asíncrono de reentrenamiento.", extra=extra)
    extra["model_version"] = "v1.1-robust-fallback"
    extra["event_type"] = "SERVICE_RECOVERED"
    logger.info("Recuperación validada: Modelo 'v1.1-robust-fallback' en línea. Recall restaurado a 0.752.", extra=extra)
    return "v1.1-robust-fallback"

# 4. Runbook Automatizado RB-002: Auto-Scaling por Alta Latencia P95
def ejecutar_runbook_rb002_latencia(latencia_p95: float, burn_rate: float, trace_id: str):
    extra = {"trace_id": trace_id, "span_id": "rb-002-exec", "model_version": "v1.1-robust-fallback", "event_type": "INCIDENT_P2_TRIGGER"}
    logger.warning(f"ALERTA ALTA ALR-002: Latencia P95={latencia_p95:.1f}ms (>150ms), BurnRate={burn_rate:.1f}x. Iniciando RB-002.", extra=extra)
    extra["event_type"] = "RUNBOOK_ACTION"
    logger.info("Paso 1 [RB-002]: Purgando caché de inferencia saturada.", extra=extra)
    logger.info("Paso 2 [RB-002]: Escalando horizontalmente réplicas de contenedores (2 -> 5 réplicas).", extra=extra)
    extra["event_type"] = "SERVICE_RECOVERED"
    logger.info("Recuperación validada: Latencia P95 estabilizada en 82.4ms.", extra=extra)
    return 82.4

# 5. Simulación de 24 horas en producción y generación de Dashboards
def simular_monitoreo():
    mlflow.set_tracking_uri("sqlite:///mlflow_monitoring.db")
    mlflow.set_experiment("Monitorizacion_Proactiva_Produccion")
    np.random.seed(42)
    ref_cargos = np.random.normal(65.0, 20.0, 1000)
    historial = []
    version_actual = "v1.2-prod"

    with mlflow.start_run(run_name="Production_Observability_Simulation_24h"):
        for hora in range(1, 25):
            trace_id = f"tr-{uuid.uuid4().hex[:8]}"
            span_id = f"sp-{hora:02d}00"

            if hora <= 8:
                prod_cargos = np.random.normal(66.0, 20.5, 500)
                recall_sim, roc_auc_sim = np.random.uniform(0.73, 0.76), np.random.uniform(0.81, 0.83)
                latencia_p95, error_rate, estado = np.random.uniform(75.0, 95.0), np.random.uniform(0.0002, 0.0008), "Normal"
            elif 9 <= hora <= 11:
                prod_cargos = np.random.normal(92.0, 28.0, 500)
                recall_sim, roc_auc_sim = np.random.uniform(0.54, 0.59), np.random.uniform(0.66, 0.69)
                latencia_p95, error_rate, estado = np.random.uniform(85.0, 105.0), np.random.uniform(0.001, 0.002), "Incidente_1_Data_Drift"
            elif hora == 12:
                prod_cargos = np.random.normal(92.0, 28.0, 500)
                psi_temp = calcular_psi(ref_cargos, prod_cargos)
                version_actual = ejecutar_runbook_rb001_rollback(psi_temp, 0.56, trace_id)
                ref_cargos = prod_cargos.copy()
                recall_sim, roc_auc_sim, latencia_p95, error_rate, estado = 0.752, 0.815, 88.0, 0.0005, "Mitigacion_RB001_Rollback"
            elif 13 <= hora <= 16:
                prod_cargos = np.random.normal(91.5, 27.0, 500)
                recall_sim, roc_auc_sim = np.random.uniform(0.74, 0.76), np.random.uniform(0.81, 0.82)
                latencia_p95, error_rate, estado = np.random.uniform(80.0, 92.0), np.random.uniform(0.0003, 0.0007), "Normal_Post_RB001"
            elif 17 <= hora <= 19:
                prod_cargos = np.random.normal(92.0, 27.5, 500)
                recall_sim, roc_auc_sim = np.random.uniform(0.73, 0.75), np.random.uniform(0.80, 0.82)
                latencia_p95, error_rate, estado = np.random.uniform(210.0, 265.0), np.random.uniform(0.008, 0.015), "Incidente_2_Alta_Latencia"
            elif hora == 20:
                prod_cargos = np.random.normal(91.0, 27.0, 500)
                recall_sim, roc_auc_sim = 0.748, 0.818
                latencia_p95 = ejecutar_runbook_rb002_latencia(245.0, 8.5, trace_id)
                error_rate, estado = 0.0004, "Mitigacion_RB002_Scaling"
            else:
                prod_cargos = np.random.normal(91.5, 26.5, 500)
                recall_sim, roc_auc_sim = np.random.uniform(0.74, 0.77), np.random.uniform(0.81, 0.83)
                latencia_p95, error_rate, estado = np.random.uniform(76.0, 88.0), np.random.uniform(0.0002, 0.0005), "Normal_Estable"

            psi_val = calcular_psi(ref_cargos, prod_cargos)
            _, ks_pval = ks_2samp(ref_cargos, prod_cargos)
            burn_rate = (error_rate / 0.001) + (max(0, latencia_p95 - 150.0) / 25.0)

            mlflow.log_metric("psi_cargos_mensuales", psi_val, step=hora)
            mlflow.log_metric("rolling_recall", recall_sim, step=hora)
            mlflow.log_metric("latency_p95_ms", latencia_p95, step=hora)
            mlflow.log_metric("error_budget_burn_rate", burn_rate, step=hora)

            extra = {"trace_id": trace_id, "span_id": span_id, "model_version": version_actual, "event_type": "METRIC_HEARTBEAT"}
            logger.info(f"Hora {hora:02d}:00 | Estado={estado} | PSI={psi_val:.3f} | Recall={recall_sim:.3f} | Latencia_P95={latencia_p95:.1f}ms | BurnRate={burn_rate:.2f}x", extra=extra)

            historial.append({
                "Hora": hora, "Estado": estado, "Version_Modelo": version_actual,
                "PSI": round(psi_val, 4), "KS_pvalue": round(ks_pval, 5),
                "Recall": round(recall_sim, 4), "ROC_AUC": round(roc_auc_sim, 4),
                "Latencia_P95_ms": round(latencia_p95, 2), "Burn_Rate": round(burn_rate, 2)
            })

    df_hist = pd.DataFrame(historial)
    df_hist.to_csv("evidencias/telemetria_24h_simulacion.csv", index=False)

    # Generar gráfico del Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=200)
    horas = df_hist["Hora"]
    axes[0, 0].plot(horas, df_hist["PSI"], marker="o", color="#1E40AF", label="PSI")
    axes[0, 0].axhline(0.25, color="red", linestyle="--", label="Umbral Crítico (0.25)")
    axes[0, 0].set_title("1. Data Drift (PSI)")
    axes[0, 0].legend()

    axes[0, 1].plot(horas, df_hist["Recall"], marker="s", color="#047857", label="Recall")
    axes[0, 1].axhline(0.70, color="red", linestyle="--", label="SLO Recall (0.70)")
    axes[0, 1].set_title("2. Desempeño del Modelo (Recall vs SLO)")
    axes[0, 1].legend()

    axes[1, 0].plot(horas, df_hist["Latencia_P95_ms"], marker="d", color="#B45309", label="Latencia P95 (ms)")
    axes[1, 0].axhline(150.0, color="red", linestyle="--", label="SLO Latencia (150ms)")
    axes[1, 0].set_title("3. Latencia P95 (ms)")
    axes[1, 0].legend()

    axes[1, 1].bar(horas, df_hist["Burn_Rate"], color="#DC2626", label="Burn Rate (x)")
    axes[1, 1].axhline(6.0, color="black", linestyle="--", label="Alerta Burn Rate (6x)")
    axes[1, 1].set_title("4. Error Budget Burn Rate")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("evidencias/dashboard_operativo_tiempo_real.png")
    plt.close()
    print("✔ Simulación completada: Logs, CSV, base de datos MLflow y Dashboard generados en 'evidencias/'.")

if __name__ == "__main__":
    simular_monitoreo()
EOF
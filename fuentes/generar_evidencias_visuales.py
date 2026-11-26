import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

os.makedirs("evidencias", exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold"})

def generar_graficos():
    # Cargar datos y trazas reales generadas
    df_ref = pd.read_csv("datos/referencia_train.csv")
    df_prod = pd.read_csv("datos/flujo_produccion.csv")

    with open("evidencias/trazas_incidentes.jsonl", "r", encoding="utf-8") as f:
        trazas = [json.loads(line) for line in f]

    etapas = [
        "V1: Normal\n(Baseline)",
        "V2: Incidente 1\n(Data Drift P1)",
        "V3: Incidente 2\n(Latencia P2)",
        "V4: Servicio\nRestaurado"
    ]
    f1_vals = [t["metricas"]["f1_score"] for t in trazas]
    lat_vals = [t["metricas"]["latencia_p95_ms"] for t in trazas]
    ks_vals = [t["metricas"]["ks_pvalue_LIMIT_BAL"] for t in trazas]
    # Consumo acumulado simulado del Error Budget mensual (0.1% permitido)
    eb_vals = [12.0, 48.0, 86.5, 88.0]

    # ---------------------------------------------------------
    # IMAGEN 1: Dashboard Operativo Integral (4 Paneles en Tiempo Real)
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("DASHBOARD OPERATIVO EN TIEMPO REAL - MOTOR SCORING CREDITICIO (UCI)", fontsize=14, fontweight="bold")

    # Panel 1: Calidad Predictiva (F1-Score)
    axs[0, 0].plot(etapas, f1_vals, marker="o", markersize=8, linewidth=2.5, color="#0b5394", label="F1-Score Observado")
    axs[0, 0].axhline(0.52, color="#d32f2f", linestyle="--", linewidth=1.8, label="SLO Mínimo (0.52)")
    axs[0, 0].annotate("ALERTA P1\n(RB-01 Activado)", xy=(1, f1_vals[1]), xytext=(1.15, f1_vals[1] - 0.06),
                       arrowprops=dict(arrowstyle="->", color="red"), color="red", fontweight="bold")
    axs[0, 0].set_title("1. Salud Predictiva del Modelo (F1-Score)")
    axs[0, 0].set_ylabel("F1-Score (Clase Impago)")
    axs[0, 0].set_ylim(0.30, 0.70)
    axs[0, 0].legend(loc="lower right")
    axs[0, 0].grid(alpha=0.3)

    # Panel 2: Latencia p95 vs SLO
    colores_lat = ["#2e7d32" if v <= 80.0 else "#c62828" for v in lat_vals]
    bars = axs[0, 1].bar(etapas, lat_vals, color=colores_lat, width=0.5)
    axs[0, 1].axhline(80.0, color="#d32f2f", linestyle="--", linewidth=1.8, label="SLO Latencia p95 (80 ms)")
    axs[0, 1].annotate("ALERTA P2\n(RB-02 Activado)", xy=(2, lat_vals[2]), xytext=(1.35, 125),
                       arrowprops=dict(arrowstyle="->", color="red"), color="red", fontweight="bold")
    for bar, val in zip(bars, lat_vals):
        axs[0, 1].text(bar.get_x() + bar.get_width()/2, val + 3, f"{val:.1f} ms", ha="center", fontweight="bold")
    axs[0, 1].set_title("2. Latencia p95 por Inferencia (ms)")
    axs[0, 1].set_ylabel("Milisegundos (ms)")
    axs[0, 1].set_ylim(0, 165)
    axs[0, 1].legend(loc="upper left")
    axs[0, 1].grid(alpha=0.3)

    # Panel 3: Monitor de Data Drift (Test Kolmogorov-Smirnov)
    colores_ks = ["#2e7d32" if p >= 0.05 else "#c62828" for p in ks_vals]
    axs[1, 0].bar(etapas, ks_vals, color=colores_ks, width=0.5)
    axs[1, 0].axhline(0.05, color="#d32f2f", linestyle="--", linewidth=1.8, label="Umbral Drift KS (p = 0.05)")
    axs[1, 0].set_title("3. Detección Estadística de Data Drift (p-value KS)")
    axs[1, 0].set_ylabel("p-value (LIMIT_BAL)")
    axs[1, 0].set_ylim(0, max(ks_vals) + 0.15)
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].grid(alpha=0.3)

    # Panel 4: Consumo de Error Budget (%)
    axs[1, 1].fill_between(etapas, eb_vals, color="#f57c00", alpha=0.25)
    axs[1, 1].plot(etapas, eb_vals, marker="s", markersize=8, linewidth=2.5, color="#e65100", label="Error Budget Consumido (%)")
    axs[1, 1].axhline(50.0, color="#fbc02d", linestyle="--", linewidth=1.8, label="Alerta Preventiva (50%)")
    axs[1, 1].axhline(100.0, color="#d32f2f", linestyle="-", linewidth=1.8, label="Agotamiento Budget (100%)")
    axs[1, 1].set_title("4. Consumo Acumulado de Error Budget (SLO 99.9%)")
    axs[1, 1].set_ylabel("Porcentaje Consumido (%)")
    axs[1, 1].set_ylim(0, 115)
    axs[1, 1].legend(loc="upper left")
    axs[1, 1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("evidencias/1_dashboard_operativo_4paneles.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # IMAGEN 2: Comparativa de Distribución de Data Drift (Referencia vs Incidente)
    # ---------------------------------------------------------
    lote_drift = df_prod.iloc[3000:6000].copy()
    lote_drift["LIMIT_BAL"] = lote_drift["LIMIT_BAL"] - 1.8
    lote_drift["PAY_0"] = lote_drift["PAY_0"] + 2.4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("EVIDENCIA DE DATA DRIFT: DISTRIBUCIÓN BASELINE VS. INCIDENTE P1", fontsize=13, fontweight="bold")

    ax1.hist(df_ref["LIMIT_BAL"], bins=30, density=True, alpha=0.6, color="#1976d2", label="Referencia (Train)")
    ax1.hist(lote_drift["LIMIT_BAL"], bins=30, density=True, alpha=0.6, color="#d32f2f", label="Producción (Drift -1.8σ)")
    ax1.set_title("Desplazamiento en Límite de Crédito (LIMIT_BAL)")
    ax1.set_xlabel("Valor Estandarizado (Z-score)")
    ax1.set_ylabel("Densidad")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.hist(df_ref["PAY_0"], bins=20, density=True, alpha=0.6, color="#1976d2", label="Referencia (Train)")
    ax2.hist(lote_drift["PAY_0"], bins=20, density=True, alpha=0.6, color="#d32f2f", label="Producción (Drift +2.4σ)")
    ax2.set_title("Desplazamiento en Historial de Mora (PAY_0)")
    ax2.set_xlabel("Valor Estandarizado (Z-score)")
    ax2.set_ylabel("Densidad")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("evidencias/2_analisis_distribucion_drift.png", dpi=300)
    plt.close()

    print("Imágenes generadas con éxito en evidencias/:")
    print(" - evidencias/1_dashboard_operativo_4paneles.png")
    print(" - evidencias/2_analisis_distribucion_drift.png")

if __name__ == "__main__":
    generar_graficos()
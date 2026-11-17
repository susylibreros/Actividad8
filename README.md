# Actividad8
cat << 'EOF' > .gitignore
__pycache__/
*.pyc
mlruns/
*.db
.env
EOF

cat << 'EOF' > README.md
# Monitorización Proactiva, Observabilidad y Gobernanza MLOps

Este repositorio contiene la implementación del sistema de monitorización en tiempo real para el servicio `churn-prediction-service`, integrando:
- **Telemetría Unificada:** Métricas (SLIs), Logs estructurados en JSON y Trazas con `trace_id` / `span_id`.
- **Detección de Data Drift:** Cálculo horario del Population Stability Index (PSI) y prueba Kolmogorov-Smirnov (KS).
- **Alertas Inteligentes basadas en SLOs y Error Budgets:** Reglas multi-ventana en `config/alert_rules_slo.json`.
- **Simulación de Incidentes y Runbooks Automatizados:**
  - `RB-001`: Rollback automático en MLflow Model Registry ante degradación por Data Drift.
  - `RB-002`: Auto-scaling de réplicas y purga de caché ante violación de latencia P95 (>150ms).

## Cómo ejecutar la simulación
```bash
python scripts/monitor_proactivo_mlops.py
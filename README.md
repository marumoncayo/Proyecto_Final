# Proyecto_Final





**Universidad San Francisco de Quito**  
**Activo:** AAPL (Apple Inc.)



**Nombre:** Maria Eulalia Moncayo
### **Objetivo**
Predecir si el precio de AAPL cerrará **arriba o abajo** respecto a su apertura diaria usando Machine Learning.

### **Resultados Principales**
-  **Modelo:** LightGBM (tuneado con GridSearchCV)
-  **Accuracy en TEST:** 79.74%
-  **F1 Score:** 0.8175 (81.75%)
-  **Simulación:** $10,000 → $28,862 en 2025 (+188.62% retorno)
-  **API REST:** Desplegada y funcional en Docker

Arquitectura 



```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Yahoo Finance  │────▶│   Postgres   │────▶│ Feature Builder │
│   (yfinance)    │     │  (raw.*)     │     │  (analytics.*)  │
└─────────────────┘     └──────────────┘     └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Jupyter Notebook │
                    │   (ML Pipeline)  │
                    └──────────────────┘
                              │
                              ▼
                        ┌──────────┐
                        │ Model API│
                        │  (Flask) │


```
##  Componentes del Sistema

### **Servicios Docker**
| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| `postgres` | 5432 | Base de datos PostgreSQL |
| `jupyter-notebook` | 8888 | Entorno de desarrollo Python |
| `pgadmin` | 5050 | Interfaz web para Postgres |
| `feature-builder` | - | Worker para construir features |
| `model-api` | 5000 | API REST del modelo |

### **Base de Datos**
```sql
-- Esquema RAW (datos crudos)
raw.prices_daily          -- Precios OHLCV diarios

-- Esquema ANALYTICS (features procesadas)
analytics.daily_features  -- Features + target para ML
```

##  Quick Start

### **1. Prerequisitos**
- Docker & Docker Compose instalados
- 8GB RAM mínimo
- 10GB espacio en disco

### **2. Configuración Inicial**
```bash
# Clonar repositorio
git clone [tu-repo]
cd proyecto-trading

# Copiar variables de ambiente
cp .env.example .env
```
### **3. Levantar Servicios**
```bash
# Levantar Postgres y Jupyter
docker compose up -d postgres jupyter-notebook

# Verificar que estén corriendo
docker compose ps
```

### **4. Ejecutar Pipeline Completo**

#### **Paso 1: Ingesta de Precios**
```bash
# Acceder a Jupyter: http://localhost:8888
# Ejecutar: notebooks/01_ingesta_prices_raw.ipynb
```
---       
**Resultado:** ~1,232 registros en `raw.prices_daily` (2021-2025)

#### **Paso 2: Construir Features**
```bash
docker compose run --rm feature-builder \
  --mode full \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2025-11-28 \
  --run-id run_001 \
  --overwrite true
```

**Resultado:** ~1,232 registros en `analytics.daily_features`

#### **Paso 3: Entrenar Modelo**
```bash
# En Jupyter, ejecutar: notebooks/ml_trading_classifier.ipynb
```

**Resultado:** Modelo tuneado en `models/best_model.pkl`

#### **Paso 4: Levantar API**
```bash
docker compose up -d model-api

# Verificar
curl http://localhost:5000/health
```

---

##  Datos y Features

### **Activo Analizado**
- **Ticker:** AAPL (Apple Inc.)
- **Período:** 2021-01-04 a 2025-11-26
- **Días bursátiles:** 1,222 (después de limpieza)

### **Features Utilizadas (12 features)**

| Categoría | Features |
|-----------|----------|
| **Lags de precios** | `close_lag1`, `close_lag2`, `close_lag3` |
| **Retornos** | `return_prev_close` |
| **Volatilidad** | `volatility_5_days`, `volatility_10_days`, `volatility_20_days` |
| **Volumen** | `volume_lag1` |
| **Temporales** | `day_of_week`, `month` |
| **Flags** | `is_monday`, `is_friday` |


### **Target**
```python
target_up = 1  si  close > open  (precio subió)
target_up = 0  si  close <= open (precio bajó o igual)
```

**Distribución:** ~54% días UP, ~46% días DOWN (clases balanceadas)

---

##  Modelos Entrenados

### **Comparación de 8 Modelos**

| Modelo | Accuracy VAL | F1 VAL | ROC-AUC | Tiempo |
|--------|--------------|--------|---------|--------|
| Logistic Regression | 76.19% | 0.8013 | 0.8696 | 0.12s |
| Linear SVC | 76.98% | 0.8105 | - | 0.07s |
| Decision Tree | 77.38% | 0.8028 | 0.8195 | 0.03s |
| Random Forest | 79.37% | 0.8156 | 0.8233 | 0.80s |
| Gradient Boosting | 77.78% | 0.8000 | 0.8306 | 0.37s |
| XGBoost | 78.97% | 0.8114 | 0.8602 | 1.44s |
| **LightGBM** | **79.76%** | **0.8172** | **0.8468** | **0.32s** ✅ |
| CatBoost | 78.97% | 0.8166 | 0.8647 | 0.80s |

### **Modelo Seleccionado: LightGBM**

**Criterios de selección:**
1.  Mayor F1 en validación (0.8172)
2.  Mejor accuracy (79.76%)
3.  Balance entre desempeño y velocidad
4.  Soporte para probabilidades (ROC-AUC disponible)

### **Hiperparámetros Tuneados (GridSearchCV)**
```python
{
    'learning_rate': 0.05,      # Más conservador
    'max_depth': 3,              # Árboles poco profundos
    'min_child_samples': 20,     # Evita overfitting
    'n_estimators': 50,          # Menos árboles
    'num_leaves': 15             # Menos complejidad
}
```

**Grid Search:** 243 combinaciones, 729 entrenamientos (CV=3)

---

##  Resultados del Modelo

### **Métricas en TEST (2025)**

| Métrica | Baseline | LightGBM Tuneado | Mejora |
|---------|----------|------------------|--------|
| **Accuracy** | 54.37% | **79.74%** | **+25.37%** |
| **F1 Score** | 0.7044 | **0.8175** | **+16.05%** |
| **Precision** | - | **76.30%** | - |
| **Recall** | - | **88.03%** | - |
| **ROC-AUC** | 0.50 | **0.8582** | **+35.82%** |

### **Matriz de Confusión (TEST - 227 días)**
```
                Predicted
              DOWN    UP
Actual DOWN    89     21
       UP      25    92
```

**Análisis:**
-  **True Positives:** 92 (predijo UP correctamente)
-  **True Negatives:** 89 (predijo DOWN correctamente)
-  **False Positives:** 21 (predijo UP, fue DOWN - compró mal)
-  **False Negatives:** 25 (predijo DOWN, fue UP - perdió oportunidad)

**Win Rate:** 79.7% (181 predicciones correctas de 227)

---

##  Simulación de Inversión

### **Estrategia Simple**
- **Capital inicial:** $10,000
- **Período:** 2025 (227 días bursátiles)
- **Regla:** 
  - Si predicción = UP → Comprar en apertura, vender al cierre
  - Si predicción = DOWN → Mantenerse en efectivo

### **Resultados**
```
 Capital inicial:  $10,000.00
 Capital final:    $28,862.37
 Ganancia:         $18,862.37
 Retorno total:    +188.62%

 Trades ejecutados: 121 de 227 días (53.3%)
   Correctos:   92 (76.0%)
   Incorrectos: 29 (24.0%)
```

### **Comparación con Estrategias Pasivas**
| Estrategia | Capital Final | Retorno | Descripción |
|------------|---------------|---------|-------------|
| Buy & Hold | ~$15,000 | +50% | Comprar y mantener todo el año |
| Baseline | ~$12,000 | +20% | Siempre comprar (sin ML) |
| **Nuestro modelo** | **$28,862** | **+188.62%** ✅ | ML con LightGBM |

### ** Disclaimer**
- **Sin costos:** No incluye comisiones (~$2/trade), slippage, ni spreads
- **Sin impuestos:** ~20-30% de las ganancias
- **Backtesting:** Resultados en datos históricos
- **Retorno realista:** ~+160% después de ajustar costos e impuestos
- **Condiciones de mercado:** 2025 fue un año alcista para AAPL

---

##  Análisis de Errores

### **Hallazgos Principales**

1. ** Error más común:** False Positives (32 casos)
   - Predijo UP pero fue DOWN
   - Impacto: Compró y perdió dinero

2. ** Volatilidad:** Factor clave en errores
   - Volatilidad promedio en **aciertos**: 0.0188
   - Volatilidad promedio en **errores**: 0.0197 (+5%)
   - **Conclusión:** El modelo falla más en días volátiles e impredecibles

3. ** Día más difícil:** Jueves
   - Tasa de error: 28.89%
   - Posible causa: Incertidumbre pre-viernes
  
4. ** Patrón detectado:** Reversión de momentum
   - False Positives: Retorno previo +0.61% (venía subiendo, se revirtió)
   - False Negatives: Retorno previo -1.23% (venía bajando, rebotó)
##  API REST

### **Endpoints Disponibles**

#### **1. Health Check**
```bash
GET http://localhost:5000/health
```
**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-07T03:52:37.879064"
}
```

#### **2. Información del Modelo**
```bash
GET http://localhost:5000/model_info
```
**Response:**
```json
{
  "model_name": "LightGBM",
  "ticker": "AAPL",
  "features": ["close_lag1", "close_lag2", ...],
   "metricas_test": {
    "accuracy": 0.7974,
    "f1": 0.8175,
    "roc_auc": 0.8582,
    "precision": 0.7630,
    "recall": 0.8803
  },
  "fecha_entrenamiento": "2025-12-06 22:30:15",
  "train_size": 995,
  "test_size": 227
}
```
#### **3. Predicción (POST /predict)**

**Request:**
```bash
POST http://localhost:5000/predict
Content-Type: application/json

{
  "close_lag1": 276.97,
  "close_lag2": 275.92,
  "close_lag3": 274.85,
  "return_prev_close": 0.0021,
  "volatility_5_days": 0.015,
  "volatility_10_days": 0.012,
  "volatility_20_days": 0.0095,
  "volume_lag1": 50000000,
  "day_of_week": 0,
  "month": 11,
  "is_monday": true,
  "is_friday": false
}
```
##  Testing y Validación

### **Tests Implementados**

1. ** Pipeline de datos**
   - Idempotencia del feature builder
   - Consistencia de fechas entre raw y analytics
   - Validación de tipos de datos

2. ** Modelo**
   - Verificación de carga de .pkl
   - Test de predicción con datos sintéticos
   - Validación de probabilidades (suman 1.0)

3. ** API**
   - Health check
   - Validación de inputs (campos requeridos)
   - Manejo de errores
   - Response time < 100ms
  
### **Métricas de Calidad**

| Métrica | Objetivo | Actual | Status |
|---------|----------|--------|--------|
| F1 Score | > 0.75 | 0.8175 | bien |
| Accuracy | > 70% | 79.74% | bien |
| ROC-AUC | > 0.75 | 0.8582 | bien |
| API Response Time | < 100ms | ~50ms | bien |
| Cobertura de fechas | > 95% | 100% | bien |

---
##  Comandos Útiles

### **Docker**
```bash
# Ver servicios activos
docker compose ps

# Ver logs de un servicio
docker compose logs jupyter-notebook
docker compose logs model-api
docker compose logs feature-builder

# Reiniciar un servicio
docker compose restart model-api

# Detener todo
docker compose down

# Detener y eliminar volúmenes (BORRA LA BD)
docker compose down -v
# Reconstruir imagen
docker compose build feature-builder
```

### **Postgres**
```bash
# Conectar a Postgres
docker compose exec postgres psql -U trading_user -d trading_db

# Dentro de psql:
\dt raw.*                           # Ver tablas en raw
\dt analytics.*                     # Ver tablas en analytics
\d raw.prices_daily                 # Describir tabla

# Queries útiles
SELECT COUNT(*) FROM raw.prices_daily;
SELECT COUNT(*) FROM analytics.daily_features;

SELECT ticker, COUNT(*), MIN(date), MAX(date) 
FROM raw.prices_daily 
GROUP BY ticker;

# Salir
\q
```
### **Feature Builder**
```bash
# Construir features para AAPL
docker compose run --rm feature-builder \
  --mode full \
  --ticker AAPL \
  --start-date 2021-01-01 \
  --end-date 2025-11-28 \
  --run-id run_001 \
  --overwrite true

# Construir solo un rango de fechas
docker compose run --rm feature-builder \
  --mode by-date-range \
  --ticker AAPL \
  --start-date 2025-01-01 \
  --end-date 2025-11-28 \
  --run-id run_002 \
  --overwrite false
```

---

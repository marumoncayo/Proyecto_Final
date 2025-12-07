#!/usr/bin/env python3
"""
API REST para predicción de movimientos de precio de AAPL
Endpoint: POST /predict
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Configuración
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/best_model.pkl')
SCALER_PATH = '/app/models/scaler.pkl'
IMPUTER_PATH = '/app/models/imputer.pkl'
FEATURES_PATH = '/app/models/feature_columns.pkl'
METADATA_PATH = '/app/models/model_metadata.pkl'

# Cargar modelo y componentes al inicio
print(" Cargando modelo...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
imputer = joblib.load(IMPUTER_PATH)
feature_columns = joblib.load(FEATURES_PATH)
metadata = joblib.load(METADATA_PATH)
print(" Modelo cargado exitosamente")


@app.route('/', methods=['GET'])
def home():
    """Endpoint raíz con información del modelo"""
    return jsonify({
        'status': 'online',
        'model': metadata['model_name'],
        'ticker': metadata['ticker'],
        'features': metadata['features'],
        'metricas_test': metadata['metricas_test'],
        'fecha_entrenamiento': metadata['fecha_entrenamiento'],
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'model_info': '/model_info (GET)'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """Información detallada del modelo"""
    return jsonify(metadata)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint principal de predicción
    
    Input JSON:
    {
        "close_lag1": 180.5,
        "close_lag2": 179.2,
        "close_lag3": 178.8,
        "return_prev_close": 0.005,
        "volatility_5_days": 0.02,
        "volatility_10_days": 0.018,
        "volatility_20_days": 0.019,
        "volume_lag1": 50000000,
        "day_of_week": 0,
        "month": 1,
        "is_monday": true,
        "is_friday": false
    }
    
    Output JSON:
    {
        "prediction": 1,
        "prediction_label": "UP",
        "probability_up": 0.75,
        "probability_down": 0.25,
        "confidence": "high",
        "timestamp": "2025-01-15T10:30:00"
    }
    """
    try:
        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No se proporcionaron datos',
                'example': {feat: 0.0 for feat in feature_columns}
            }), 400
        
        # Validar que todas las features estén presentes
        missing_features = [f for f in feature_columns if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Features faltantes: {missing_features}',
                'required_features': feature_columns
            }), 400
        
        # Crear DataFrame con las features en el orden correcto
        input_df = pd.DataFrame([data])[feature_columns]
        
        # Preprocesar (imputar + escalar)
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        
        # Predicción
        prediction = int(model.predict(input_scaled)[0])
        probabilities = model.predict_proba(input_scaled)[0]
        prob_down = float(probabilities[0])
        prob_up = float(probabilities[1])
        
        # Determinar confianza
        confidence = 'high' if max(prob_up, prob_down) > 0.7 else 'medium' if max(prob_up, prob_down) > 0.55 else 'low'
        
        # Respuesta
        response = {
            'prediction': prediction,
            'prediction_label': 'UP' if prediction == 1 else 'DOWN',
            'probability_up': round(prob_up, 4),
            'probability_down': round(prob_down, 4),
            'confidence': confidence,
            'recommendation': 'BUY at open, SELL at close' if prediction == 1 else 'HOLD (stay in cash)',
            'timestamp': datetime.now().isoformat(),
            'model': metadata['model_name'],
            'ticker': metadata['ticker']
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


if __name__ == '__main__':
    port = int(os.getenv('API_PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

#!/usr/bin/env python3
"""
build_features.py - CLI para construir analytics.daily_features

Este script lee datos de raw.prices_daily y construye la tabla
analytics.daily_features con features para Machine Learning.

Uso:
    python build_features.py --mode full --ticker AAPL --start-date 2021-01-01 \
                             --end-date 2025-11-28 --run-id run_001 --overwrite true
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')


class FeatureBuilder:
    """Construye features diarias para ML desde raw.prices_daily"""
    
    def __init__(self, engine, schema_raw='raw', schema_analytics='analytics'):
        self.engine = engine
        self.schema_raw = schema_raw
        self.schema_analytics = schema_analytics
        
    def log(self, message):
        """Logger simple con timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")
    
    def load_raw_prices(self, ticker, start_date, end_date):
        """
        Carga datos de raw.prices_daily para un ticker y rango de fechas
        
        Args:
            ticker (str): Símbolo del activo
            start_date (str): Fecha inicio YYYY-MM-DD
            end_date (str): Fecha fin YYYY-MM-DD
            
        Returns:
            pd.DataFrame: Datos crudos ordenados por fecha
        """
        self.log(f"Cargando datos raw para {ticker}...")
        
        query = f"""
        SELECT 
            date, ticker, open, high, low, close, adj_close, volume,
            run_id, ingested_at_utc, source_name
        FROM {self.schema_raw}.prices_daily
        WHERE ticker = '{ticker}'
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        ORDER BY date ASC
        """
        
        df = pd.read_sql(query, self.engine)
        
        if df.empty:
            self.log(f" No se encontraron datos para {ticker} en el rango especificado")
            return None
        
        self.log(f" {len(df)} registros cargados ({df['date'].min()} a {df['date'].max()})")
        return df
    
    def calculate_features(self, df):
        """
        Calcula features de mercado desde datos raw
        
        Args:
            df (pd.DataFrame): Datos raw de precios
            
        Returns:
            pd.DataFrame: Datos con features calculadas
        """
        self.log("Calculando features...")
        
        # Copiar DataFrame
        features = df.copy()
        
        # Convertir date a datetime si no lo es
        features['date'] = pd.to_datetime(features['date'])
        
        # === FEATURES TEMPORALES ===
        features['year'] = features['date'].dt.year
        features['month'] = features['date'].dt.month
        features['day_of_week'] = features['date'].dt.dayofweek  # 0=Lunes, 6=Domingo
        
        # Flags temporales
        features['is_monday'] = (features['day_of_week'] == 0).astype(bool)
        features['is_friday'] = (features['day_of_week'] == 4).astype(bool)
        
        # === RETORNOS ===
        # Retorno intradiario (close vs open del mismo día)
        features['return_close_open'] = (
            (features['close'] - features['open']) / features['open']
        )
        
        # Retorno vs cierre previo (lag 1)
        features['close_lag1'] = features['close'].shift(1)
        features['return_prev_close'] = (
            features['close'] / features['close_lag1'] - 1
        )
        
        # === LAGS ADICIONALES ===
        features['close_lag2'] = features['close'].shift(2)
        features['close_lag3'] = features['close'].shift(3)
        features['volume_lag1'] = features['volume'].shift(1)
        
        # === VOLATILIDAD (rolling std de retornos) ===
        # Calcular retornos diarios para volatilidad
        returns = features['return_prev_close']
        
        # Volatilidad 5 días
        features['volatility_5_days'] = returns.rolling(window=5, min_periods=2).std()
        
        # Volatilidad 10 días
        features['volatility_10_days'] = returns.rolling(window=10, min_periods=5).std()
        
        # Volatilidad 20 días
        features['volatility_20_days'] = returns.rolling(window=20, min_periods=10).std()
        
        # === METADATOS ===
        features['ingested_at_utc'] = datetime.utcnow()
        
        self.log(f" Features calculadas: {len(features.columns)} columnas")
        
        return features
    
    def save_features(self, df, run_id, overwrite=False):
        """
        Guarda features en analytics.daily_features
        
        Args:
            df (pd.DataFrame): DataFrame con features
            run_id (str): ID de la ejecución
            overwrite (bool): Si True, elimina registros existentes antes de insertar
        """
        self.log(f"Guardando features en {self.schema_analytics}.daily_features...")
        
        # Agregar run_id
        df['run_id'] = run_id
        
        # Seleccionar columnas finales en el orden correcto
        columns_final = [
            # Identificación
            'date', 'ticker', 'year', 'month', 'day_of_week',
            # Precios
            'open', 'close', 'high', 'low', 'volume',
            # Retornos
            'return_close_open', 'return_prev_close',
            # Volatilidad
            'volatility_5_days', 'volatility_10_days', 'volatility_20_days',
            # Lags
            'close_lag1', 'close_lag2', 'close_lag3', 'volume_lag1',
            # Flags
            'is_monday', 'is_friday',
            # Metadatos
            'run_id', 'ingested_at_utc'
        ]
        
        df_final = df[columns_final].copy()
        
        # Si overwrite, eliminar registros existentes
        if overwrite:
            ticker = df_final['ticker'].iloc[0]
            date_min = df_final['date'].min()
            date_max = df_final['date'].max()
            
            delete_query = f"""
            DELETE FROM {self.schema_analytics}.daily_features
            WHERE ticker = '{ticker}'
              AND date >= '{date_min}'
              AND date <= '{date_max}'
            """
            
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(delete_query))
                    conn.commit()
                    self.log(f"  Registros existentes eliminados")
            except Exception as e:
                self.log(f"  No se pudieron eliminar registros existentes: {e}")
        
        # Guardar en PostgreSQL
        try:
            df_final.to_sql(
                name='daily_features',
                con=self.engine,
                schema=self.schema_analytics,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=500
            )
            
            self.log(f" {len(df_final)} registros insertados")
            self.log(f"   Fecha mín: {df_final['date'].min()}")
            self.log(f"   Fecha máx: {df_final['date'].max()}")
            
        except Exception as e:
            self.log(f" Error al guardar features: {e}")
            raise
    
    def build_full(self, ticker, start_date, end_date, run_id, overwrite=False):
        """
        Construye features completas para un ticker
        
        Args:
            ticker (str): Símbolo del activo
            start_date (str): Fecha inicio
            end_date (str): Fecha fin
            run_id (str): ID de ejecución
            overwrite (bool): Sobrescribir datos existentes
        """
        self.log("="*60)
        self.log(f"MODO: FULL - Ticker: {ticker}")
        self.log("="*60)
        
        start_time = datetime.now()
        
        # 1. Cargar datos raw
        df_raw = self.load_raw_prices(ticker, start_date, end_date)
        if df_raw is None or df_raw.empty:
            self.log(" No hay datos para procesar")
            return
        
        # 2. Calcular features
        df_features = self.calculate_features(df_raw)
        
        # 3. Guardar features
        self.save_features(df_features, run_id, overwrite)
        
        # Duración
        duration = (datetime.now() - start_time).total_seconds()
        self.log(f" Proceso completado en {duration:.2f} segundos")
        self.log("="*60)


def main():
    """Función principal del CLI"""
    
    parser = argparse.ArgumentParser(
        description='Construye analytics.daily_features desde raw.prices_daily',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Modo full para un ticker
  python build_features.py --mode full --ticker AAPL --start-date 2021-01-01 \\
                           --end-date 2025-11-28 --run-id run_001 --overwrite true
  
  # Procesar rango de fechas específico
  python build_features.py --mode by-date-range --ticker MSFT \\
                           --start-date 2024-01-01 --end-date 2024-12-31 \\
                           --run-id run_002
        """
    )
    
    # Argumentos requeridos
    parser.add_argument('--mode', 
                       choices=['full', 'by-date-range'],
                       required=True,
                       help='Modo de ejecución')
    
    parser.add_argument('--ticker',
                       type=str,
                       required=True,
                       help='Símbolo del activo (ej: AAPL)')
    
    parser.add_argument('--start-date',
                       type=str,
                       required=True,
                       help='Fecha inicio YYYY-MM-DD')
    
    parser.add_argument('--end-date',
                       type=str,
                       required=True,
                       help='Fecha fin YYYY-MM-DD')
    
    parser.add_argument('--run-id',
                       type=str,
                       required=True,
                       help='ID de la ejecución')
    
    parser.add_argument('--overwrite',
                       type=str,
                       choices=['true', 'false'],
                       default='false',
                       help='Sobrescribir datos existentes')
    
    args = parser.parse_args()
    
    # Leer configuración desde variables de ambiente
    PG_HOST = os.getenv('PG_HOST', 'postgres')
    PG_PORT = os.getenv('PG_PORT', '5432')
    PG_DB = os.getenv('PG_DB', 'trading_db')
    PG_USER = os.getenv('PG_USER', 'trading_user')
    PG_PASSWORD = os.getenv('PG_PASSWORD')
    PG_SCHEMA_RAW = os.getenv('PG_SCHEMA_RAW', 'raw')
    PG_SCHEMA_ANALYTICS = os.getenv('PG_SCHEMA_ANALYTICS', 'analytics')
    
    if not PG_PASSWORD:
        print(" ERROR: Variable PG_PASSWORD no definida")
        sys.exit(1)
    
    # Crear conexión
    connection_string = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    engine = create_engine(connection_string)
    
    # Crear builder
    builder = FeatureBuilder(engine, PG_SCHEMA_RAW, PG_SCHEMA_ANALYTICS)
    
    # Convertir overwrite a boolean
    overwrite = args.overwrite.lower() == 'true'
    
    # Ejecutar según el modo
    if args.mode == 'full' or args.mode == 'by-date-range':
        builder.build_full(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            run_id=args.run_id,
            overwrite=overwrite
        )
    
    print("\n Script finalizado exitosamente")


if __name__ == '__main__':
    main()
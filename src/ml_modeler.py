# src/ml_modeler.py
"""
This module contains the machine learning (ML) modeling engine (v2.0).
It defines, trains, and uses separate LSTM models for the main and bonus number
pools to predict lottery numbers for multi-pool games.
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
import tensorflow as tf
import random
from typing import Dict, List, Optional, Tuple

# --- Deterministic Seeding ---
SEED = 42
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# V2.0 IMPORTS: Centralized game rules and data harvester
from src.data_harvester import get_merged_data
from src.game_configs import get_game_rules, GAME_RULES

# --- Constants ---
MODELS_DIR = "models"
SEQUENCE_LENGTH = 10  # Number of past draws to use for predicting the next one

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')

# --- Dynamic Path Functions (Multi-Pool Aware) ---
def get_model_path(game: str, pool_type: str) -> str:
    """Returns the standardized path for a game's model file for a specific pool."""
    return os.path.join(MODELS_DIR, f"lstm_{game}_{pool_type}_model.keras")

def get_scaler_path(game: str, pool_type: str) -> str:
    """Returns the standardized path for a game's scaler file for a specific pool."""
    return os.path.join(MODELS_DIR, f"{game}_{pool_type}_scaler.joblib")

# --- Core ML Functions (Refactored for Multi-Pool) ---
def load_and_preprocess_pool_data(game: str, pool_type: str, df: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]]:
    """Loads and preprocesses data for a specific number pool ('main' or 'bonus')."""
    try:
        rules = get_game_rules(game)[pool_type]
        columns = rules['columns']
        
        if not columns or any(col not in df.columns for col in columns):
            logging.warning(f"Data preprocessing skipped for '{game}' {pool_type} pool: Columns not found. Expected: {columns}")
            return None

        # Drop rows with missing values in the essential pool columns
        df_pool = df.dropna(subset=columns)
        draws = df_pool[columns].values.astype(float)
        
        if len(draws) <= SEQUENCE_LENGTH:
            raise ValueError(f"Dataset for {game} {pool_type} pool is too small. Need more than {SEQUENCE_LENGTH} draws.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_draws = scaler.fit_transform(draws)

        X, y = [], []
        for i in range(len(scaled_draws) - SEQUENCE_LENGTH):
            X.append(scaled_draws[i:i + SEQUENCE_LENGTH])
            y.append(scaled_draws[i + SEQUENCE_LENGTH])

        return np.array(X), np.array(y), scaled_draws, scaler

    except Exception as e:
        logging.error(f"Error in load_and_preprocess_pool_data for '{game}' {pool_type} pool: {e}")
        return None

def build_lstm_model(input_shape: Tuple[int, int], num_outputs: int) -> Sequential:
    """Builds a dynamically sized LSTM model."""
    tf.random.set_seed(SEED)
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=50, return_sequences=True),
        LSTM(units=50),
        Dense(units=25, activation='relu'),
        Dense(units=num_outputs)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info(f"LSTM model built for {num_outputs} outputs.")
    return model

def save_model_and_scaler(game: str, pool_type: str, model: Sequential, scaler: MinMaxScaler):
    """Saves the model and scaler for a specific game and pool."""
    model_path = get_model_path(game, pool_type)
    scaler_path = get_scaler_path(game, pool_type)
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Model for '{game}' {pool_type} pool saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model or scaler for '{game}' {pool_type} pool: {e}")

def load_model_and_scaler(game: str, pool_type: str) -> Optional[Tuple[Sequential, MinMaxScaler]]:
    """Loads the model and scaler for a specific game and pool."""
    model_path = get_model_path(game, pool_type)
    scaler_path = get_scaler_path(game, pool_type)
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        logging.info(f"Model and scaler for '{game}' {pool_type} pool loaded.")
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading model or scaler for '{game}' {pool_type} pool: {e}")
        return None

def _predict_and_finalize_pool(model: Sequential, scaler: MinMaxScaler, all_scaled_draws: np.ndarray, pool_rules: Dict) -> List[int]:
    """Internal function to predict numbers for a single pool and finalize them."""
    num_to_predict = pool_rules["count"]
    max_val = pool_rules["max"]

    last_sequence = all_scaled_draws[-SEQUENCE_LENGTH:]
    input_data = np.expand_dims(last_sequence, axis=0)

    predicted_scaled = model.predict(input_data, verbose=0)
    predicted_unscaled = scaler.inverse_transform(predicted_scaled)

    predicted_numbers = np.clip(np.round(predicted_unscaled.flatten()), 1, max_val)
    
    final_numbers = set()
    for num in sorted(predicted_numbers):
        if not np.isnan(num) and len(final_numbers) < num_to_predict:
            final_numbers.add(int(num))

    # Fill if not enough unique numbers were generated
    if len(final_numbers) < num_to_predict:
        logging.warning(f"Prediction resulted in fewer than {num_to_predict} unique numbers. Filling gap.")
        all_possible = set(range(1, max_val + 1))
        missing_candidates = sorted(list(all_possible - final_numbers))
        fill_count = num_to_predict - len(final_numbers)
        final_numbers.update(missing_candidates[:fill_count])
                
    return sorted(list(final_numbers))

# --- High-Level API Functions (Multi-Pool Aware) ---

def train_and_generate_ml_numbers(game: str, epochs: int = 50) -> Optional[Dict[str, List[int]]]:
    """
    Trains, saves, and generates numbers for ALL pools of a given game.
    """
    logging.info(f"--- Starting full training process for '{game.upper()}' (v2.0) ---")
    
    df_full = get_merged_data(game)
    if df_full is None or df_full.empty:
        logging.error(f"Cannot train '{game}', no data available.")
        return None

    full_results = {"main": [], "bonus": []}
    game_rules = get_game_rules(game)

    for pool_type in ['main', 'bonus']:
        pool_rules = game_rules[pool_type]
        if pool_rules['count'] == 0:
            continue

        logging.info(f"--- Training model for '{game}' - {pool_type.upper()} POOL ---")
        
        processed_data = load_and_preprocess_pool_data(game, pool_type, df_full)
        if processed_data is None:
            logging.error(f"Failed to process data for {pool_type} pool. Skipping.")
            continue
        
        X_train, y_train, all_scaled_draws, scaler = processed_data
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_outputs = pool_rules["count"]
        model = build_lstm_model(input_shape, num_outputs)

        logging.info(f"Training {pool_type} model for {epochs} epochs...")
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, shuffle=False)
        
        save_model_and_scaler(game, pool_type, model, scaler)
        
        logging.info(f"Generating numbers from new {pool_type} model...")
        numbers = _predict_and_finalize_pool(model, scaler, all_scaled_draws, pool_rules)
        full_results[pool_type] = numbers

    return full_results


def generate_ml_numbers(game: str) -> Optional[Dict[str, List[int]]]:
    """
    Loads existing models for each pool and generates a full set of numbers.
    """
    df_full = get_merged_data(game)
    if df_full is None or df_full.empty:
        logging.error(f"Cannot generate ML numbers for '{game}', no data available.")
        return None

    full_results = {"main": [], "bonus": []}
    game_rules = get_game_rules(game)

    for pool_type in ['main', 'bonus']:
        pool_rules = game_rules[pool_type]
        if pool_rules['count'] == 0:
            continue
        
        logging.info(f"--- Generating from existing model for '{game}' - {pool_type.upper()} POOL ---")
        
        loaded_artifacts = load_model_and_scaler(game, pool_type)
        if loaded_artifacts is None:
            logging.warning(f"Could not load model for '{game}' {pool_type} pool. Run with --train first.")
            continue
        
        model, scaler = loaded_artifacts
        
        try:
            pool_columns = pool_rules['columns']
            df_pool = df_full.dropna(subset=pool_columns)
            draws = df_pool[pool_columns].values.astype(float)
            all_scaled_draws = scaler.transform(draws)
            
            numbers = _predict_and_finalize_pool(model, scaler, all_scaled_draws, pool_rules)
            full_results[pool_type] = numbers

        except Exception as e:
            logging.error(f"Failed to prepare data for {pool_type} prediction: {e}")

    # If both main and bonus pools failed, return None
    if not full_results["main"] and not full_results["bonus"]:
        return None

    return full_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Machine Learning Lottery Number Generator (v2.0)")
    parser.add_argument("--game", type=str, required=True, choices=GAME_RULES.keys(), help="The lottery game to model.")
    parser.add_argument("--train", action='store_true', help="Flag to force retraining of the model(s) for the specified game.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    
    args = parser.parse_args()
    
    if args.train:
        print(f"--- Training ML Model(s) for {args.game.upper()} ---")
        numbers = train_and_generate_ml_numbers(args.game, args.epochs)
    else:
        print(f"--- Generating Numbers using existing ML Model(s) for {args.game.upper()} ---")
        numbers = generate_ml_numbers(args.game)

    if numbers:
        print(f"\n--- Prediction for {args.game.upper()} ---")
        print(f"  Main Numbers: {numbers.get('main', 'N/A')}")
        if numbers.get('bonus'):
            print(f"  Bonus Numbers: {numbers.get('bonus')}")
    else:
        print(f"\nCould not generate numbers for {args.game.upper()}. See logs for details.")
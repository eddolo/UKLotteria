# src/ml_modeler.py
"""
This module contains the machine learning (ML) modeling engine.
It defines, trains, and uses an LSTM model to predict lottery numbers for various games.
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

# --- Deterministic Seeding ---
SEED = 42
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Import game-aware components from other modules
from .data_harvester import get_merged_data
from .statistical_modeler import GAME_RULES

# --- Constants ---
MODELS_DIR = "models"
SEQUENCE_LENGTH = 10  # Number of past draws to use for predicting the next one

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')

# --- Dynamic Path Functions ---
def get_model_path(game: str) -> str:
    """Returns the standardized path for a game's model file."""
    return os.path.join(MODELS_DIR, f"lstm_{game}_model.keras")

def get_scaler_path(game: str) -> str:
    """Returns the standardized path for a game's scaler file."""
    return os.path.join(MODELS_DIR, f"{game}_scaler.joblib")

# --- Core ML Functions ---
def load_and_preprocess_data(game: str):
    """
    Loads and preprocesses data for a specific game using the merged data pipeline.
    """
    try:
        rules = GAME_RULES[game]
        num_balls = rules["main_balls"]
        
        df = get_merged_data(game)
        if df is None:
            raise ValueError(f"Data loading and merging failed for game '{game}'.")

        # Dynamically find the main ball columns
        main_ball_cols = [col for col in df.columns if col.lower().startswith(('ball', 'n')) and 'lucky' not in col.lower() and 'star' not in col.lower() and 'thunderball' not in col.lower() and 'life' not in col.lower()]
        
        ball_columns = main_ball_cols[:num_balls]
        
        if len(ball_columns) != num_balls:
            raise ValueError(f"Expected {num_balls} main ball columns for {game}, but found {len(ball_columns)}. Columns found: {df.columns.tolist()}")

        # FIX: Drop rows with missing values in the essential ball columns to prevent NaNs
        df.dropna(subset=ball_columns, inplace=True)
            
        draws = df[ball_columns].values.astype(float)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_draws = scaler.fit_transform(draws)

        X, y = [], []
        if len(scaled_draws) <= SEQUENCE_LENGTH:
            raise ValueError(f"Dataset for {game} is too small. Need more than {SEQUENCE_LENGTH} draws.")

        for i in range(len(scaled_draws) - SEQUENCE_LENGTH):
            X.append(scaled_draws[i:i + SEQUENCE_LENGTH])
            y.append(scaled_draws[i + SEQUENCE_LENGTH])

        return np.array(X), np.array(y), scaled_draws, scaler

    except Exception as e:
        logging.error(f"Error in load_and_preprocess_data for '{game}': {e}")
        return None, None, None, None

def build_lstm_model(input_shape, num_outputs):
    """
    Builds a dynamically sized LSTM model.
    """
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

def save_model_and_scaler(game: str, model, scaler):
    """Saves the model and scaler for a specific game."""
    model_path = get_model_path(game)
    scaler_path = get_scaler_path(game)
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Model for '{game}' saved to {model_path} and scaler to {scaler_path}")
    except Exception as e:
        logging.error(f"Error saving model or scaler for '{game}': {e}")

def load_model_and_scaler(game: str):
    """Loads the model and scaler for a specific game."""
    model_path = get_model_path(game)
    scaler_path = get_scaler_path(game)
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        logging.info(f"Model and scaler for '{game}' loaded successfully.")
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading model or scaler for '{game}': {e}")
        return None, None

def _predict_and_finalize(model, scaler, all_scaled_draws, game: str):
    """Internal function to predict numbers and finalize them based on game rules."""
    rules = GAME_RULES[game]
    num_balls = rules["main_balls"]
    max_ball = rules["max_main_ball"]

    last_sequence = all_scaled_draws[-SEQUENCE_LENGTH:]
    input_data = np.expand_dims(last_sequence, axis=0)

    predicted_scaled = model.predict(input_data, verbose=0)
    predicted_unscaled = scaler.inverse_transform(predicted_scaled)

    predicted_numbers = np.clip(np.round(predicted_unscaled.flatten()), 1, max_ball)
    
    final_numbers = set()
    # FIX: Add a check for NaN to prevent the script from crashing
    for num in sorted(predicted_numbers):
        if not np.isnan(num) and len(final_numbers) < num_balls:
            final_numbers.add(int(num))

    # Fill if not enough unique numbers were generated
    if len(final_numbers) < num_balls:
        logging.warning(f"Prediction for {game} resulted in fewer than {num_balls} unique numbers. Filling gap.")
        all_possible = set(range(1, max_ball + 1))
        missing_candidates = sorted(list(all_possible - final_numbers))
        fill_count = num_balls - len(final_numbers)
        final_numbers.update(missing_candidates[:fill_count])
                
    return sorted(list(final_numbers))

# --- High-Level API Functions for CLI/UI ---

def train_and_generate_ml_numbers(game: str, epochs: int = 50):
    """
    A full-cycle function that trains, saves, and generates numbers.
    This is a deterministic process and returns a dictionary.
    """
    logging.info(f"Starting deterministic training process for '{game}'...")

    X_train, y_train, all_scaled_draws, scaler = load_and_preprocess_data(game)
    if X_train is None:
        return None

    rules = GAME_RULES[game]
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_outputs = rules["main_balls"]
    model = build_lstm_model(input_shape, num_outputs)

    logging.info(f"Training model for {epochs} epochs. This may take a moment...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, shuffle=False)
    logging.info("Training complete.")

    save_model_and_scaler(game, model, scaler)

    logging.info("Generating numbers from the new model...")
    main_numbers = _predict_and_finalize(model, scaler, all_scaled_draws, game)

    # Ensure consistent dictionary output
    return {"main": main_numbers, "special": []}


def generate_ml_numbers(game: str):
    """
    Loads an existing model and generates numbers deterministically using the merged dataset.
    """
    model, scaler = load_model_and_scaler(game)
    if model is None:
        logging.error(f"Could not load existing model for '{game}'. Cannot generate numbers.")
        return None

    try:
        df = get_merged_data(game)
        if df is None:
            raise ValueError(f"Data loading and merging failed for game '{game}'.")
            
        rules = GAME_RULES[game]
        num_balls = rules["main_balls"]
        
        main_ball_cols = [col for col in df.columns if col.lower().startswith(('ball', 'n')) and 'lucky' not in col.lower() and 'star' not in col.lower() and 'thunderball' not in col.lower() and 'life' not in col.lower()]
        ball_columns = main_ball_cols[:num_balls]

        if len(ball_columns) != num_balls:
            raise ValueError(f"Expected {num_balls} main ball columns for {game}, but found {len(ball_columns)}.")

        # FIX: Drop rows with missing values in the essential ball columns to prevent NaNs
        df.dropna(subset=ball_columns, inplace=True)

        draws = df[ball_columns].values.astype(float)
        all_scaled_draws = scaler.transform(draws)
    except Exception as e:
        logging.error(f"Failed to prepare data for prediction for '{game}': {e}")
        return None
    
    # ML model now needs to return a dict similar to the statistical one
    main_numbers = _predict_and_finalize(model, scaler, all_scaled_draws, game)
    
    # For now, ML model only predicts main numbers. Special numbers are returned as an empty list.
    return {"main": main_numbers, "special": []}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Machine Learning Lottery Number Generator")
    parser.add_argument("--game", type=str, required=True, choices=GAME_RULES.keys(), help="The lottery game to model (e.g., 'lotto').")
    parser.add_argument("--train", action='store_true', help="Flag to force retraining of the model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    
    args = parser.parse_args()
    
    if args.train:
        print(f"--- Training ML Model for {args.game.upper()} ---")
        numbers = train_and_generate_ml_numbers(args.game, args.epochs)
        if numbers:
            print(f"Generated numbers after training: {numbers}")
    else:
        print(f"--- Generating Numbers using existing ML Model for {args.game.upper()} ---")
        # Need to return dict with main and special numbers
        result = generate_ml_numbers(args.game)
        if result:
            print(f"Main Numbers: {result['main']}")
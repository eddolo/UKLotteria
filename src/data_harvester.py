# src/data_harvester.py
import pandas as pd
import os
import requests
import logging
from typing import Dict, Optional, List, Any

# Import the new centralized game rules
from src.game_configs import get_game_rules, GAME_RULES

# --- Constants & Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Defines paths and URLs for data acquisition. This is separate from game *rules*.
def get_game_properties() -> Dict[str, Dict[str, str]]:
    """Generates the game properties dictionary with dynamic paths."""
    base_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    properties = {}
    for game_name in GAME_RULES.keys():
        properties[game_name] = {
            "url": f"https://www.national-lottery.co.uk/results/{game_name.replace('setforlife', 'set-for-life')}/draw-history/csv",
            "base_path": os.path.join(base_data_dir, game_name, 'history.csv'),
            "live_path": os.path.join(base_data_dir, game_name, 'live_draws.csv')
        }
    return properties

GAME_PROPERTIES = get_game_properties()
DATA_URLS: Dict[str, str] = {game: details["url"] for game, details in GAME_PROPERTIES.items()}
logging.info("GAME_PROPERTIES and DATA_URLS initialized for v2.0 architecture.")

def get_data_path(game: str, data_type: str = 'base') -> str:
    """Returns the absolute data path for a given game and data type ('base' or 'live')."""
    if game not in GAME_PROPERTIES:
        raise ValueError(f"Invalid game '{game}'. Supported games are: {list(GAME_PROPERTIES.keys())}")
    
    path_key = f"{data_type}_path"
    if path_key not in GAME_PROPERTIES[game]:
        raise ValueError(f"Invalid data_type '{data_type}'. Must be 'base' or 'live'.")
        
    return GAME_PROPERTIES[game][path_key]

def fetch_live_lottery_data(game: str) -> bool:
    """Fetches the latest draw history for a specific game and saves it to a 'live' file."""
    if game not in GAME_PROPERTIES:
        logging.error(f"Invalid game '{game}'. Cannot fetch data.")
        return False

    config = GAME_PROPERTIES[game]
    url = config["url"]
    file_path = config["live_path"]

    logging.info(f"Attempting to download '{game}' live data from {url}...")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        logging.info(f"Successfully saved '{game}' live data to {file_path}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download live data for '{game}': {e}")
        return False
    except IOError as e:
        logging.error(f"Failed to save live data to file for '{game}': {e}")
        return False

def get_merged_data(game: str) -> Optional[pd.DataFrame]:
    """Loads, merges, and validates historical and live data for a specific game."""
    base_path = get_data_path(game, 'base')
    live_path = get_data_path(game, 'live')

    try:
        if os.path.exists(base_path):
            base_df = pd.read_csv(base_path)
            logging.info(f"Loaded {len(base_df)} records from base file: {base_path}")
        else:
            base_df = pd.DataFrame()
            logging.warning(f"Base file not found at {base_path}. Proceeding with live data only.")

        if os.path.exists(live_path):
            live_df = pd.read_csv(live_path)
            logging.info(f"Loaded {len(live_df)} records from live file: {live_path}")
        else:
            live_df = pd.DataFrame()
            logging.warning(f"Live file not found at {live_path}. Proceeding with base data only.")

    except Exception as e:
        logging.error(f"CRITICAL: Failed to read CSV data for '{game}'. Error: {e}")
        return None

    if base_df.empty and live_df.empty:
        logging.error(f"No data available for '{game}' from any source.")
        return None
        
    combined_df = pd.concat([base_df, live_df], ignore_index=True)
    
    # Standardize column names before processing
    combined_df = standardize_column_names(combined_df)
    
    # Use Draw Date and Main numbers for deduplication
    game_rules = get_game_rules(game)
    main_cols = game_rules['main']['columns']
    
    # The first column is always the date column in the raw CSV
    date_col_original_name = combined_df.columns[0]
    
    # Ensure all required columns exist before trying to deduplicate
    dedupe_cols = [date_col_original_name] + main_cols
    valid_dedupe_cols = [col for col in dedupe_cols if col in combined_df.columns]

    if len(valid_dedupe_cols) < game_rules['main']['count'] + 1:
        logging.warning(f"Could not find enough columns for reliable deduplication for '{game}'. Skipping.")
    else:
        initial_rows = len(combined_df)
        combined_df.drop_duplicates(subset=valid_dedupe_cols, keep='last', inplace=True)
        final_rows = len(combined_df)
        logging.info(f"Removed {initial_rows - final_rows} duplicate records for '{game}'.")

    # Sort by date
    combined_df[date_col_original_name] = pd.to_datetime(combined_df[date_col_original_name])
    combined_df = combined_df.sort_values(by=date_col_original_name, ascending=False).reset_index(drop=True)

    return load_and_validate_data(game, df=combined_df)

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes DataFrame columns to match the `GAME_RULES` spec."""
    # A mapping from potential CSV headers to our standard internal names.
    # This is crucial because the Lottery operator changes headers occasionally.
    standard_map = {
        'Draw Date': 'Draw_Date',
        'Ball 1': 'Ball_1', 'Ball 2': 'Ball_2', 'Ball 3': 'Ball_3',
        'Ball 4': 'Ball_4', 'Ball 5': 'Ball_5', 'Ball 6': 'Ball_6',
        'Lucky Star 1': 'Lucky_Star_1', 'Lucky Star 2': 'Lucky_Star_2',
        'Thunderball': 'Thunderball',
        'Life Ball': 'Life_Ball',
        'Bonus Ball': 'Bonus_Ball' # For Lotto
    }
    
    # Apply cleaning: strip spaces and replace known variations
    new_columns = {}
    for col in df.columns:
        clean_col = col.strip()
        new_columns[col] = standard_map.get(clean_col, clean_col.replace(' ', '_'))
        
    df.rename(columns=new_columns, inplace=True)
    return df

def load_and_validate_data(game: str, df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Validates that a DataFrame contains all required columns for the specified game."""
    if df is None:
        logging.error(f"No DataFrame provided for validation for game '{game}'.")
        return None
    
    if df.empty:
        logging.warning(f"Data for '{game}' is empty, no validation possible.")
        return df

    game_rules = get_game_rules(game)
    required_cols = game_rules['main']['columns'] + game_rules['bonus']['columns']
    
    # The first column is always expected to be the date
    required_cols.insert(0, df.columns[0]) 

    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logging.error(f"VALIDATION FAILED for '{game}': Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")
        return None

    logging.info(f"VALIDATION SUCCESS for '{game}': All required columns found. Total draws: {len(df)}")
    return df

if __name__ == "__main__":
    print("--- Running Data Harvester (v2.0) for All Games ---")
    
    for game_name in GAME_RULES.keys():
        print(f"\n--- Processing: {game_name.upper()} ---")
        fetch_live_lottery_data(game_name)
        
    print("\n--- Testing Merged Data Pipeline (v2.0) ---")
    for game_name in GAME_RULES.keys():
        print(f"\n--- Merging Data for: {game_name.upper()} ---")
        merged_data = get_merged_data(game_name)
        if merged_data is not None:
            print(f"--- Pipeline Test Complete for {game_name.upper()}: Merged data appears valid. Total records: {len(merged_data)} ---")
        else:
            print(f"--- Pipeline Test FAILED for {game_name.upper()}. Please check logs. ---")
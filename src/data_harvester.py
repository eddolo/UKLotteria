# src/data_harvester.py
import pandas as pd
import os
import requests
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Game Configurations ---
# Defines paths for the large historical dataset (base) and the file for recent draws (live)
def get_game_configs() -> Dict[str, Dict[str, str]]:
    """Generates the game configuration dictionary with dynamic paths."""
    base_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    return {
        "lotto": {
            "url": "https://www.national-lottery.co.uk/results/lotto/draw-history/csv",
            "base_path": os.path.join(base_data_dir, 'lotto', 'history.csv'),
            "live_path": os.path.join(base_data_dir, 'lotto', 'live_draws.csv')
        },
        "euromillions": {
            "url": "https://www.national-lottery.co.uk/results/euromillions/draw-history/csv",
            "base_path": os.path.join(base_data_dir, 'euromillions', 'history.csv'),
            "live_path": os.path.join(base_data_dir, 'euromillions', 'live_draws.csv')
        },
        "thunderball": {
            "url": "https://www.national-lottery.co.uk/results/thunderball/draw-history/csv",
            "base_path": os.path.join(base_data_dir, 'thunderball', 'history.csv'),
            "live_path": os.path.join(base_data_dir, 'thunderball', 'live_draws.csv')
        },
        "setforlife": {
            "url": "https://www.national-lottery.co.uk/results/set-for-life/draw-history/csv",
            "base_path": os.path.join(base_data_dir, 'setforlife', 'history.csv'),
            "live_path": os.path.join(base_data_dir, 'setforlife', 'live_draws.csv')
        }
    }

GAME_CONFIGS = get_game_configs()
DATA_URLS: Dict[str, str] = {game: details["url"] for game, details in GAME_CONFIGS.items()}

def get_data_path(game: str, data_type: str = 'base') -> str:
    """
    Returns the absolute data path for a given game and data type ('base' or 'live').
    """
    if game not in GAME_CONFIGS:
        raise ValueError(f"Invalid game '{game}'. Supported games are: {list(GAME_CONFIGS.keys())}")
    
    path_key = f"{data_type}_path"
    if path_key not in GAME_CONFIGS[game]:
        raise ValueError(f"Invalid data_type '{data_type}'. Must be 'base' or 'live'.")
        
    return GAME_CONFIGS[game][path_key]

def fetch_live_lottery_data(game: str) -> bool:
    """
    Fetches the latest draw history for a specific game and saves it to a separate 'live' file.

    Args:
        game (str): The name of the lottery game.

    Returns:
        bool: True if data was successfully downloaded, False otherwise.
    """
    if game not in GAME_CONFIGS:
        logging.error(f"Invalid game '{game}'. Cannot fetch data.")
        return False

    config = GAME_CONFIGS[game]
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
    """
    Loads the base historical data and the live data, merges them,
    removes duplicates, and returns a final, comprehensive DataFrame.

    Args:
        game (str): The name of the lottery game.

    Returns:
        pd.DataFrame or None: A merged DataFrame or None if data is missing.
    """
    base_path = get_data_path(game, 'base')
    live_path = get_data_path(game, 'live')

    # Load base historical data
    if not os.path.exists(base_path):
        logging.warning(f"Base historical data file not found for '{game}' at {base_path}. Proceeding with live data only.")
        base_df = pd.DataFrame()
    else:
        try:
            base_df = pd.read_csv(base_path)
            # Standardize date format for merging
            base_df['Draw Date'] = pd.to_datetime(base_df.iloc[:, 0]).dt.strftime('%d-%b-%Y')
        except Exception as e:
            logging.error(f"Could not read or process base file {base_path}: {e}")
            return None

    # Load live data
    if not os.path.exists(live_path):
        logging.warning(f"Live data file not found for '{game}' at {live_path}. Returning base data only.")
        return load_and_validate_data(game, df=base_df)
    else:
        try:
            live_df = pd.read_csv(live_path)
            # Standardize date format for merging
            live_df['Draw Date'] = pd.to_datetime(live_df.iloc[:, 0]).dt.strftime('%d-%b-%Y')
        except Exception as e:
            logging.error(f"Could not read or process live file {live_path}: {e}")
            return load_and_validate_data(game, df=base_df)
    
    # Merge and de-duplicate
    logging.info(f"Merging {len(base_df)} historical records with {len(live_df)} live records for '{game}'.")
    combined_df = pd.concat([base_df, live_df], ignore_index=True)
    
    # Identify a consistent set of columns for deduplication
    # We use the date and first 6 numbers as a unique key for a draw.
    subset_cols = [col for col in combined_df.columns if col.startswith('Ball') or col in ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']]
    subset_cols = ['Draw Date'] + subset_cols[:6]
    
    # Ensure all subset columns exist before trying to deduplicate
    valid_subset_cols = [col for col in subset_cols if col in combined_df.columns]
    
    if len(valid_subset_cols) < 7:
        logging.warning("Could not find enough columns to reliably deduplicate. Skipping.")
    else:
        initial_rows = len(combined_df)
        combined_df.drop_duplicates(subset=valid_subset_cols, keep='last', inplace=True)
        final_rows = len(combined_df)
        logging.info(f"Removed {initial_rows - final_rows} duplicate records.")

    # Sort by date in descending order
    date_col = combined_df.columns[0]
    combined_df[date_col] = pd.to_datetime(combined_df[date_col])
    combined_df = combined_df.sort_values(by=date_col, ascending=False).reset_index(drop=True)

    return load_and_validate_data(game, df=combined_df)


def load_and_validate_data(game: str, df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """
    Validates a given DataFrame or loads one from the base path.

    Args:
        game (str): The name of the lottery game.
        df (pd.DataFrame, optional): A DataFrame to validate. If None, loads from disk.

    Returns:
        pd.DataFrame or None: A validated DataFrame or None if invalid.
    """
    if df is None:
        file_path = get_data_path(game, 'base')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file for '{game}' not found at: {file_path}.")
        try:
            df = pd.read_csv(file_path, header=0)
        except Exception as e:
            raise ValueError(f"Failed to read or parse the CSV file for '{game}': {e}")
    
    if df.empty:
        logging.error(f"Data for '{game}' is empty.")
        return None

    # Clean up column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
    logging.info(f"Data for '{game}' loaded. Total draws: {len(df)}")
    
    return df

if __name__ == "__main__":
    print("--- Running Data Harvester for All Games ---")
    
    games_to_update = list(GAME_CONFIGS.keys())
    for game_name in games_to_update:
        print(f"\n--- Processing: {game_name.upper()} ---")
        fetch_live_lottery_data(game_name)
        
    print("\n--- Testing Merged Data Pipeline ---")
    for game_name in games_to_update:
        print(f"\n--- Merging Data for: {game_name.upper()} ---")
        merged_data = get_merged_data(game_name)
        if merged_data is not None:
            print(f"--- Validation Complete for {game_name.upper()}: Merged data appears valid. Total records: {len(merged_data)} ---")
        else:
            print(f"--- Merging Failed for {game_name.upper()}. Please check logs. ---")
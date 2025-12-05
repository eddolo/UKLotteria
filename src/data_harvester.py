# src/data_harvester.py
import pandas as pd
import os
import requests
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Game Configurations ---
# A centralized place for game-specific details
GAME_CONFIGS: Dict[str, Dict[str, str]] = {
    "lotto": {
        "url": "https://www.national-lottery.co.uk/results/lotto/draw-history/csv",
        "data_path": os.path.join(os.path.dirname(__file__), '..', 'data', 'lotto', 'history.csv')
    },
    "euromillions": {
        "url": "https://www.national-lottery.co.uk/results/euromillions/draw-history/csv",
        "data_path": os.path.join(os.path.dirname(__file__), '..', 'data', 'euromillions', 'history.csv')
    },
    "thunderball": {
        "url": "https://www.national-lottery.co.uk/results/thunderball/draw-history/csv",
        "data_path": os.path.join(os.path.dirname(__file__), '..', 'data', 'thunderball', 'history.csv')
    },
    "setforlife": {
        "url": "https://www.national-lottery.co.uk/results/set-for-life/draw-history/csv",
        "data_path": os.path.join(os.path.dirname(__file__), '..', 'data', 'setforlife', 'history.csv')
    }
}

# Exportable constant for the UI to display data source URLs
DATA_URLS: Dict[str, str] = {game: details["url"] for game, details in GAME_CONFIGS.items()}


def get_data_path(game: str) -> str:
    """Returns the absolute data path for a given game."""
    if game not in GAME_CONFIGS:
        raise ValueError(f"Invalid game '{game}'. Supported games are: {list(GAME_CONFIGS.keys())}")
    return GAME_CONFIGS[game]["data_path"]

def fetch_and_save_lottery_data(game: str) -> bool:
    """
    Fetches the latest draw history for a specific game and saves it locally.

    Args:
        game (str): The name of the lottery game (e.g., 'lotto').

    Returns:
        bool: True if data was successfully downloaded and saved, False otherwise.
    """
    if game not in GAME_CONFIGS:
        logging.error(f"Invalid game '{game}'. Cannot fetch data.")
        return False

    config = GAME_CONFIGS[game]
    url = config["url"]
    file_path = config["data_path"]

    logging.info(f"Attempting to download '{game}' data from {url}...")
    try:
        # Ensure the directory for the game's data exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Fetch the data
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save the content to the CSV file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        logging.info(f"Successfully saved '{game}' data to {file_path}")
        return True

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download data for '{game}': {e}")
        return False
    except IOError as e:
        logging.error(f"Failed to save data to file for '{game}': {e}")
        return False

def load_and_validate_data(game: str) -> pd.DataFrame:
    """
    Loads and validates lottery data for a specific game from its CSV file.

    Args:
        game (str): The name of the lottery game.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the lottery data.

    Raises:
        FileNotFoundError: If the data file for the game does not exist.
        ValueError: If the file is empty or the data is invalid.
    """
    file_path = get_data_path(game)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for '{game}' not found at: {file_path}. Please run the data update first.")

    try:
        df = pd.read_csv(file_path, header=0)
    except Exception as e:
        raise ValueError(f"Failed to read or parse the CSV file for '{game}': {e}")

    if df.empty:
        raise ValueError(f"The data file for '{game}' is empty.")

    logging.info(f"Data for '{game}' loaded successfully.")
    logging.info(f"Number of draws found for '{game}': {len(df)}")
    
    # Clean up column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    return df

if __name__ == "__main__":
    print("--- Running Data Harvester for All Games ---")
    
    games_to_update = list(GAME_CONFIGS.keys())
    success_count = 0
    
    for game_name in games_to_update:
        print(f"\n--- Processing: {game_name.upper()} ---")
        if fetch_and_save_lottery_data(game_name):
            print(f"--- Validating Downloaded Data for {game_name.upper()} ---")
            try:
                load_and_validate_data(game_name)
                print(f"--- Validation Complete for {game_name.upper()}: Data appears valid. ---")
                success_count += 1
            except (FileNotFoundError, ValueError) as e:
                print(f"--- Validation Failed for {game_name.upper()}: {e} ---")
        else:
            print(f"--- Data Harvesting Failed for {game_name.upper()}. Please check logs. ---")
            
    print(f"\n--- Harvester Finished: Successfully updated {success_count}/{len(games_to_update)} games. ---")
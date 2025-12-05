# train_models.py
"""
This script is the dedicated entry point for training all machine learning models.
It ensures that the project is treated as a package, avoiding relative import errors.
"""
import os
import logging
from src.statistical_modeler import GAME_RULES
from src.ml_modeler import train_and_generate_ml_numbers, MODELS_DIR # CORRECTED IMPORT
from src.data_harvester import fetch_live_lottery_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to orchestrate the data fetching and model training process.
    """
    logging.info("--- Starting Master Training Script ---")
    
    # Ensure the models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    games_to_process = list(GAME_RULES.keys())
    
    # 1. Fetch latest data for all games first
    logging.info("--- Phase 1: Fetching all live data ---")
    fetch_success_count = 0
    for game_name in games_to_process:
        logging.info(f"Fetching live data for {game_name.upper()}...")
        if fetch_live_lottery_data(game_name):
            fetch_success_count += 1
    logging.info(f"--- Data fetching complete. Successfully updated {fetch_success_count}/{len(games_to_process)} games. ---")
    
    # 2. Train a model for each game using the newly fetched and merged data
    logging.info("\n--- Phase 2: Training all ML models ---")
    training_success_count = 0
    for game_name in games_to_process:
        logging.info(f"\n--- Processing game: {game_name.upper()} ---")
        try:
            # The train function now internally handles getting the merged data
            generated_numbers = train_and_generate_ml_numbers(game_name, epochs=50)
            if generated_numbers:
                logging.info(f"--- Successfully processed and trained model for {game_name}. Example numbers: {generated_numbers} ---")
                training_success_count += 1
            else:
                 logging.error(f"Training process ran for {game_name} but did not return numbers.")
        except Exception as e:
            logging.error(f"An unhandled error occurred during the training process for {game_name}: {e}", exc_info=True)
            
    logging.info(f"\n--- Master Training Script Finished. Successfully trained {training_success_count}/{len(games_to_process)} models. ---")

if __name__ == '__main__':
    main()
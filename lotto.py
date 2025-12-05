# lotto.py
"""
Main Command-Line Interface (CLI) for the UK Lottery Number Generator.

This tool allows users to:
1. Update the historical lottery data for a specific game.
2. Generate lottery numbers for a specific game using statistical or ML models.
"""

import argparse
import sys
import os
from src.data_harvester import fetch_live_lottery_data, get_data_path
from src.statistical_modeler import generate_statistical_numbers, GAME_RULES
from src.ml_modeler import (
    train_and_generate_ml_numbers,
    get_model_path,
    get_scaler_path,
    generate_ml_numbers as generate_from_existing_model
)

def ensure_data_exists(game: str):
    """
    Checks for the data file for a given game and downloads it if missing.
    Exits the script if the download fails.
    """
    # We trigger the download process if the main historical file is missing,
    # as the live file is always fetched.
    historical_data_path = get_data_path(game, data_type='base') # CORRECTED KEYWORD ARGUMENT
    
    if not os.path.exists(historical_data_path):
        print(f"Historical data for '{game}' not found. Attempting to download live data...")
        if not fetch_live_lottery_data(game):
            print(f"\nAutomatic data download for '{game}' failed. The application may not work correctly without data.")
        else:
            print(f"Live data for '{game}' downloaded successfully.\n")


def handle_statistical_generation(game: str):
    """Handles logic for statistical number generation."""
    print(f"Executing: Number Generation for '{game.upper()}' with Method 'statistical'")
    ensure_data_exists(game)
    numbers = generate_statistical_numbers(game)
    if numbers:
        print("\n--- Statistical Model Generated Numbers ---")
        print(f"Prediction for {game.upper()}: {numbers['main']}")
        if 'special' in numbers and numbers['special']:
            print(f"Special Numbers: {numbers['special']}")
        print("-----------------------------------------")
    else:
        print(f"Could not generate statistical numbers for '{game}'.")


def handle_ml_generation(game: str):
    """Handles logic for ML number generation."""
    print(f"Executing: Number Generation for '{game.upper()}' with Method 'ml'")
    ensure_data_exists(game)

    model_path = get_model_path(game)
    scaler_path = get_scaler_path(game)

    # If model doesn't exist, recommend training it first
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"\nModel for '{game}' not found. Please run `python train_models.py` first.")
        print("This is a one-time process to build the necessary prediction models.")
        sys.exit(1)
    else:
        # Model exists, just generate numbers
        print(f"Found existing model for '{game}'. Generating numbers...")
        numbers = generate_from_existing_model(game)
        if numbers is None:
            print(f"\nML number generation failed for '{game}'. This could be due to missing data or a model issue.")
            sys.exit(1)

    print("\n--- Machine Learning (LSTM) Generated Numbers ---")
    print(f"Prediction for {game.upper()}: {numbers['main']}")
    if 'special' in numbers and numbers['special']:
            print(f"Special Numbers: {numbers['special']}")
    print("---------------------------------------------")

def main():
    """Main function to parse arguments and execute commands."""
    supported_games = list(GAME_RULES.keys())
    parser = argparse.ArgumentParser(
        description="UK Lottery Number Generator.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # Subparser for 'update' command
    parser_update = subparsers.add_parser('update', help='Force an update of the live lottery data.')
    parser_update.add_argument(
        '--game',
        type=str,
        required=True,
        choices=supported_games,
        help=f"The lottery game to update. Choices: {', '.join(supported_games)}"
    )

    # Subparser for 'generate' command
    parser_generate = subparsers.add_parser('generate', help='Generate lottery numbers.')
    parser_generate.add_argument(
        '--game',
        type=str,
        required=True,
        choices=supported_games,
        help=f"The lottery game to generate numbers for. Choices: {', '.join(supported_games)}"
    )
    parser_generate.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['statistical', 'ml'],
        help=(
            "The prediction method to use.\n"
            "'statistical': Based on number frequencies, overdue status, etc.\n"
            "'ml': Uses a trained LSTM neural network."
        )
    )

    # Simplified argument parsing for the new structure
    args = parser.parse_args()

    if args.command == 'update':
        print(f"Executing: Force Data Update for '{args.game.upper()}'")
        if not fetch_live_lottery_data(args.game):
            print("Data update failed. Please check your connection and try again.")
            sys.exit(1)
        print("Data updated successfully.")

    elif args.command == 'generate':
        if args.method == 'statistical':
            handle_statistical_generation(args.game)
        elif args.method == 'ml':
            handle_ml_generation(args.game)

if __name__ == "__main__":
    # A simple check to guide users if they run the script with no arguments
    if len(sys.argv) == 1:
        # Instead of full help, give a hint about the new commands
        print("Usage: python lotto.py [command] [options]")
        print("Commands: 'generate', 'update'")
        print("Example: python lotto.py generate --game lotto --method ml")
        sys.exit(1)
    main()
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
from src.data_harvester import fetch_and_save_lottery_data, get_data_path
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
    data_path = get_data_path(game)
    if not os.path.exists(data_path):
        print(f"Data file for '{game}' not found at '{data_path}'.")
        print("Attempting to download it automatically...")
        if not fetch_and_save_lottery_data(game):
            print(f"\nAutomatic data download for '{game}' failed. Please check your connection.")
            sys.exit(1)
        print(f"Data for '{game}' downloaded successfully.\n")

def handle_statistical_generation(game: str):
    """Handles logic for statistical number generation."""
    print(f"Executing: Number Generation for '{game.upper()}' with Method 'statistical'")
    ensure_data_exists(game)
    numbers = generate_statistical_numbers(game)
    print("\n--- Statistical Model Generated Numbers ---")
    print(f"Prediction for {game.upper()}: {numbers}")
    print("-----------------------------------------")

def handle_ml_generation(game: str):
    """Handles logic for ML number generation."""
    print(f"Executing: Number Generation for '{game.upper()}' with Method 'ml'")
    ensure_data_exists(game)

    model_path = get_model_path(game)
    scaler_path = get_scaler_path(game)

    # If model doesn't exist, train it first
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"\nModel for '{game}' not found. A one-time training process will now begin.")
        print("This may take a few minutes...")
        numbers = train_and_generate_ml_numbers(game, epochs=50) # First time training
        if numbers is None:
            print(f"\nML model training and generation failed for '{game}'.")
            sys.exit(1)
    else:
        # Model exists, just generate numbers
        print(f"Found existing model for '{game}'. Generating numbers...")
        numbers = generate_from_existing_model(game)
        if numbers is None:
            print(f"\nML number generation failed for '{game}'.")
            sys.exit(1)

    print("\n--- Machine Learning (LSTM) Generated Numbers ---")
    print(f"Prediction for {game.upper()}: {numbers}")
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
    parser_update = subparsers.add_parser('update', help='Force an update of the historical lottery data.')
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

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command == 'update':
        print(f"Executing: Force Data Update for '{args.game.upper()}'")
        if not fetch_and_save_lottery_data(args.game):
            print("Data update failed. Please check your connection and try again.")
            sys.exit(1)
        print("Data updated successfully.")

    elif args.command == 'generate':
        if args.method == 'statistical':
            handle_statistical_generation(args.game)
        elif args.method == 'ml':
            handle_ml_generation(args.game)

if __name__ == "__main__":
    main()
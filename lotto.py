# lotto.py
"""
Main Command-Line Interface (CLI) for the UK Lottery Number Generator (v2.0).

This tool allows users to:
1.  Generate lottery numbers for a specific game using statistical or ML models.
2.  Update the historical lottery data for a specific game.
"""

import sys
import os
import argparse

# Add the project root to the Python path to allow absolute imports from 'src'
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# v2.0 Imports: Now using centralized game configurations
from src.data_harvester import fetch_live_lottery_data, get_data_path
from src.game_configs import get_game_rules, GAME_RULES
from src.statistical_modeler import generate_statistical_numbers
from src.ml_modeler import (
    get_model_path,
    generate_ml_numbers as generate_from_existing_model
)

def ensure_data_exists(game: str):
    """
    Checks for the data file for a given game and downloads it if missing.
    """
    # CORRECTED: The data harvester expects 'base' for the historical data file.
    historical_data_path = get_data_path(game, data_type='base')

    if not os.path.exists(historical_data_path):
        print(f"Historical data for '{game}' not found. Attempting to download live data...")
        if not fetch_live_lottery_data(game):
            print(f"\nAutomatic data download for '{game}' failed. The application may not work correctly without data.")
        else:
            print(f"Live data for '{game}' downloaded successfully.\n")

def print_results(game: str, numbers: dict, model_type: str):
    """
    Standardized function to print prediction results.
    It uses game-specific labels for bonus numbers.
    """
    if not numbers or (not numbers.get('main') and not numbers.get('bonus')):
        print(f"Could not generate {model_type} numbers for '{game}'.")
        return

    game_rules = get_game_rules(game)
    bonus_label = game_rules.get('bonus', {}).get('name', 'Bonus Numbers')

    print(f"\n--- {model_type.title()} Model Generated Numbers ---")
    print(f"Prediction for {game.upper()}:")
    
    if numbers.get('main'):
        print(f"  Main Numbers:  {numbers['main']}")
    
    if numbers.get('bonus'):
        print(f"  {bonus_label}: {numbers['bonus']}")
    
    print("-------------------------------------------------")

def handle_statistical_generation(game: str):
    """Handles logic for statistical number generation."""
    print(f"Executing: Number Generation for '{game.upper()}' with Method 'statistical'")
    ensure_data_exists(game)
    numbers = generate_statistical_numbers(game)
    print_results(game, numbers, 'statistical')


def handle_ml_generation(game: str):
    """Handles logic for ML number generation (v2.0)."""
    print(f"Executing: Number Generation for '{game.upper()}' with Method 'ml'")
    ensure_data_exists(game)

    # v2.0 check: We only need to verify the 'main' model exists as a proxy
    # for whether the game has been trained.
    main_model_path = get_model_path(game, 'main')

    if not os.path.exists(main_model_path):
        print(f"\nModel for '{game}' not found. Please run the training script first:")
        print(f"  python train_models.py --game {game}")
        print("\nThis is a one-time process to build the necessary prediction models.")
        sys.exit(1)
    
    print(f"Found existing model(s) for '{game}'. Generating numbers...")
    numbers = generate_from_existing_model(game)
    print_results(game, numbers, 'machine learning')


def main():
    """Main function to parse arguments and execute commands."""
    supported_games = list(GAME_RULES.keys())
    parser = argparse.ArgumentParser(
        description="UK Lottery Number Generator (v2.0).",
        formatter_class=argparse.RawTextHelpFormatter
    )

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
            "'ml': Uses trained LSTM neural network(s)."
        )
    )

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
    if len(sys.argv) == 1:
        print("Usage: python lotto.py [command] [options]")
        print("Commands: 'generate', 'update'")
        print("Example: python lotto.py generate --game euromillions --method ml")
        sys.exit(1)
    main()
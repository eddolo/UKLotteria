# src/statistical_modeler.py
"""
This module contains the statistical modeling engine for generating lottery numbers.
It analyzes historical data for various games to calculate frequencies and other properties.
"""
import pandas as pd
import logging
import random
from typing import Dict, Any, List

# --- Game Rules ---
# A centralized place for game-specific rules for analysis.
GAME_RULES: Dict[str, Dict[str, Any]] = {
    "lotto": {
        "main_balls": 6,
        "max_main_ball": 59,
        "bonus_balls": 0,
        "max_bonus_ball": None
    },
    "euromillions": {
        "main_balls": 5,
        "max_main_ball": 50,
        "bonus_balls": 2,
        "max_bonus_ball": 12,
        "bonus_ball_prefix": "lucky_star"
    },
    "thunderball": {
        "main_balls": 5,
        "max_main_ball": 39,
        "bonus_balls": 1,
        "max_bonus_ball": 14,
        "bonus_ball_prefix": "thunderball"
    },
    "setforlife": {
        "main_balls": 5,
        "max_main_ball": 47,
        "bonus_balls": 1,
        "max_bonus_ball": 10,
        "bonus_ball_prefix": "life_ball"
    }
}

def analyze_number_frequency(df: pd.DataFrame, game: str) -> pd.Series:
    """
    Analyzes the frequency of each main ball number for a specific game.

    Args:
        df (pd.DataFrame): DataFrame containing the lottery history.
        game (str): The name of the lottery game.

    Returns:
        pd.Series: A series with ball numbers as index and frequency as values.
    """
    rules = GAME_RULES[game]
    num_main_balls = rules["main_balls"]
    
    ball_columns = [col for col in df.columns if 'ball' in col and 'lucky' not in col and 'thunderball' not in col and 'life' not in col][:num_main_balls]
    if len(ball_columns) != num_main_balls:
        raise ValueError(f"Could not find the expected {num_main_balls} main ball columns for {game}.")

    all_balls = df[ball_columns].values.flatten()
    frequency = pd.Series(all_balls).value_counts()
    return frequency

def analyze_overdue_numbers(df: pd.DataFrame, game: str) -> pd.Series:
    """
    Analyzes how many draws have passed since each main number last appeared.

    Args:
        df (pd.DataFrame): DataFrame with newest draws first.
        game (str): The name of the lottery game.

    Returns:
        pd.Series: A series with ball numbers as index and overdue count as values.
    """
    rules = GAME_RULES[game]
    num_main_balls = rules["main_balls"]
    max_main_ball = rules["max_main_ball"]

    ball_columns = [col for col in df.columns if 'ball' in col and 'lucky' not in col and 'thunderball' not in col and 'life' not in col][:num_main_balls]
    if len(ball_columns) != num_main_balls:
        raise ValueError(f"Could not find the expected {num_main_balls} main ball columns for {game}.")

    all_possible_numbers = set(range(1, max_main_ball + 1))
    overdue_dict = {}

    for number in all_possible_numbers:
        is_present = df[ball_columns].isin([number]).any(axis=1)
        if is_present.any():
            last_seen_index = is_present.idxmax()
            overdue_dict[number] = last_seen_index
        else:
            overdue_dict[number] = len(df)
    
    return pd.Series(overdue_dict).sort_values(ascending=False)

def generate_statistical_numbers(game: str) -> List[int]:
    """
    Generates a balanced, deterministic set of lottery numbers for a specific game.

    This function is deterministic. For the same historical data, it will always
    produce the same output by seeding its random number generator based on the
    game and the number of draws.

    Args:
        game (str): The name of the lottery game.

    Returns:
        list: A sorted list of statistically chosen lottery numbers.
    """
    try:
        # Import locally to avoid top-level relative import issues
        from .data_harvester import load_and_validate_data

        rules = GAME_RULES[game]
        num_to_generate = rules["main_balls"]

        df = load_and_validate_data(game)

        # --- Create a deterministic seed ---
        # The seed is based on the game name and the number of draws.
        # This ensures that for the same data, the result is always the same.
        deterministic_seed = len(df) + sum(ord(c) for c in game)
        random.seed(deterministic_seed)

        frequency = analyze_number_frequency(df, game)
        # Convert to list to make it subscriptable for deterministic selection
        hot_numbers_pool = frequency.head(20).index.tolist()
        cold_numbers_pool = frequency.tail(20).index.tolist()

        overdue = analyze_overdue_numbers(df, game)
        overdue_numbers_pool = overdue.head(20).index.tolist()

        final_selection = set()

        # Use deterministic slicing instead of random.sample
        final_selection.update(hot_numbers_pool[:min(2, len(hot_numbers_pool))])

        cold_candidates = [n for n in cold_numbers_pool if n not in final_selection]
        final_selection.update(cold_candidates[:min(2, len(cold_candidates))])

        # Use a seeded shuffle and then slice to ensure deterministic selection
        while len(final_selection) < num_to_generate:
            overdue_candidates = [n for n in overdue_numbers_pool if n not in final_selection]
            if overdue_candidates:
                random.shuffle(overdue_candidates) # Shuffle is deterministic with seed
                final_selection.add(overdue_candidates[0])
            else:
                break

        # Fallback filler logic, also made deterministic
        combined_pool = list(set(hot_numbers_pool + cold_numbers_pool + overdue_numbers_pool))
        random.shuffle(combined_pool) # Shuffle is deterministic with seed

        filler_candidates = [n for n in combined_pool if n not in final_selection]

        fill_count = num_to_generate - len(final_selection)
        if fill_count > 0:
            final_selection.update(filler_candidates[:fill_count])

        return sorted(list(final_selection))[:num_to_generate]

    except (FileNotFoundError, ValueError, IndexError) as e:
        logging.error(f"Could not generate statistical numbers for '{game}': {e}")
        print(f"Warning: Statistical analysis for '{game}' failed. Falling back to random numbers.")
        # Ensure the fallback is also deterministic
        seed_fallback = sum(ord(c) for c in game)
        random.seed(seed_fallback)
        max_ball_fallback = GAME_RULES.get(game, {}).get("max_main_ball", 60)
        num_gen_fallback = GAME_RULES.get(game, {}).get("main_balls", 6)
        return sorted(random.sample(range(1, max_ball_fallback + 1), num_gen_fallback))

if __name__ == "__main__":
    # This block allows the script to be run directly for testing.
    # It adjusts the Python path to handle the relative imports correctly.
    import sys
    import os

    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Now that the path is set, we can perform the local imports for the test
    from src.data_harvester import load_and_validate_data
    
    print("--- Running Statistical Modeler Test for All Games ---")
    
    for game_name in GAME_RULES.keys():
        print(f"\n--- Generating Numbers for: {game_name.upper()} ---")
        try:
            numbers = generate_statistical_numbers(game_name)
            print(f"Statistically Generated Numbers: {numbers}")

            # Show some detailed analysis for verification
            df = load_and_validate_data(game_name)
            frequency_series = analyze_number_frequency(df, game_name)
            overdue_series = analyze_overdue_numbers(df, game_name)
            
            print(f"Top 5 Hot Numbers: {frequency_series.head(5).index.tolist()}")
            print(f"Top 5 Cold Numbers: {frequency_series.tail(5).index.tolist()}")
            print(f"Top 5 Overdue Numbers: {overdue_series.head(5).index.tolist()}")

        except Exception as e:
            print(f"An error occurred during testing for {game_name.upper()}: {e}")
# src/statistical_modeler.py
"""
This module contains the statistical modeling engine for generating lottery numbers.
It analyzes historical data for various games to calculate frequencies and other properties.
"""
import pandas as pd
import logging
import random
from typing import Dict, Any, List

# Import the new data pipeline function directly
from .data_harvester import get_merged_data

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

def _get_main_ball_columns(df: pd.DataFrame, game: str) -> List[str]:
    """Helper function to robustly identify main ball columns."""
    rules = GAME_RULES[game]
    num_main_balls = rules["main_balls"]
    
    # Find columns that look like main balls, excluding bonus balls
    main_ball_cols = [
        col for col in df.columns 
        if (col.lower().startswith('ball') or col.lower().startswith('n')) 
        and 'lucky' not in col.lower() 
        and 'star' not in col.lower() 
        and 'thunderball' not in col.lower() 
        and 'life' not in col.lower()
    ]
    
    # Take the first N columns that match
    ball_columns = main_ball_cols[:num_main_balls]
    
    if len(ball_columns) != num_main_balls:
        raise ValueError(f"Could not find the expected {num_main_balls} main ball columns for {game}. Found: {ball_columns} in {df.columns.tolist()}")
        
    return ball_columns

def analyze_number_frequency(df: pd.DataFrame, game: str) -> pd.Series:
    """Analyzes the frequency of each main ball number."""
    ball_columns = _get_main_ball_columns(df, game)
    all_balls = df[ball_columns].values.flatten()
    return pd.Series(all_balls).value_counts()

def analyze_overdue_numbers(df: pd.DataFrame, game: str) -> pd.Series:
    """Analyzes how many draws have passed since each main number last appeared."""
    rules = GAME_RULES[game]
    max_main_ball = rules["max_main_ball"]
    ball_columns = _get_main_ball_columns(df, game)
    
    all_possible_numbers = set(range(1, max_main_ball + 1))
    overdue_dict = {}

    for number in all_possible_numbers:
        # Check if the number is present in any of the ball columns for any row
        is_present = df[ball_columns].isin([number]).any(axis=1)
        if is_present.any():
            # idxmax() finds the index of the first occurrence (which is the most recent draw since data is sorted)
            last_seen_index = is_present.idxmax()
            overdue_dict[number] = last_seen_index
        else:
            # If the number never appeared, it's overdue by the total number of draws
            overdue_dict[number] = len(df)
    
    return pd.Series(overdue_dict).sort_values(ascending=False)

def generate_statistical_numbers(game: str) -> List[int]:
    """
    Generates a balanced, deterministic set of lottery numbers for a specific game
    using the merged historical and live data.
    """
    try:
        rules = GAME_RULES[game]
        num_to_generate = rules["main_balls"]

        # UPDATED: Use the merged data pipeline
        df = get_merged_data(game)
        if df is None:
            raise ValueError(f"Failed to get merged data for '{game}'.")

        # Create a deterministic seed based on game name and number of draws
        deterministic_seed = len(df) + sum(ord(c) for c in game)
        random.seed(deterministic_seed)

        frequency = analyze_number_frequency(df, game)
        hot_numbers_pool = frequency.head(20).index.tolist()
        cold_numbers_pool = frequency.tail(20).index.tolist()

        overdue = analyze_overdue_numbers(df, game)
        overdue_numbers_pool = overdue.head(20).index.tolist()

        final_selection = set()

        # Deterministic selection logic remains the same
        final_selection.update(hot_numbers_pool[:min(2, len(hot_numbers_pool))])
        cold_candidates = [n for n in cold_numbers_pool if n not in final_selection]
        final_selection.update(cold_candidates[:min(2, len(cold_candidates))])

        while len(final_selection) < num_to_generate:
            overdue_candidates = [n for n in overdue_numbers_pool if n not in final_selection]
            if overdue_candidates:
                random.shuffle(overdue_candidates) # Shuffle is deterministic with seed
                final_selection.add(overdue_candidates[0])
            else:
                break

        # Fallback filler logic
        combined_pool = list(set(hot_numbers_pool + cold_numbers_pool + overdue_numbers_pool))
        random.shuffle(combined_pool)
        filler_candidates = [n for n in combined_pool if n not in final_selection]
        fill_count = num_to_generate - len(final_selection)
        if fill_count > 0:
            final_selection.update(filler_candidates[:fill_count])

        return sorted(list(final_selection))[:num_to_generate]

    except (FileNotFoundError, ValueError, IndexError) as e:
        logging.error(f"Could not generate statistical numbers for '{game}': {e}")
        print(f"Warning: Statistical analysis for '{game}' failed. Falling back to random numbers.")
        seed_fallback = sum(ord(c) for c in game)
        random.seed(seed_fallback)
        max_ball_fallback = GAME_RULES.get(game, {}).get("max_main_ball", 60)
        num_gen_fallback = GAME_RULES.get(game, {}).get("main_balls", 6)
        return sorted(random.sample(range(1, max_ball_fallback + 1), num_gen_fallback))

if __name__ == "__main__":
    print("--- Running Statistical Modeler Test for All Games ---")
    
    for game_name in GAME_RULES.keys():
        print(f"\n--- Generating Numbers for: {game_name.upper()} ---")
        try:
            numbers = generate_statistical_numbers(game_name)
            print(f"Statistically Generated Numbers: {numbers}")

            # Show detailed analysis for verification using the merged data
            df = get_merged_data(game_name)
            if df is not None:
                frequency_series = analyze_number_frequency(df, game_name)
                overdue_series = analyze_overdue_numbers(df, game_name)
                
                print(f"Top 5 Hot Numbers: {frequency_series.head(5).index.tolist()}")
                print(f"Top 5 Cold Numbers: {frequency_series.tail(5).index.tolist()}")
                print(f"Top 5 Overdue Numbers: {overdue_series.head(5).index.tolist()}")
            else:
                print("Could not load merged data for detailed analysis.")

        except Exception as e:
            print(f"An error occurred during testing for {game_name.upper()}: {e}")
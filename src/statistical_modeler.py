# src/statistical_modeler.py
"""
This module contains the statistical modeling engine for generating lottery numbers.
It analyzes historical data for frequency, overdue numbers, and other properties
for both main and bonus number pools.
"""
import pandas as pd
import logging
import random
from typing import Dict, Any, List, Optional

# V2.0 IMPORTS: Centralized game rules and data harvester
from src.data_harvester import get_merged_data
from src.game_configs import get_game_rules, GAME_RULES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _analyze_pool_frequency(df: pd.DataFrame, columns: List[str]) -> Optional[pd.Series]:
    """Analyzes the frequency of numbers within a specific pool (main or bonus)."""
    if not columns or any(col not in df.columns for col in columns):
        logging.warning(f"Frequency analysis skipped: one or more columns not found. Looked for: {columns}")
        return None
    
    # Ensure all data is numeric, coercing errors to NaN and then dropping them
    pool_values = df[columns].stack().dropna()
    pool_values = pd.to_numeric(pool_values, errors='coerce').dropna()
    
    if pool_values.empty:
        return None
        
    return pool_values.value_counts()

def _analyze_pool_overdue(df: pd.DataFrame, columns: List[str], min_val: int, max_val: int) -> Optional[pd.Series]:
    """Analyzes how many draws have passed since each number in a pool last appeared."""
    if not columns or any(col not in df.columns for col in columns):
        logging.warning(f"Overdue analysis skipped: one or more columns not found. Looked for: {columns}")
        return None
        
    all_possible_numbers = set(range(min_val, max_val + 1))
    overdue_dict = {}

    for number in all_possible_numbers:
        # Check if the number is present in any of the pool columns for any row
        is_present = df[columns].isin([number]).any(axis=1)
        if is_present.any():
            # idxmax() finds the index of the first occurrence (most recent draw)
            last_seen_index = is_present.idxmax()
            overdue_dict[number] = last_seen_index
        else:
            # If the number never appeared, it's overdue by the total number of draws
            overdue_dict[number] = len(df)
            
    if not overdue_dict:
        return None

    return pd.Series(overdue_dict).sort_values(ascending=False)

def _generate_pool_numbers(df: pd.DataFrame, rules: Dict[str, Any], seed: int) -> List[int]:
    """Generates a set of numbers for a single pool (main or bonus)."""
    num_to_generate = rules['count']
    if num_to_generate == 0:
        return []

    # Use a pool-specific seed for deterministic but distinct results
    random.seed(seed)

    frequency = _analyze_pool_frequency(df, rules['columns'])
    overdue = _analyze_pool_overdue(df, rules['columns'], rules['min'], rules['max'])

    if frequency is None or overdue is None:
        logging.warning(f"Falling back to random sampling for pool due to analysis failure. Columns: {rules['columns']}")
        return sorted(random.sample(range(rules['min'], rules['max'] + 1), num_to_generate))

    hot_numbers = frequency.head(10).index.tolist()
    cold_numbers = frequency.tail(10).index.tolist()
    overdue_numbers = overdue.head(10).index.tolist()

    final_selection = set()

    # Deterministic selection logic
    final_selection.update(hot_numbers[:min(1, len(hot_numbers))])
    cold_candidates = [n for n in cold_numbers if n not in final_selection]
    final_selection.update(cold_candidates[:min(1, len(cold_candidates))])

    while len(final_selection) < num_to_generate:
        overdue_candidates = [n for n in overdue_numbers if n not in final_selection]
        if overdue_candidates:
            random.shuffle(overdue_candidates)
            final_selection.add(overdue_candidates[0])
        else:
            break
    
    # Fallback filler
    combined_pool = list(set(hot_numbers + cold_numbers + overdue_numbers))
    random.shuffle(combined_pool)
    filler_candidates = [n for n in combined_pool if n not in final_selection]
    fill_count = num_to_generate - len(final_selection)
    if fill_count > 0:
        final_selection.update(filler_candidates[:fill_count])

    return sorted([int(n) for n in final_selection])[:num_to_generate]


def generate_statistical_numbers(game: str) -> Dict[str, List[int]]:
    """
    Generates statistically-derived lottery numbers for both main and bonus pools.
    """
    logging.info(f"--- Generating statistical numbers for {game.upper()} (v2.0) ---")
    try:
        rules = get_game_rules(game)
        df = get_merged_data(game)
        
        if df is None or df.empty:
            raise ValueError(f"No valid data available for '{game}' to perform analysis.")

        # Create a deterministic seed based on game name and data state
        base_seed = len(df) + sum(ord(c) for c in game)
        
        # Generate numbers for the main pool
        main_numbers = _generate_pool_numbers(df, rules['main'], seed=base_seed)
        
        # Generate numbers for the bonus pool (with a different seed)
        bonus_numbers = _generate_pool_numbers(df, rules['bonus'], seed=base_seed + 1)

        return {"main": main_numbers, "bonus": bonus_numbers}

    except (ValueError, KeyError) as e:
        logging.error(f"Could not generate statistical numbers for '{game}': {e}", exc_info=True)
        print(f"Warning: Statistical analysis for '{game}' failed. Falling back to random numbers.")
        
        # Fallback to pure random generation if analysis fails
        try:
            rules = get_game_rules(game)
            main_rules = rules['main']
            bonus_rules = rules['bonus']
            
            main_fallback = sorted(random.sample(range(main_rules['min'], main_rules['max'] + 1), main_rules['count']))
            bonus_fallback = []
            if bonus_rules['count'] > 0:
                bonus_fallback = sorted(random.sample(range(bonus_rules['min'], bonus_rules['max'] + 1), bonus_rules['count']))
            
            return {"main": main_fallback, "bonus": bonus_fallback}
        except Exception as fallback_e:
            logging.critical(f"CRITICAL: Fallback random generation also failed for '{game}': {fallback_e}")
            return {"main": [], "bonus": []}


if __name__ == "__main__":
    print("--- Running Statistical Modeler Test (v2.0) for All Games ---")
    
    for game_name in GAME_RULES.keys():
        print(f"\n--- Generating Numbers for: {game_name.upper()} ---")
        try:
            numbers = generate_statistical_numbers(game_name)
            print(f"Statistically Generated Main Numbers: {numbers.get('main', 'N/A')}")
            if numbers.get('bonus'):
                print(f"Statistically Generated Bonus Numbers: {numbers.get('bonus')}")
            
            # Optional: Display detailed analysis for verification
            df = get_merged_data(game_name)
            if df is not None:
                game_rules = get_game_rules(game_name)
                main_freq = _analyze_pool_frequency(df, game_rules['main']['columns'])
                if main_freq is not None:
                    print(f"  - Top 5 Hot Main Numbers: {main_freq.head(5).index.tolist()}")
                    print(f"  - Top 5 Cold Main Numbers: {main_freq.tail(5).index.tolist()}")
                
                if game_rules['bonus']['count'] > 0:
                    bonus_freq = _analyze_pool_frequency(df, game_rules['bonus']['columns'])
                    if bonus_freq is not None:
                        print(f"  - Top 3 Hot Bonus Numbers: {bonus_freq.head(3).index.tolist()}")

        except Exception as e:
            print(f"An error occurred during testing for {game_name.upper()}: {e}")
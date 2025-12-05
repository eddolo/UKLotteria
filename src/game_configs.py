# src/game_configs.py

"""
Centralized configuration for all lottery game rules.

This module defines the structural rules for each lottery game, including the number
of balls, their ranges, and the column names used in the data files. This provides
a single source of truth for the entire application.
"""

from typing import Dict, Any, List

GAME_RULES: Dict[str, Dict[str, Any]] = {
    "lotto": {
        "main": {
            "count": 6,
            "min": 1,
            "max": 59,
            "columns": [f"Ball_{i}" for i in range(1, 7)]
        },
        "bonus": {
            "count": 0,
            "min": 0,
            "max": 0,
            "columns": []
        }
    },
    "euromillions": {
        "main": {
            "count": 5,
            "min": 1,
            "max": 50,
            "columns": [f"Ball_{i}" for i in range(1, 6)]
        },
        "bonus": {
            "count": 2,
            "min": 1,
            "max": 12,
            "columns": ["Lucky_Star_1", "Lucky_Star_2"]
        }
    },
    "thunderball": {
        "main": {
            "count": 5,
            "min": 1,
            "max": 39,
            "columns": [f"Ball_{i}" for i in range(1, 6)]
        },
        "bonus": {
            "count": 1,
            "min": 1,
            "max": 14,
            "columns": ["Thunderball"]
        }
    },
    "setforlife": {
        "main": {
            "count": 5,
            "min": 1,
            "max": 47,
            "columns": [f"Ball_{i}" for i in range(1, 6)]
        },
        "bonus": {
            "count": 1,
            "min": 1,
            "max": 10,
            "columns": ["Life_Ball"]
        }
    }
}

def get_game_rules(game_name: str) -> Dict[str, Any]:
    """
    Retrieves the rules for a specific game.

    Args:
        game_name (str): The name of the lottery game.

    Returns:
        Dict[str, Any]: The rules for the specified game.

    Raises:
        ValueError: If the game_name is not found in GAME_RULES.
    """
    if game_name not in GAME_RULES:
        raise ValueError(f"Game '{game_name}' not supported. Please choose from {list(GAME_RULES.keys())}")
    return GAME_RULES[game_name]
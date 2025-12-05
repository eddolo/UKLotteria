# app.py
"""
Streamlit Web App for the UK Lottery Number Generator (v2.0).

This version separates the user interface from the model training process.
The app is now responsible for displaying predictions from pre-trained models,
not for training them on-demand.
"""

import sys
import os
import streamlit as st
import pandas as pd
import time
import logging

# --- App Configuration ---
# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - APP - %(levelname)s - %(message)s'
)

st.set_page_config(
    page_title="UK Lottery Number Generator v2.0",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="auto",
)

logging.info("--- Streamlit App Script START (v2.0) ---")

# --- Import project modules (v2.0 Imports) ---
try:
    from src.data_harvester import fetch_live_lottery_data, get_data_path, get_merged_data
    from src.game_configs import get_game_rules, GAME_RULES
    from src.statistical_modeler import generate_statistical_numbers
    # CORRECTED IMPORT: Removed train_models_for_game, will use generate_ml_numbers directly.
    from src.ml_modeler import generate_ml_numbers, get_model_path
    logging.info("--- All project modules imported successfully! ---")
except ImportError as e:
    # Use a more descriptive error message and log the full traceback
    logging.critical(f"--- FAILED to import project modules. Error: {e} ---", exc_info=True)
    st.error(f"A critical error occurred during application startup: {e}. The application cannot continue. Please check the console logs for a full traceback.")
    st.stop()

SUPPORTED_GAMES = list(GAME_RULES.keys())

# --- State Initialization ---
if 'generate' not in st.session_state:
    st.session_state.generate = False
if 'game' not in st.session_state:
    st.session_state.game = SUPPORTED_GAMES[0]
if 'method' not in st.session_state:
    st.session_state.method = 'Statistical'

# --- Backend Logic for Streamlit (v2.0 REFACTORED) ---

@st.cache_data(show_spinner=False)
def ensure_data_exists(game: str) -> bool:
    """Checks for data and downloads if missing."""
    logging.info(f"Checking for data for game: {game}")
    data_path = get_data_path(game, data_type='base')
    if not os.path.exists(data_path):
        st.info(f"Historical data for '{game.upper()}' not found. Attempting download...")
        if fetch_live_lottery_data(game):
            st.success(f"Data for '{game.upper()}' downloaded successfully!")
            # Short delay and rerun to ensure UI updates with new data context
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"Failed to download data for '{game.upper()}'. App may not function correctly.")
            return False
    return True

def handle_ml_generation_streamlit(game: str):
    """
    Handles ML number generation by using pre-trained models.
    If models are not found, it displays an error instead of training.
    """
    logging.info(f"Handling ML number generation for {game}.")
    main_model_path = get_model_path(game, 'main')
    
    # CRITICAL CHANGE: Check if the model exists. Do NOT train.
    if not os.path.exists(main_model_path):
        st.error(
            f"ML model for '{game.upper()}' not found. Please train the model first.\n\n"
            f"Run this command in your terminal from the project directory:\n\n"
            f"`python train_models.py --game {game}`"
        )
        logging.error(f"ML model for {game} not found. Aborting generation.")
        return None
    
    # If model exists, generate numbers
    logging.info(f"Existing ML model found for {game}. Generating numbers.")
    with st.spinner(f"Loading trained '{game.upper()}' model(s) and generating numbers..."):
        numbers = generate_ml_numbers(game)
        if numbers is None:
            st.error(f"Failed to generate numbers. The model may be corrupt or the data may be incompatible.")
            logging.error("The 'generate_ml_numbers' function returned None.")
    return numbers

@st.cache_data(show_spinner=False)
def get_number_frequencies(_game: str) -> pd.Series:
    """Calculates and caches frequency for main numbers."""
    logging.info(f"Calculating number frequencies for {_game}.")
    try:
        df = get_merged_data(_game)
        if df is None or df.empty: return None
        game_rules = get_game_rules(_game)
        main_ball_cols = game_rules['main']['columns']
        all_numbers = df[main_ball_cols].values.flatten()
        return pd.Series(all_numbers[~pd.isna(all_numbers)]).value_counts().sort_index()
    except Exception as e:
        logging.error(f"Error calculating frequencies for {_game}: {e}")
        return None

# --- UI Rendering ---

st.title("🔮 UK Lottery Number Generator v2.0")

# --- Sidebar ---
st.sidebar.header("Game Selection")
selected_game = st.sidebar.selectbox(
    "Choose a lottery game:",
    SUPPORTED_GAMES,
    index=SUPPORTED_GAMES.index(st.session_state.game)
)

# If the game selection changes, clear caches and rerun the script
if selected_game != st.session_state.game:
    logging.info(f"Game changed from '{st.session_state.game}' to '{selected_game}'.")
    st.session_state.game = selected_game
    st.cache_data.clear() # Clear all cached data
    st.rerun()

# Ensure data exists for the selected game before proceeding
if not ensure_data_exists(st.session_state.game):
    st.warning("Data is not available for the selected game. Please try again later.")
    st.stop()

st.sidebar.markdown("---")
if st.sidebar.button(f"Force Update for '{st.session_state.game.upper()}'"):
    with st.spinner(f"Fetching latest data..."):
        if fetch_live_lottery_data(st.session_state.game):
            st.sidebar.success("Data updated!")
            st.cache_data.clear()
            time.sleep(1) # Brief pause for user to see the message
            st.rerun()
        else:
            st.sidebar.error("Data update failed.")

st.sidebar.markdown("---")
st.sidebar.header("Generate Numbers")
st.session_state.method = st.sidebar.radio(
    "Choose prediction method:",
    ('Statistical', 'Machine Learning'),
    index=('Statistical', 'Machine Learning').index(st.session_state.method)
)

if st.sidebar.button(f"Generate for '{st.session_state.game.upper()}'"):
    st.session_state.generate = True

# --- Main Content Area (v2.0) ---
if st.session_state.generate:
    active_game = st.session_state.game
    active_method = st.session_state.method
    st.info(f"Generating numbers for **{active_game.upper()}** using the **{active_method}** model...")
    
    numbers = None
    if active_method == 'Statistical':
        numbers = generate_statistical_numbers(active_game)
    elif active_method == 'Machine Learning':
        numbers = handle_ml_generation_streamlit(active_game)

    # v2.0: Updated display logic for multi-pool results
    if numbers and isinstance(numbers, dict):
        game_rules = get_game_rules(active_game)
        main_label = "Main Numbers"
        bonus_label = game_rules.get('bonus', {}).get('name', 'Bonus Numbers')
        
        st.subheader(f"Generated Prediction for {active_game.upper()}")
        
        main_numbers = numbers.get('main')
        bonus_numbers = numbers.get('bonus')

        # Display numbers in columns
        col1, col2 = st.columns(2)
        if main_numbers:
            with col1:
                st.success(f"**{main_label}:** `{', '.join(map(str, sorted(main_numbers)))}`")
        
        if bonus_numbers:
            with col2:
                st.warning(f"**{bonus_label}:** `{', '.join(map(str, sorted(bonus_numbers)))}`")

    elif numbers is not None: # Case where generation failed but didn't return None (e.g. statistical fails)
        st.error(f"Failed to generate numbers for **{active_game.upper()}**.")

    st.session_state.generate = False # Reset flag

st.markdown("---")

# --- Info/Viz Panes (v2.0) ---
st.header(f"Data Dashboard: {st.session_state.game.upper()}")
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("#### 📊 Model Information")
    df = get_merged_data(st.session_state.game)
    st.markdown(f"**Training Data:** `{len(df) if df is not None else 'N/A'}` historical draws.")
    
    # CORRECTED: Changed 'model_type' to 'pool_type' to match function definition
    main_model_path = get_model_path(st.session_state.game, pool_type='main')
    if os.path.exists(main_model_path):
        st.markdown(f"**ML Model Last Trained:** `{time.ctime(os.path.getmtime(main_model_path))}`")
    else:
        st.markdown("**ML Model Status:** `Not yet trained.`")

with col2:
    st.markdown("#### 📈 Historical Number Frequency (Main Numbers)")
    frequencies = get_number_frequencies(st.session_state.game)
    if frequencies is not None and not frequencies.empty:
        st.bar_chart(frequencies)
    else:
        st.warning("Could not display frequency chart.")

st.sidebar.markdown("---")
st.sidebar.info("Developed by ReCurse AI (v2.0)")

logging.info("--- Streamlit App Script END ---")
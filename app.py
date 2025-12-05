# app.py
"""
Streamlit Web App for the UK Lottery Number Generator.
"""

import streamlit as st
import os
import pandas as pd
import time
import logging

# --- App Configuration ---
# Configure logging to be visible in the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - APP - %(levelname)s - %(message)s'
)

st.set_page_config(
    page_title="UK Lottery Number Generator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="auto",
)

logging.info("--- Streamlit App Script START ---")

# --- Import project modules AFTER basic config ---
from src.data_harvester import fetch_live_lottery_data, get_data_path, get_merged_data, DATA_URLS
from src.statistical_modeler import generate_statistical_numbers, GAME_RULES
from src.ml_modeler import (
    train_and_generate_ml_numbers,
    generate_ml_numbers as generate_from_existing_model,
    get_model_path
)

SUPPORTED_GAMES = list(GAME_RULES.keys())

# --- State Initialization ---
logging.info("Initializing session state...")
if 'generate' not in st.session_state:
    st.session_state.generate = False
if 'game' not in st.session_state:
    st.session_state.game = SUPPORTED_GAMES[0]
if 'method' not in st.session_state:
    st.session_state.method = 'Statistical'
logging.info(f"Initial state: game='{st.session_state.game}', method='{st.session_state.method}'")

# --- Backend Logic for Streamlit ---

@st.cache_data(show_spinner=False)
def ensure_data_exists(game: str):
    logging.info(f"Checking for data for game: {game}")
    data_path = get_data_path(game, data_type='live')
    if not os.path.exists(data_path):
        st.info(f"Live data for '{game.upper()}' not found. Attempting download...")
        logging.info(f"Downloading data for {game}...")
        if fetch_live_lottery_data(game):
            st.success(f"Live data for '{game.upper()}' downloaded successfully!")
            logging.info(f"Data for {game} downloaded.")
            time.sleep(1)
            st.rerun() # Rerun to refresh the UI now that data exists
        else:
            st.error(f"Failed to download data for '{game.upper()}'. App may not function correctly.")
            logging.error(f"Failed to download data for {game}.")
            return False
    logging.info(f"Data for {game} exists.")
    return True

def handle_ml_generation_streamlit(game: str):
    logging.info(f"Handling ML number generation for {game}.")
    model_path = get_model_path(game)
    if not os.path.exists(model_path):
        st.warning(f"ML model for '{game.upper()}' not found. Training now...")
        logging.info(f"ML model for {game} not found. Starting training.")
        with st.spinner('Training the neural network... Please be patient.'):
            numbers = train_and_generate_ml_numbers(game, epochs=50)
        if numbers:
            st.success("Model training complete!")
            logging.info("Model training successful.")
        else:
            st.error(f"Model training failed for '{game.upper()}'.")
            logging.error("Model training failed.")
            return None
    else:
        logging.info(f"Existing ML model found for {game}. Generating numbers.")
        with st.spinner(f"Loading trained '{game.upper()}' model..."):
            numbers = generate_from_existing_model(game)
            if numbers is None:
                st.error(f"Failed to generate numbers using the existing model.")
                logging.error("Failed to generate numbers from existing model.")
    return numbers

@st.cache_data(show_spinner=False)
def get_number_frequencies(_game: str):
    logging.info(f"Calculating number frequencies for {_game}.")
    try:
        df = get_merged_data(_game)
        if df is None: return None
        rules = GAME_RULES[_game]
        ball_columns = [col for col in df.columns if col.lower().startswith(('ball', 'n'))][:rules["main_balls"]]
        if not ball_columns: return None
        all_numbers = df[ball_columns].values.flatten()
        return pd.Series(all_numbers[~pd.isna(all_numbers)]).value_counts().sort_index()
    except Exception as e:
        logging.error(f"Error calculating frequencies: {e}")
        return None

# --- UI Rendering ---

st.title("🔮 UK Lottery Number Generator")
logging.info("UI: Title rendered.")

# --- Sidebar ---
logging.info("UI: Rendering sidebar.")
st.sidebar.header("Game Selection")
selected_game = st.sidebar.selectbox("Choose a lottery game:", SUPPORTED_GAMES, index=SUPPORTED_GAMES.index(st.session_state.game))

if selected_game != st.session_state.game:
    logging.info(f"Game changed from '{st.session_state.game}' to '{selected_game}'. Rerunning.")
    st.session_state.game = selected_game
    st.cache_data.clear()
    st.rerun()

logging.info("UI: Calling ensure_data_exists.")
if not ensure_data_exists(st.session_state.game):
    st.warning("Data is not available. Please try again later.")
    logging.warning("ensure_data_exists returned False. Stopping script execution.")
    st.stop()
logging.info("UI: ensure_data_exists check passed.")

st.sidebar.markdown("---")
if st.sidebar.button(f"Force Update for '{st.session_state.game.upper()}'"):
    logging.info("UI: 'Force Update' button clicked.")
    with st.spinner(f"Fetching latest data..."):
        if fetch_live_lottery_data(st.session_state.game):
            st.sidebar.success("Data updated!")
            st.cache_data.clear()
            time.sleep(1)
            logging.info("UI: Rerunning after successful update.")
            st.rerun()
        else:
            st.sidebar.error("Data update failed.")

st.sidebar.markdown("---")
st.sidebar.header("Generate Numbers")
st.session_state.method = st.sidebar.radio("Choose your generation method:", ('Statistical', 'Machine Learning'), index=('Statistical', 'Machine Learning').index(st.session_state.method))

if st.sidebar.button(f"Generate for '{st.session_state.game.upper()}'"):
    logging.info(f"UI: 'Generate' button clicked for game '{st.session_state.game}' with method '{st.session_state.method}'.")
    st.session_state.generate = True

# --- Main Content Area ---
logging.info("UI: Checking if main content should be generated.")
if st.session_state.generate:
    logging.info("UI: Generating main content.")
    active_game = st.session_state.game
    active_method = st.session_state.method
    st.info(f"Generating numbers for **{active_game.upper()}** using the **{active_method}** model...")
    numbers = None
    if active_method == 'Statistical':
        numbers = generate_statistical_numbers(active_game)
    elif active_method == 'Machine Learning':
        numbers = handle_ml_generation_streamlit(active_game)

    if numbers and isinstance(numbers, dict) and 'main' in numbers:
        main_nums = ", ".join(map(str, numbers['main']))
        special_nums = ", ".join(map(str, numbers.get('special', [])))
        st.success(f"**Main Numbers:** `{main_nums}`")
        if special_nums:
            st.success(f"**Special Numbers:** `{special_nums}`")
    else:
        st.error(f"Failed to generate numbers for **{active_game.upper()}**.")

    st.session_state.generate = False # Reset flag
    logging.info("UI: Main content generation complete.")

st.markdown("---")

# --- Info/Viz Panes ---
logging.info("UI: Rendering info and visualization panes.")
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown(f"#### 📊 Model Information: {st.session_state.game.upper()}")
    df = get_merged_data(st.session_state.game)
    st.markdown(f"**Training Data:** `{len(df) if df is not None else 'N/A'}` historical draws.")
    model_path = get_model_path(st.session_state.game)
    if os.path.exists(model_path):
        st.markdown(f"**ML Model Last Trained:** `{time.ctime(os.path.getmtime(model_path))}`")
    else:
        st.markdown("**ML Model Status:** `Not yet trained.`")

with col2:
    st.markdown(f"#### 📈 Historical Number Frequency ({st.session_state.game.upper()})")
    frequencies = get_number_frequencies(st.session_state.game)
    if frequencies is not None and not frequencies.empty:
        st.bar_chart(frequencies)
    else:
        st.warning("Could not display frequency chart.")
logging.info("UI: Info and visualization panes rendered.")

st.sidebar.markdown("---")
st.sidebar.info("Developed by ReCurse AI.")
logging.info("--- Streamlit App Script END ---")
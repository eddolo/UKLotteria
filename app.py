# app.py
"""
Streamlit Web App for the UK Lottery Number Generator.

This user-friendly, browser-based interface allows users to:
- Select a specific UK lottery game.
- Update the historical lottery data for that game.
- Choose between "Statistical" and "Machine Learning" models.
- View the generated lottery numbers.
- See basic data visualizations for the selected game.
"""

import streamlit as st
import os
import pandas as pd
import time

# Import game-aware components
from src.data_harvester import fetch_and_save_lottery_data, get_data_path, DATA_URLS
from src.statistical_modeler import generate_statistical_numbers, GAME_RULES
from src.ml_modeler import (
    train_and_generate_ml_numbers,
    generate_ml_numbers as generate_from_existing_model,
    get_model_path,
    get_scaler_path,
    load_and_validate_data
)

# --- App Configuration ---
st.set_page_config(
    page_title="UK Lottery Number Generator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Game & Model Information ---
SUPPORTED_GAMES = list(GAME_RULES.keys())

# --- Backend Logic for Streamlit ---

def ensure_data_exists(game: str):
    """
    Checks for a game's data file and downloads it if missing.
    Returns True if data is present or was successfully downloaded.
    """
    data_path = get_data_path(game)
    if not os.path.exists(data_path):
        st.info(f"Data for '{game.upper()}' not found. Downloading automatically...")
        with st.spinner(f"Fetching latest '{game.upper()}' data from source..."):
            if fetch_and_save_lottery_data(game):
                st.success(f"Data for '{game.upper()}' downloaded!")
                time.sleep(2)
                st.rerun()
                return True
            else:
                st.error("Automatic data download failed. Check connection or try the 'Update Data' button.")
                return False
    return True

# --- Caching Note ---
# Caching is more complex with parameters. For simplicity during this refactor,
# we will call functions directly. For performance optimization, st.cache_data
# would be used like: @st.cache_data
# def get_statistical_numbers(game: str): ...

def handle_ml_generation_streamlit(game: str):
    """Handles the ML generation logic for a specific game within Streamlit."""
    model_path = get_model_path(game)
    scaler_path = get_scaler_path(game)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.warning(f"ML model for '{game.upper()}' not found. A one-time training process will now begin. This may take several minutes.")
        
        progress_bar = st.progress(0, text=f"Starting training for '{game.upper()}'...")
        
        with st.spinner('Training the neural network... Please be patient.'):
            # This function now handles the whole process internally
            numbers = train_and_generate_ml_numbers(game, epochs=50)

        if numbers is None:
            st.error(f"Model training failed for '{game.upper()}'. Please check the logs.")
            return None
        
        progress_bar.progress(100, text="Training complete!")
        st.success("Model training complete! Numbers generated from the new model.")
        time.sleep(2)
        progress_bar.empty()
    
    else:
        with st.spinner(f"Loading trained '{game.upper()}' model and generating prediction..."):
            numbers = generate_from_existing_model(game)
            if numbers is None:
                st.error(f"Failed to generate numbers using the existing '{game.upper()}' model.")
                return None
    
    return numbers

@st.cache_data
def get_number_frequencies(game: str):
    """Loads data and calculates number frequencies for a specific game."""
    try:
        df = load_and_validate_data(game)
        if df is None:
            return None
            
        rules = GAME_RULES[game]
        num_balls = rules["main_balls"]
        ball_columns = [col for col in df.columns if 'ball' in col and 'lucky' not in col and 'thunderball' not in col and 'life' not in col][:num_balls]

        all_numbers = df[ball_columns].values.flatten()
        frequencies = pd.Series(all_numbers).value_counts().sort_index()
        return frequencies
    except Exception as e:
        st.error(f"Failed to calculate frequencies for '{game}': {e}")
        return None

# --- Main App UI ---

def main():
    """Renders the Streamlit user interface."""
    st.title("🔮 UK Lottery Number Generator")

    # --- Sidebar Controls ---
    st.sidebar.header("Game Selection")
    game = st.sidebar.selectbox("Choose a lottery game:", SUPPORTED_GAMES, key="game_selection")

    if not ensure_data_exists(game):
        st.stop()

    st.sidebar.markdown("---")

    if st.sidebar.button(f"Force Update for '{game.upper()}'"):
        with st.spinner(f"Fetching latest '{game.upper()}' data..."):
            if fetch_and_save_lottery_data(game):
                st.sidebar.success("Data updated!")
                st.cache_data.clear() # Clear all cached data
                time.sleep(1)
                st.rerun()
            else:
                st.sidebar.error("Data update failed.")

    st.sidebar.markdown("---")
    st.sidebar.header("Generate Numbers")
    method = st.sidebar.radio(
        "Choose your generation method:",
        ('Statistical', 'Machine Learning'),
        key="method_selection"
    )

    if st.sidebar.button(f"Generate for '{game.upper()}'"):
        st.session_state.generate = True
        st.session_state.game = game
        st.session_state.method = method
        st.session_state.numbers = None # Reset previous numbers

    # --- Main Content Area ---
    
    # Display generated numbers if the button was pressed
    if 'generate' in st.session_state and st.session_state.generate:
        active_game = st.session_state.game
        active_method = st.session_state.method
        
        st.info(f"Generating numbers for **{active_game.upper()}** using the **{active_method}** model...")

        if active_method == 'Statistical':
            numbers = generate_statistical_numbers(active_game)
        elif active_method == 'Machine Learning':
            numbers = handle_ml_generation_streamlit(active_game)
        
        if numbers:
            st.success(f"**Generated Numbers ({active_method} for {active_game.upper()}):** `{numbers}`")
        else:
            st.error(f"Failed to generate numbers for **{active_game.upper()}** using the **{active_method}** method.")
        
        # Reset state to prevent re-generation on page interaction
        st.session_state.generate = False


    st.markdown("---")

    # --- Information and Visualization Panes ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"#### 📊 Model Information: {game.upper()}")
        st.markdown(f"**Data Source:**")
        st.code(DATA_URLS[game], language=None)
        
        data_path = get_data_path(game)
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            st.markdown(f"**Training Data:** `{len(df)}` historical draws.")

        model_path = get_model_path(game)
        if os.path.exists(model_path):
            model_time = os.path.getmtime(model_path)
            st.markdown(f"**ML Model Last Trained:** `{time.ctime(model_time)}`")
        else:
            st.markdown("**ML Model Status:** `Not yet trained.`")

    with col2:
        st.markdown(f"#### 📈 Historical Number Frequency ({game.upper()})")
        frequencies = get_number_frequencies(game)
        if frequencies is not None:
            st.bar_chart(frequencies)
            max_ball = GAME_RULES[game]["max_main_ball"]
            st.caption(f"This chart shows how many times each number (1-{max_ball}) has been drawn.")
        else:
            st.warning("Could not display frequency chart. Data may be missing or invalid.")
            
    st.sidebar.markdown("---")
    st.sidebar.info("Developed by ReCurse AI.")


if __name__ == "__main__":
    main()
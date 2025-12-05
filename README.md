# 🔮 UK Lottery Number Generator

This project is an AI-driven application for generating UK National Lottery numbers using both statistical analysis and machine learning. It provides two distinct models and two user interfaces (a CLI and a web app) to interact with them.

## Core Features

*   **Automated Data Harvesting**: Automatically fetches the latest historical lottery results from a public data source.
*   **Statistical Modeler**: A deterministic engine that analyzes historical data to find number frequencies ("hot" and "cold" numbers), overdue numbers, and other statistical properties to generate a set of numbers.
*   **Machine Learning Modeler**: An advanced engine using an LSTM (Long Short-Term Memory) neural network to learn complex patterns from the sequence of past draws. The model's behavior is deterministic for a given dataset.
*   **Command-Line Interface (CLI)**: A fast and simple tool for power users to update data and generate numbers directly from the terminal.
*   **Streamlit Web App**: A user-friendly, browser-based interface to visualize data, update results, and generate numbers with the click of a button.

## Project Structure

```
.
├── app.py                  # The Streamlit Web App entry point
├── data/
│   └── lottery_history.csv # Stores the historical lottery data
├── lotto.py                # The Command-Line Interface (CLI) entry point
├── models/
│   └── lstm_model.h5       # The trained Machine Learning model
├── README.md               # This README file
├── requirements.txt        # Project dependencies (to be generated)
└── src/
    ├── __init__.py
    ├── data_harvester.py     # Module for fetching and saving lottery data
    ├── ml_modeler.py         # Module for the ML prediction model
    └── statistical_modeler.py# Module for the statistical prediction model
```

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    *(Note: A `requirements.txt` file will be generated as the final step of this project. Once available, run the following command.)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Initial Data:**
    Before running any generator, you must first download the historical data.
    ```bash
    python lotto.py --update-data
    ```

## Usage

You can interact with the project through the CLI or the Web App.

### 1. Command-Line Interface (`lotto.py`)

The CLI is the quickest way to get numbers.

*   **Update Historical Data:**
    ```bash
    python lotto.py --update-data
    ```

*   **Generate Numbers using the Statistical Model:**
    ```bash
    python lotto.py --method statistical
    ```

*   **Generate Numbers using the Machine Learning Model:**
    *(Note: The first time you run this, it will automatically train the model, which may take several minutes.)*
    ```bash
    python lotto.py --method ml
    ```

### 2. Streamlit Web App (`app.py`)

The web app provides a rich, interactive experience.

1.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your browser:**
    Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Use the Interface:**
    *   Use the **"Update Lottery Data"** button in the sidebar to fetch the latest results.
    *   Select either the **"Statistical"** or **"Machine Learning"** method.
    *   Click **"Generate Numbers"** to see the results.

    *   View the historical number frequency chart to gain insights into the data.

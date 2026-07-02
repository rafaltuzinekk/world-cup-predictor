# 🏆 World Cup Predictor: Advanced ELO & Machine Learning Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Engineering-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This repository contains a comprehensive, end-to-end Data Science pipeline designed to predict the outcomes of the FIFA World Cup. The project demonstrates how to handle "Scope Expansion" in data analytics by approaching the tournament from multiple analytical angles, blending over 150 years of historical match data with real-time tournament statistics.

The architecture is divided into **Three Core Modules** to handle both micro-level match predictions and macro-level tournament simulations.

## 🏗️ Architecture & Modules

### Moduł 1: Data Foundation (The Engine)
**Scripts:** `01_data_cleaning.ipynb` ➔ `03_fbref_scraping.ipynb`
* **Historical Data:** Ingestion and standardization of Kaggle datasets covering over 40,000 international matches.
* **Custom Elo Rating:** A mathematical engine built from scratch to calculate the absolute strength of national teams up to the current day.
* **Live Scraping:** Bypassing anti-bot protections to extract real-time group stage data (Possession, Goals) directly from FBref.

### Moduł 2: Micro Analysis (Machine Learning)
**Scripts:** `04_feature_engineering.ipynb` ➔ `06_visualizations.ipynb`
* **Objective:** Predict specific upcoming knockout matches based on current form.
* **Methodology:** A **Random Forest Classifier** is trained to evaluate "differentials" (Elo difference vs. Possession/Goal difference). The model applies a "Form Index" to adjust the raw historical probabilities based on current on-pitch dominance.
* **Output:** Visualized probabilities of advancement for selected highly-anticipated matches.

### Moduł 3: Macro Analysis (Tournament Simulations)
**Scripts:** `07_bracket_simulator.ipynb` ➔ `08_deterministic_bracket.ipynb`
* **Objective:** Simulate the entire 32-team knockout stage to find the ultimate champion.
* **Stochastic Approach (Monte Carlo):** Running 10,000 parallel realities using the Elo probabilities to account for variance, luck, and bracket difficulty.
* **Deterministic Approach:** A strict, math-based progression finding the single most likely path to the final, injecting actual real-world results to keep the bracket accurate.

## 📊 Key Results & Artifacts
The raw outputs, including visual charts (`.png`), step-by-step bracket logs (`.txt`), and full statistical reports (`.csv`), can be found in the `/data/processed/` directory. 

## 🚀 How to Run Locally
1. Clone this repository.
2. Set up a local Python environment and install the standard data stack (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).
3. Run the notebooks sequentially from `01` to `08` to completely rebuild the database, train the ML model, and run new simulations.

---
*Created as a comprehensive portfolio project demonstrating proficiency in Data Engineering, Predictive Modeling, Econometrics, and Python programming.*

# 🏆 World Cup Predictor: Advanced ELO & Machine Learning Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Engineering-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This repository contains a comprehensive, end-to-end Data Science pipeline designed to predict the outcomes of the FIFA World Cup knockout stages. Instead of relying on a single metric, this project bridges **historical econometrics** with **modern machine learning**, blending over 150 years of international football data with real-time tournament statistics.

The project features a dual-engine prediction system:
1. **A Stochastic Monte Carlo Simulator** (10,000 parallel realities evaluating tournament variance).
2. **A Deterministic Bracket Engine** (A strict, math-based progression finding the single most likely path to the final).

## 🧠 The Mathematics & Machine Learning
### The Elo Rating System
At the core of the historical analysis is a custom-built Elo rating engine. The probability of Team A defeating Team B is continuously calculated and updated using the standard formula:

$$P(A) = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

### The Random Forest "Form Modifier"
Historical greatness (Elo) is not enough to win a modern tournament. The system scrapes live, in-tournament data (Possession %, Goal Differentials) directly from FBref. A **Random Forest Classifier** is trained on historical World Cup matchups and then applies a "Form Index" to adjust the raw Elo probabilities based on current on-pitch dominance.

## 📂 Architecture & Data Pipeline
The project is strictly structured into sequential Jupyter Notebooks to showcase a clean data engineering workflow:

* `01_data_cleaning.ipynb` - Ingestion and standardization of raw Kaggle datasets.
* `02_elo_engine.ipynb` - Simulation of 40,000+ matches to calculate the modern Elo standings.
* `03_fbref_scraping.ipynb` - Bypassing anti-bot protections to extract real-time group stage data from FBref.
* `04_feature_engineering.ipynb` - Data blending and calculating "differentials" (e.g., Elo difference, Possession difference) using fuzzy matching.
* `05_model_training.ipynb` - Training the Random Forest algorithm to evaluate knockout probabilities.
* `06_visualizations.ipynb` - Rendering professional, presentation-ready Seaborn/Matplotlib charts.
* `07_bracket_simulator.ipynb` - Running 10,000 Monte Carlo simulations on the strict 32-team tournament topology.
* `08_deterministic_bracket.ipynb` - Generating the absolute most mathematically likely path, injecting actual real-world results to keep the bracket accurate.

## 📊 Key Results
The raw outputs, including the visual charts, `.txt` bracket logs, and full CSV reports, can be found in the `/data/processed/` directory. The dual-model approach successfully demonstrates the difference between absolute strength (Deterministic) and tournament survivability (Monte Carlo variance).

## 🚀 How to Run Locally
1. Clone this repository.
2. Set up a local `.venv` and install required dependencies (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).
3. Run the notebooks sequentially from `01` to `08` to completely rebuild the database and run new simulations.

---
*Created as a comprehensive portfolio project demonstrating proficiency in Data Engineering, Predictive Modeling, and Python programming.*

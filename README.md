# AI-Based Food Waste Reduction System (SDG 12)

## Overview
This project demonstrates how Artificial Intelligence can be used to reduce food waste by predicting consumption patterns and supporting sustainable decision-making.

Aligned with **UN SDG 12: Responsible Consumption and Production**.

## Problem Statement
Food waste occurs due to poor demand forecasting and lack of data-driven planning. This leads to economic loss and increased environmental impact.

## Solution
An AI-based system that:
- Analyzes historical food consumption data
- Predicts future demand using machine learning
- Provides recommendations to reduce over-purchasing and waste

## Visual Overview

### System Architecture
![System Architecture](diagrams/system_architecture.drawio.png)

### Conceptual Dashboard View
![Dashboard Mockup](diagrams/dashboard_mockup.png)

## AI Technologies Used
- Machine Learning (Regression)
- Python
- Pandas, NumPy
- Scikit-learn

## Target Users
- Households
- Small food vendors
- Restaurants and cafeterias

## Project Status
Prototype implementation for one kitchen with demand forecasting, recipient matching,
basic route ordering, and redistribution history.

## Data architecture

The dashboard uses four CSV files in [`data/`](./data):

- [`kitchen_history.csv`](./data/kitchen_history.csv) trains the demand model.
- [`daily_operations.csv`](./data/daily_operations.csv) provides the current kitchen scenario.
- [`recipients.csv`](./data/recipients.csv) stores active organizations that can receive surplus.
- [`redistribution_records.csv`](./data/redistribution_records.csv) stores planned or delivered allocations.

On first startup, the dashboard trains a categorical demand model and writes
[`demand_model.joblib`](./demand_model.joblib). The model predicts actual consumption;
expected surplus is then calculated as planned production minus predicted demand.

## Run the Dashboard
From the project root, install the dependencies and start Streamlit:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

The dashboard opens in a browser. Use the kitchen scenario controls to forecast demand,
review expected surplus, match recipients, and record a redistribution plan.

## SDG Alignment
- SDG 12: Responsible Consumption and Production
- SDG 13: Climate Action

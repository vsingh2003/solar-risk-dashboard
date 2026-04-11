# SPICE Solar Underperformance Analysis Dashboard

## Overview
This project is an interactive Streamlit dashboard developed for the SPICE project to analyze solar inverter performance and identify underperformance or anomaly patterns in solar production data. The dashboard combines data analysis, feature engineering, model evaluation, and deployment into one user-facing application.

## Live App
streamlit app link
https://solar-risk-dashboard-app.streamlit.app/


## Problem Statement
This project focuses on identifying anomaly patterns and underperformance in solar inverter production data. The goal was to analyze solar generation behavior, detect suspicious low-output days, and present the findings through an interactive Streamlit dashboard.

## Project Objective
The objective of this project was to build a practical dashboard that supports solar performance monitoring and anomaly-related insights using inverter-level production data. It was also designed to demonstrate how a machine learning and analytics workflow can be deployed in an accessible format.

## My Role
I completed the workflow for this project, including exploratory data analysis, cleaning, feature engineering, modeling, application development, deployment, and documentation.

## Repository Structure
This repository includes:
- `1_EDA_and_Cleaning.ipynb` — exploratory data analysis and preprocessing
- `2_Feature_Engineering.ipynb` — feature creation and transformation
- `3_Modeling_and_Evaluation.ipynb` — model building and evaluation
- `app.py` — main Streamlit dashboard file
- `pipeline.py` — reusable workflow logic
- `requirements.txt` — dependencies for running the app
- `pages/` — additional app pages
- `images/` — screenshots or visual assets
- CSV files containing project outputs and reporting results

## Dataset
The project uses solar production data with inverter-level performance variables and related outputs. The workflow focuses on detecting underperformance patterns and suspicious low-generation behavior in solar systems.

## Tools Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter / Google Colab
- GitHub

## Process
- Collected and prepared solar inverter production data
- Performed exploratory data analysis and cleaning
- Engineered relevant features for underperformance analysis
- Built and evaluated machine learning workflow components
- Developed a Streamlit dashboard for interactive presentation
- Organized the codebase into notebooks, scripts, and deployment files
- Deployed the app for live access and portfolio use

## Results
This project resulted in a deployed Streamlit application that presents solar underperformance analysis in an interactive and accessible format. The repository also contains modular notebooks, app code, supporting scripts, CSV outputs, and deployment dependencies, which improve reproducibility and project clarity.

## Deployment Value
This project demonstrates how a data science workflow can move beyond notebooks into a deployed application. The deployment adds practical value by making the analysis easier to explore, present, and share.

## Reflection
This project helped me understand how to move from exploratory analysis and modeling into deployment using Streamlit. By organizing the workflow into notebooks, pipeline files, and an interactive dashboard, I was able to connect data preparation, model outputs, and user-facing presentation in one project. This strengthened my understanding of how machine learning and analytics solutions can be deployed in a more practical and accessible format. It also improved my confidence in building portfolio-ready projects that combine technical analysis, deployment, and presentation.

## Portfolio Relevance
This project demonstrates:
- Interactive dashboard deployment
- Streamlit application development
- End-to-end data science workflow
- Solar underperformance and anomaly analysis
- Project structuring and reproducibility
- Applied machine learning presentation

## How to Run Locally
1. Clone this repository
2. Install dependencies from `requirements.txt`
3. Run the Streamlit app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
This project is part of my applied machine learning and data science portfolio and highlights my ability to combine analysis, modeling, deployment, and presentation in a real-world style project.

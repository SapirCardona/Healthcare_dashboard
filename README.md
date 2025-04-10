# Mental Health in Tech – Interactive Dashboard

This Streamlit app presents an interactive dashboard based on a cleaned dataset about mental health in the tech industry. It enables data exploration, filtering, visual storytelling, and includes a simple machine learning model to predict the likelihood of seeking mental health treatment.

## Project Goals

- Visualize key trends and patterns around mental health among tech employees.
- Empower users to filter the data by age, gender, and country to explore insights that are most relevant to them.
- Predict the likelihood of seeking treatment using a Random Forest Classifier based on personal input.
- Encourage awareness and discussion about mental health in the tech sector through data.

## Features

- Clean and intuitive UI with custom color palette for a calm, supportive feel.
- Sidebar filters for age, gender, and country.
- Interactive visualizations (bar charts, histograms, pie charts, heatmaps).
- Option to download filtered dataset for further analysis.
- ML-powered prediction lab to estimate the probability of seeking treatment, based on user-provided details.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Plotly
- Streamlit

## Dataset

The dataset used is a cleaned version of a mental health survey among tech workers, originally sourced from OSMI (Open Sourcing Mental Illness), and includes fields such as:
- Age
- Gender
- Country
- Mental health history
- Work environment and benefits
- Whether the respondent has sought treatment

## Machine Learning

The app uses a Random Forest Classifier trained on selected features (age, gender, benefits, family history, work interference) to predict the probability of seeking treatment.

## Dataset

Source: [Kaggle – Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

## Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/mental-health-dashboard.git
   cd mental-health-dashboard


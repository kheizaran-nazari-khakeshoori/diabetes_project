# Diabetes Project

A simple Python project for loading, cleaning, and saving a diabetes dataset.

## Structure

```text
diabetes_project/
├── data/
│   ├── raw/
│   │   └── diabetes.csv
│   └── processed/
├── src/
│   ├── __init__.py
│   └── data_cleaning.py
├── .gitignore
├── README.md
├── requirements.txt
└── main.py
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Put the raw dataset at `data/raw/diabetes.csv`.

3. Run the cleaning script:

```bash
python main.py
```

The cleaned file will be saved to `data/processed/diabetes_cleaned.csv`.

The pipeline also generates plots in `report/plots/`, including class distribution, feature correlation, and model performance charts.

# CodeAlpha Sales Prediction

A machine learning project that predicts product sales from advertising spend across TV, radio, and newspaper channels.

## Overview

This project uses linear regression to understand how advertising budgets affect sales. It loads the dataset, trains a model, evaluates prediction accuracy, and displays an actual-vs-predicted sales chart.

## Features

- Loads advertising sales data from CSV
- Splits data into training and testing sets
- Trains a linear regression model
- Calculates Mean Absolute Error and R2 Score
- Visualizes actual sales compared with predicted sales

## Technologies Used

- Python
- Pandas
- Matplotlib
- scikit-learn

## Project Structure

```text
sales_prediction/
├── sales_prediction.py
├── Advertising_sales.csv
└── README.d.md
```

## How to Run

1. Open the `sales_prediction` folder.
2. Install the required Python packages:

```bash
pip install pandas matplotlib scikit-learn
```

3. Run the project:

```bash
python sales_prediction.py
```

## Output

The script prints model evaluation metrics and opens a scatter plot comparing actual and predicted sales.

## Author

Ashutosh Tiwari

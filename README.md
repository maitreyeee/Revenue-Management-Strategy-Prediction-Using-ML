# Revenue Management Strategy Prediction Using ML

This project compares two airline RM strategies—UDP and ProBP—using simulated data. It uses forecast accuracy metrics and ML models to predict which strategy performs better under different demand conditions, and interprets key drivers using logistic regression, Random Forest and XGBoost.

## Project Structure

```
.
├── data_processing/
│   ├── data_loader.py
│   ├── data_preprocessor.py
│   ├── ml_analysis.py
│   └── visualization.py
├── Data/
│   └── PvsUdata/
├── main.py
├── test_data_processing.py
└── README.md
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/maitreyeee/Revenue-Management-Strategy-Prediction-Using-ML.git
cd Revenue-Management-Strategy-Prediction-Using-ML
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your data files in the `Data/PvsUdata/` directory.

2. Run the main script:
```bash
python main.py
```

This will:
- Load and preprocess the data
- Calculate forecast metrics
- Train and compare ML models
- Generate visualizations
- Save results

## Features

- Data preprocessing and feature engineering
- Multiple ML models (Random Forest, Logistic Regression, XGBoost)
- Model comparison and evaluation
- Visualization of results
- SHAP value analysis for model interpretation

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- shap

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
# NYC Airbnb ML Pipeline

This project implements a complete MLOps pipeline for predicting Airbnb prices in NYC using machine learning. The pipeline includes data cleaning, testing, model training, and evaluation steps.

## Project Links

- **W&B Project**: https://wandb.ai/duytm112-western-governor-university/nyc_airbnb
- **GitHub Repository**: [Your GitHub repository URL here]

## Project Overview

This project demonstrates a complete ML pipeline that:
1. Downloads raw Airbnb data
2. Cleans and preprocesses the data
3. Performs data validation and testing
4. Splits data into training and test sets
5. Trains a Random Forest model with hyperparameter optimization
6. Evaluates model performance on test data

## Pipeline Steps

### 1. Download (`download`)
- Downloads sample data from the provided dataset
- Uploads raw data to W&B as `sample.csv`

### 2. Basic Cleaning (`basic_cleaning`)
- Removes outliers based on price range (10-350 USD)
- Filters data to NYC area (longitude: -74.25 to -73.50, latitude: 40.5 to 41.2)
- Converts date columns to proper format
- Creates cleaned dataset as `clean_sample.csv`

### 3. Data Check (`data_check`)
- Validates data quality through multiple tests:
  - Column names validation
  - Neighborhood names validation
  - Geographic boundaries validation
  - Row count validation (15,000 < rows < 1,000,000)
  - Price range validation
  - Distribution similarity test (KL divergence)

### 4. Data Split (`data_split`)
- Splits cleaned data into training/validation (80%) and test (20%) sets
- Uses stratification by neighborhood_group
- Creates `trainval_data.csv` and `test_data.csv`

### 5. Train Random Forest (`train_random_forest`)
- Implements preprocessing pipeline with:
  - Categorical encoding (OneHotEncoder for neighborhood_group, OrdinalEncoder for room_type)
  - Missing value imputation
  - Feature engineering (days since last review)
  - TF-IDF features for property names
- Trains Random Forest model with optimized hyperparameters
- Logs model metrics (MAE, R²) to W&B
- Exports model as MLflow artifact

### 6. Test Regression Model (`test_regression_model`)
- Evaluates trained model on test dataset
- Compares test performance with validation performance
- Ensures no overfitting

## Model Performance

The best performing model achieved:
- **Validation Performance**: R² = 0.551, MAE = 34.18
- **Test Performance**: R² = 0.562, MAE = 33.85

### Best Hyperparameters
- `n_estimators`: 200
- `max_depth`: 50
- `min_samples_split`: 4
- `min_samples_leaf`: 3
- `max_features`: 0.5

## Environment Setup

1. Install Miniconda
2. Create environment: `conda env create -f environment.yml`
3. Activate environment: `conda activate nyc_airbnb_dev`
4. Login to W&B: `wandb login [your-api-key]`

## Running the Pipeline

### Run entire pipeline:
```bash
mlflow run .
```

### Run specific steps:
```bash
mlflow run . -P steps=download,basic_cleaning
```

### Run with hyperparameter optimization:
```bash
mlflow run . -P steps=train_random_forest -P hydra_options="modeling.random_forest.max_depth=10,50 modeling.random_forest.n_estimators=100,200 -m"
```

## Project Structure

```
Project-Build-an-ML-Pipeline-Starter/
├── main.py                 # Main pipeline orchestration
├── config.yaml            # Configuration parameters
├── environment.yml        # Conda environment specification
├── MLproject             # MLflow project definition
├── src/
│   ├── basic_cleaning/   # Data cleaning step
│   ├── data_check/       # Data validation step
│   ├── eda/             # Exploratory data analysis
│   └── train_random_forest/ # Model training step
└── components/           # Pre-implemented components
    ├── get_data/        # Data download component
    ├── train_val_test_split/ # Data splitting component
    └── test_regression_model/ # Model testing component
```

## Key Features

- **Reproducible Pipeline**: All steps are versioned and tracked
- **Data Validation**: Comprehensive testing ensures data quality
- **Hyperparameter Optimization**: Automated search for best model parameters
- **Model Tracking**: All experiments tracked in W&B
- **Modular Design**: Each step is independent and reusable
- **Configuration Management**: All parameters managed through config.yaml

## Technologies Used

- **MLflow**: Pipeline orchestration and model tracking
- **Weights & Biases**: Experiment tracking and artifact management
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Pandas**: Data manipulation
- **Hydra**: Configuration management
- **Pytest**: Data validation testing

## Future Improvements

1. Add more sophisticated feature engineering
2. Implement additional ML algorithms (XGBoost, Neural Networks)
3. Add model interpretability tools
4. Implement automated retraining pipeline
5. Add model monitoring and drift detection

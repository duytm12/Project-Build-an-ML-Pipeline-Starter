# NYC Airbnb ML Pipeline

## Project Overview

This project demonstrates a complete MLOps pipeline for predicting Airbnb rental prices in New York City. As part of the MLOps course, I built an end-to-end machine learning pipeline that showcases best practices in data engineering, model development, and deployment automation.

### What I Built

I developed a comprehensive ML pipeline that:
- **Ingests and cleans** raw Airbnb data from NYC
- **Validates data quality** through automated testing
- **Trains machine learning models** with hyperparameter optimization
- **Evaluates model performance** and prevents overfitting
- **Deploys models** with proper versioning and release management

This project represents a real-world scenario where a property management company needs to estimate rental prices based on similar properties, with new data arriving weekly requiring model retraining.

## Project Links

- **W&B Project**: https://wandb.ai/duytm112-western-governor-university/nyc_airbnb
- **GitHub Repository**: https://github.com/duytm12/Project-Build-an-ML-Pipeline-Starter

## Technical Implementation

### Pipeline Architecture

The pipeline consists of 6 main steps, each implemented as a modular MLflow component:

1. **Download** (`download`) - Fetches raw data and uploads to W&B
2. **Basic Cleaning** (`basic_cleaning`) - Removes outliers and filters geographic boundaries
3. **Data Check** (`data_check`) - Validates data quality through automated tests
4. **Data Split** (`data_split`) - Splits data into training/validation/test sets
5. **Train Random Forest** (`train_random_forest`) - Trains model with preprocessing pipeline
6. **Test Regression Model** (`test_regression_model`) - Evaluates model on test set

### Key Technical Features

#### Data Processing
- **Geographic filtering**: Ensures data points are within NYC boundaries (longitude: -74.25 to -73.50, latitude: 40.5 to 41.2)
- **Price outlier removal**: Filters prices between $10-$350
- **Missing value imputation**: Handles missing data with appropriate strategies
- **Feature engineering**: Creates time-based features (days since last review)

#### Machine Learning Pipeline
- **Preprocessing**: Categorical encoding, TF-IDF for text features, date transformations
- **Model**: Random Forest with optimized hyperparameters
- **Validation**: Stratified sampling by neighborhood group
- **Evaluation**: R² and MAE metrics with overfitting detection

#### MLOps Best Practices
- **Version control**: All code changes tracked in Git
- **Artifact management**: Data and models stored in W&B
- **Experiment tracking**: All runs logged with metrics and parameters
- **Release management**: Versioned releases for production deployment
- **Testing**: Automated data quality and model performance tests

## Model Performance

After hyperparameter optimization, the best model achieved:
- **Validation Performance**: R² = 0.551, MAE = 34.18
- **Test Performance**: R² = 0.562, MAE = 33.85

### Optimized Hyperparameters
- `n_estimators`: 200
- `max_depth`: 50
- `min_samples_split`: 4
- `min_samples_leaf`: 3
- `max_features`: 0.5

The model shows no signs of overfitting, with test performance comparable to validation performance.

## Environment Setup

### Prerequisites
- Python 3.9+
- Conda package manager
- W&B account

### Installation
```bash
# Clone the repository
git clone https://github.com/duytm12/Project-Build-an-ML-Pipeline-Starter.git
cd Project-Build-an-ML-Pipeline-Starter

# Create conda environment
conda env create -f environment.yml
conda activate nyc_airbnb_dev

# Login to W&B
wandb login [your-api-key]
```

## Usage

### Run Complete Pipeline
```bash
mlflow run .
```

### Run Specific Steps
```bash
# Run only data cleaning
mlflow run . -P steps=download,basic_cleaning

# Run with custom parameters
mlflow run . -P hydra_options="etl.min_price=20 etl.max_price=300"
```

### Hyperparameter Optimization
```bash
mlflow run . -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.max_depth=10,50 modeling.random_forest.n_estimators=100,200 -m"
```

### Test Released Pipeline
```bash
mlflow run https://github.com/duytm12/Project-Build-an-ML-Pipeline-Starter.git \
  -v v1.0.0 -P hydra_options="etl.sample='sample2.csv'"
```

## Project Structure

```
Project-Build-an-ML-Pipeline-Starter/
├── src/                          # Pipeline step implementations
│   ├── basic_cleaning/           # Data cleaning step
│   ├── data_check/              # Data validation tests
│   └── train_random_forest/     # Model training step
├── components/                   # Reusable MLflow components
├── config.yaml                  # Pipeline configuration
├── main.py                      # Main pipeline orchestration
├── environment.yml              # Conda environment
└── README.md                    # This file
```

## What I Learned

This project taught me valuable skills in:

1. **MLOps Engineering**: Building production-ready ML pipelines
2. **Data Engineering**: Cleaning, validating, and processing real-world data
3. **Model Development**: Feature engineering and hyperparameter optimization
4. **Testing**: Implementing automated tests for data and model quality
5. **Version Control**: Managing code, data, and model versions
6. **Deployment**: Creating reproducible releases for production use

## Future Improvements

Potential enhancements for this pipeline:
- Add more sophisticated feature engineering
- Implement model interpretability tools
- Add A/B testing capabilities
- Integrate with CI/CD pipelines
- Add monitoring and alerting systems

## License

This project is part of the MLOps course curriculum and follows the original project's licensing terms.

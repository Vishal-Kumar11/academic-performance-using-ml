# Academic Performance Prediction using Machine Learning

A comprehensive machine learning system for predicting student academic performance using multiple algorithms and advanced data analysis techniques.

## Overview

This project implements an end-to-end machine learning pipeline to predict student math performance based on demographic and educational factors. The system compares 13 different algorithms and provides interactive visualizations for model interpretation.

## Features

- **13 ML Models**: Linear Regression, Ridge, Lasso, K-Neighbors, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, CatBoost, LightGBM, Support Vector Regressor, Neural Network
- **Model Comparison**: Comprehensive evaluation with cross-validation and multiple metrics
- **Web Interface**: Flask application with Bootstrap UI for predictions
- **Batch Processing**: CSV upload for multiple predictions
- **Model Explainability**: SHAP values for feature importance
- **Interactive Dashboard**: Plotly visualizations for model performance

## Dataset

The Student Performance Dataset contains 1,000 student records with 8 features:
- Gender, Race/Ethnicity, Parental Education Level
- Lunch Program, Test Preparation Course
- Reading Score, Writing Score
- Math Score (target variable)

## Technical Stack

- **Backend**: Python 3.8+, Flask 3.0
- **ML Libraries**: scikit-learn, XGBoost, CatBoost, LightGBM, TensorFlow
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Frontend**: Bootstrap 5, HTML/CSS/JavaScript
- **Deployment**: Docker, Docker Compose

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Vishal-Kumar11/academic-performance-using-ml.git
cd academic-performance-using-ml
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python application.py
```

The application will be available at `http://localhost:5000`

## Usage

### Single Prediction
Navigate to the Predict page and enter student information to get a math score prediction.

### Batch Processing
Upload a CSV file with multiple student records for batch predictions.

### Model Dashboard
View model comparison results, performance metrics, and feature importance analysis.

## Project Structure

```
├── application.py              # Flask web application
├── requirements.txt            # Python dependencies
├── config.py                  # Configuration settings
├── src/                       # Source code
│   ├── components/            # ML pipeline components
│   ├── pipeline/             # Training and prediction pipelines
│   └── utils/                # Utility functions
├── templates/                 # HTML templates
├── static/                    # CSS and JavaScript files
├── artifacts/                 # Trained models and data
├── docs/                      # Documentation
└── notebook/                  # Jupyter notebooks
```

## Model Performance

The best performing model is CatBoost with:
- R² Score: 87.21%
- MAE: 4.23
- RMSE: 5.67

## API Endpoints

- `GET /` - Home page
- `GET /predict` - Prediction form
- `POST /predict` - Submit prediction
- `GET /batch` - Batch processing page
- `POST /batch` - Process batch file
- `GET /dashboard` - Model dashboard
- `GET /api/model-info` - Model information API

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact

Vishal Kumar
- GitHub: [@Vishal-Kumar11](https://github.com/Vishal-Kumar11)
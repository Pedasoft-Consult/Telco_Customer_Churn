# Telco Customer Churn Prediction

This project implements a machine learning solution to predict customer churn for a telecommunications company. The application analyzes customer characteristics and service usage to identify customers at risk of churning, allowing for proactive retention strategies.

## Project Structure

The project follows a modular structure with separate components for data processing, model training, and web application deployment:

```
customer_churn_prediction/
│
├── config/                    # Configuration files
│   ├── config.yaml            # Main configuration
│   └── model_config.yaml      # Model-specific configurations
│
├── data/                      # Data directory
│   ├── raw/                   # Original dataset
│   ├── processed/             # Processed dataset
│   └── external/              # External data sources
│
├── logs/                      # Application logs
│
├── models/                    # Trained models
│   ├── trained/               # Saved model files
│   └── evaluation/            # Model performance metrics
│
├── notebooks/                 # Jupyter notebooks
│
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   ├── models/                # Model training and evaluation
│   ├── utils/                 # Utility functions
│   └── visualization/         # Data visualization
│
├── webapp/                    # Flask web application
│   ├── templates/             # HTML templates
│   └── static/                # CSS, JS, and images
│
├── tests/                     # Unit tests
│
├── requirements.txt           # Project dependencies
├── setup.py                   # Package setup
├── README.md                  # Project documentation
└── main.py                    # Main entry point
```

## Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables, and scales numerical features.
- **Feature Engineering**: Creates domain-specific features to improve model performance.
- **Model Development**: Implements and compares multiple machine learning models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Neural Networks (optional)
- **Model Evaluation**: Evaluates models using metrics such as accuracy, precision, recall, F1 score, and AUC-ROC.
- **Web Application**: Provides a user-friendly interface for making predictions on new customer data.
- **Logging**: Comprehensive logging system to track application behavior.
- **Configuration Management**: Centralized configuration for all components of the application.


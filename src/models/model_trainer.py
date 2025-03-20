import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import json

from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger()
config = get_config()


class ModelTrainer:
    """
    Class for training and evaluating machine learning models for customer churn prediction.
    """

    def __init__(self):
        """Initialize the ModelTrainer with configuration."""
        self.model_config = config.get_section('models')
        self.eval_config = config.get_section('evaluation')
        self.model_save_config = config.get_section('model_saving')

        # Get evaluation settings
        self.cv_enabled = self.eval_config.get('cross_validation', {}).get('enabled', True)
        self.cv_folds = self.eval_config.get('cross_validation', {}).get('n_splits', 5)
        self.random_state = self.eval_config.get('cross_validation', {}).get('random_state', 42)

        # Metrics to calculate
        self.metrics = self.eval_config.get('metrics',
                                            ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

        # Initialize dictionary to store model results
        self.models = {}
        self.results = {}

    def train_all_models(self, X_train, y_train):
        """
        Train all enabled models from the configuration.

        Args:
            X_train (pandas.DataFrame): Training features.
            y_train (pandas.Series): Training target.

        Returns:
            dict: Dictionary of trained models.
        """
        logger.info("Starting to train all enabled models")

        # Train logistic regression if enabled
        if self.model_config.get('logistic_regression', {}).get('enabled', True):
            logger.info("Training Logistic Regression model")
            self.train_logistic_regression(X_train, y_train)

        # Train random forest if enabled
        if self.model_config.get('random_forest', {}).get('enabled', True):
            logger.info("Training Random Forest model")
            self.train_random_forest(X_train, y_train)

        # Train XGBoost if enabled
        if self.model_config.get('xgboost', {}).get('enabled', True):
            logger.info("Training XGBoost model")
            self.train_xgboost(X_train, y_train)

        # Train neural network if enabled
        if self.model_config.get('neural_network', {}).get('enabled', False):
            logger.info("Training Neural Network model")
            self.train_neural_network(X_train, y_train)

        logger.info(f"Completed training {len(self.models)} models")
        return self.models

    def train_logistic_regression(self, X_train, y_train):
        """
        Train a logistic regression model with hyperparameter tuning.

        Args:
            X_train (pandas.DataFrame): Training features.
            y_train (pandas.Series): Training target.

        Returns:
            GridSearchCV: Trained logistic regression model with best parameters.
        """
        logger.info("Setting up Logistic Regression with hyperparameter tuning")

        # Get hyperparameters from config
        hyperparams = self.model_config.get('logistic_regression', {}).get('hyperparameters', {})

        # Create parameter grid - ensure all values are lists
        param_grid = {
            'C': hyperparams.get('C', [0.001, 0.01, 0.1, 1, 10, 100]),
            'penalty': hyperparams.get('penalty', ['l1', 'l2']),
            'solver': hyperparams.get('solver', ['liblinear']),
            'max_iter': hyperparams.get('max_iter', [1000])  # Fixed: wrapped in a list
        }

        # Create base model
        base_model = LogisticRegression(random_state=self.random_state)

        # Create grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        # Fit the model
        logger.info("Fitting Logistic Regression with GridSearchCV")
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        logger.info(f"Best Logistic Regression parameters: {grid_search.best_params_}")
        logger.info(f"Best Logistic Regression CV score: {grid_search.best_score_:.4f}")

        # Store the model
        self.models['logistic_regression'] = best_model

        # Calculate and store cross-validation metrics if enabled
        if self.cv_enabled:
            self._calculate_cv_metrics('logistic_regression', best_model, X_train, y_train)

        return best_model

    def train_random_forest(self, X_train, y_train):
        """
        Train a random forest model with hyperparameter tuning.

        Args:
            X_train (pandas.DataFrame): Training features.
            y_train (pandas.Series): Training target.

        Returns:
            GridSearchCV: Trained random forest model with best parameters.
        """
        logger.info("Setting up Random Forest with hyperparameter tuning")

        # Get hyperparameters from config
        hyperparams = self.model_config.get('random_forest', {}).get('hyperparameters', {})

        # Create parameter grid - ensure all values are lists
        param_grid = {
            'n_estimators': hyperparams.get('n_estimators', [100, 200, 300]),
            'max_depth': hyperparams.get('max_depth', [5, 8, 15, 25, 30]),
            'min_samples_split': hyperparams.get('min_samples_split', [2, 5, 10, 15]),
            'min_samples_leaf': hyperparams.get('min_samples_leaf', [1, 2, 5, 10])
        }

        # Create base model
        base_model = RandomForestClassifier(random_state=self.random_state)

        # Create grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        # Fit the model
        logger.info("Fitting Random Forest with GridSearchCV")
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        logger.info(f"Best Random Forest parameters: {grid_search.best_params_}")
        logger.info(f"Best Random Forest CV score: {grid_search.best_score_:.4f}")

        # Store the model
        self.models['random_forest'] = best_model

        # Calculate and store cross-validation metrics if enabled
        if self.cv_enabled:
            self._calculate_cv_metrics('random_forest', best_model, X_train, y_train)

        return best_model

    def train_xgboost(self, X_train, y_train):
        """
        Train an XGBoost model with hyperparameter tuning.

        Args:
            X_train (pandas.DataFrame): Training features.
            y_train (pandas.Series): Training target.

        Returns:
            GridSearchCV: Trained XGBoost model with best parameters.
        """
        logger.info("Setting up XGBoost with hyperparameter tuning")

        # Get hyperparameters from config
        hyperparams = self.model_config.get('xgboost', {}).get('hyperparameters', {})

        # Create parameter grid - ensure all values are lists
        param_grid = {
            'n_estimators': hyperparams.get('n_estimators', [100, 200, 300]),
            'max_depth': hyperparams.get('max_depth', [3, 5, 7, 9]),
            'learning_rate': hyperparams.get('learning_rate', [0.01, 0.05, 0.1, 0.2]),
            'subsample': hyperparams.get('subsample', [0.8, 0.9, 1.0]),
            'colsample_bytree': hyperparams.get('colsample_bytree', [0.8, 0.9, 1.0])
        }

        # Create base model - Removed the deprecated use_label_encoder parameter
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=self.random_state,
            eval_metric=['logloss']
        )

        # Create grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        # Fit the model
        logger.info("Fitting XGBoost with GridSearchCV")
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
        logger.info(f"Best XGBoost CV score: {grid_search.best_score_:.4f}")

        # Store the model
        self.models['xgboost'] = best_model

        # Calculate and store cross-validation metrics if enabled
        if self.cv_enabled:
            self._calculate_cv_metrics('xgboost', best_model, X_train, y_train)

        return best_model

    def train_neural_network(self, X_train, y_train):
        """
        Train a neural network model with hyperparameter tuning.

        Args:
            X_train (pandas.DataFrame): Training features.
            y_train (pandas.Series): Training target.

        Returns:
            GridSearchCV: Trained neural network model with best parameters.
        """
        logger.info("Setting up Neural Network with hyperparameter tuning")

        # Get hyperparameters from config
        hyperparams = self.model_config.get('neural_network', {}).get('hyperparameters', {})

        # Create parameter grid - ensure all values are lists
        param_grid = {
            'hidden_layer_sizes': hyperparams.get('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)]),
            'activation': hyperparams.get('activation', ['relu', 'tanh']),
            'alpha': hyperparams.get('alpha', [0.0001, 0.001, 0.01]),
            'learning_rate_init': hyperparams.get('learning_rate_init', [0.001, 0.01])
        }

        # Create base model - ensure max_iter is a parameter in the model constructor, not in param_grid
        max_iter_value = hyperparams.get('max_iter', 1000)
        if isinstance(max_iter_value, list):
            max_iter_value = max_iter_value[0]  # Take first value if it's a list

        base_model = MLPClassifier(
            max_iter=max_iter_value,
            random_state=self.random_state
        )

        # Create grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        # Fit the model
        logger.info("Fitting Neural Network with GridSearchCV")
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        logger.info(f"Best Neural Network parameters: {grid_search.best_params_}")
        logger.info(f"Best Neural Network CV score: {grid_search.best_score_:.4f}")

        # Store the model
        self.models['neural_network'] = best_model

        # Calculate and store cross-validation metrics if enabled
        if self.cv_enabled:
            self._calculate_cv_metrics('neural_network', best_model, X_train, y_train)

        return best_model

    def _calculate_cv_metrics(self, model_name, model, X, y):
        """
        Calculate cross-validation metrics for a model.

        Args:
            model_name (str): Name of the model.
            model: Trained model.
            X (pandas.DataFrame): Features.
            y (pandas.Series): Target.
        """
        logger.info(f"Calculating cross-validation metrics for {model_name}")

        cv_results = {}

        # Calculate CV scores for each metric
        for metric in self.metrics:
            if metric == 'accuracy':
                scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='accuracy')
            elif metric == 'precision':
                scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='precision')
            elif metric == 'recall':
                scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='recall')
            elif metric == 'f1':
                scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='f1')
            elif metric == 'roc_auc':
                scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='roc_auc')

            cv_results[metric] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'values': scores.tolist()
            }

            logger.info(f"{model_name} {metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        # Store the results
        self.results[model_name] = {
            'cv_results': cv_results
        }

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models on the test set.

        Args:
            X_test (pandas.DataFrame): Test features.
            y_test (pandas.Series): Test target.

        Returns:
            dict: Dictionary with evaluation results.
        """
        logger.info("Evaluating all trained models on test set")

        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name} on test set")

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            test_results = {}

            if 'accuracy' in self.metrics:
                test_results['accuracy'] = float(accuracy_score(y_test, y_pred))

            if 'precision' in self.metrics:
                test_results['precision'] = float(precision_score(y_test, y_pred))

            if 'recall' in self.metrics:
                test_results['recall'] = float(recall_score(y_test, y_pred))

            if 'f1' in self.metrics:
                test_results['f1'] = float(f1_score(y_test, y_pred))

            if 'roc_auc' in self.metrics:
                test_results['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))

            # Store test results
            if model_name in self.results:
                self.results[model_name]['test_results'] = test_results
            else:
                self.results[model_name] = {
                    'test_results': test_results
                }

            # Log the results
            for metric, value in test_results.items():
                logger.info(f"{model_name} test {metric}: {value:.4f}")

        return self.results

    def save_models(self):
        """
        Save all trained models to disk.

        Returns:
            dict: Dictionary with paths to saved models.
        """
        logger.info("Saving trained models to disk")

        save_path = self.model_save_config.get('save_path', 'models/trained/')
        saved_models = {}

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save each model
        for model_name, model in self.models.items():
            model_path = os.path.join(save_path, f"{model_name}.pkl")

            try:
                logger.info(f"Saving {model_name} to {model_path}")
                joblib.dump(model, model_path)
                saved_models[model_name] = model_path
            except Exception as e:
                logger.error(f"Error saving {model_name}: {e}")

        # Save best model
        best_model_name = self.select_best_model()
        if best_model_name:
            best_model_path = os.path.join(save_path, self.model_save_config.get('best_model_name', 'best_model.pkl'))

            try:
                logger.info(f"Saving best model ({best_model_name}) to {best_model_path}")
                joblib.dump(self.models[best_model_name], best_model_path)
                saved_models['best_model'] = best_model_path
            except Exception as e:
                logger.error(f"Error saving best model: {e}")

        # Save evaluation results
        results_path = self.model_save_config.get('evaluation_results_path', 'models/evaluation/model_comparison.json')

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        try:
            logger.info(f"Saving evaluation results to {results_path}")
            with open(results_path, 'w') as f:
                results_with_best = self.results.copy()
                results_with_best['best_model'] = best_model_name
                json.dump(results_with_best, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")

        return saved_models

    def select_best_model(self):
        """
        Select the best model based on F1 score on the test set.

        Returns:
            str: Name of the best model.
        """
        logger.info("Selecting best model based on F1 score")

        best_model = None
        best_score = -1

        for model_name, result in self.results.items():
            if 'test_results' in result and 'f1' in result['test_results']:
                f1 = result['test_results']['f1']

                if f1 > best_score:
                    best_score = f1
                    best_model = model_name

        if best_model:
            logger.info(f"Best model: {best_model} with F1 score: {best_score:.4f}")
        else:
            logger.warning("Could not determine best model")

        return best_model


# Singleton instance for easy access
_model_trainer_instance = None


def get_model_trainer():
    """
    Get a singleton instance of the ModelTrainer.

    Returns:
        ModelTrainer: A ModelTrainer instance.
    """
    global _model_trainer_instance
    if _model_trainer_instance is None:
        _model_trainer_instance = ModelTrainer()
    return _model_trainer_instance


if __name__ == "__main__":
    # Test the model trainer
    from src.data.data_loader import get_data_loader
    from src.data.preprocessor import get_preprocessor

    # Load and preprocess data
    data_loader = get_data_loader()
    preprocessor = get_preprocessor()

    df_raw = data_loader.load_raw_data()
    df_processed = preprocessor.preprocess_data(df_raw)
    X_train, X_test, y_train, y_test = preprocessor.train_test_data_split(df_processed)

    # Train models
    model_trainer = get_model_trainer()
    models = model_trainer.train_all_models(X_train, y_train)

    # Evaluate models
    results = model_trainer.evaluate_models(X_test, y_test)

    # Save models
    saved_models = model_trainer.save_models()

    # Print results
    print("\nTest Results:")
    for model_name, result in results.items():
        if 'test_results' in result:
            print(f"\n{model_name}:")
            for metric, value in result['test_results'].items():
                print(f"  {metric}: {value:.4f}")

    print(f"\nBest model: {model_trainer.select_best_model()}")
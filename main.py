import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import json

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import get_logger
from src.utils.config_loader import get_config
from src.data.data_loader import get_data_loader
from src.data.preprocessor import get_preprocessor
from src.data.feature_engineering import get_feature_engineer
from src.models.model_trainer import get_model_trainer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Customer Churn Prediction Pipeline')

    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')

    parser.add_argument('--data-only', action='store_true',
                        help='Only run data preprocessing and feature engineering')

    parser.add_argument('--train-only', action='store_true',
                        help='Only run model training on preprocessed data')

    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only run model evaluation on trained models')

    parser.add_argument('--feature-engineering', action='store_true',
                        help='Apply advanced feature engineering techniques')

    parser.add_argument('--selected-models', type=str, nargs='+',
                        choices=['logistic_regression', 'random_forest', 'xgboost', 'neural_network'],
                        help='Select specific models to train')

    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save outputs')

    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


def run_data_pipeline(args):
    """Run the data preprocessing and feature engineering pipeline."""
    logger = get_logger()
    config = get_config(args.config)
    data_loader = get_data_loader()
    preprocessor = get_preprocessor()

    logger.info("Starting data pipeline")

    # Load raw data
    df_raw = data_loader.load_raw_data()

    # Preprocess data
    df_processed = preprocessor.preprocess_data(df_raw)

    # Apply feature engineering if requested
    if args.feature_engineering:
        logger.info("Applying advanced feature engineering")
        feature_engineer = get_feature_engineer()
        df_processed = feature_engineer.apply_feature_engineering(
            df_processed,
            include_interactions=True,
            include_polynomials=True,
            include_ratios=True,
            select_features=True,
            n_features=20
        )

    # Save processed data
    output_path = args.output_dir if args.output_dir else config.get_nested_value('data.processed_data_path')

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df_processed.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")

    return df_processed


def run_training_pipeline(args, df_processed=None):
    """Run the model training pipeline."""
    logger = get_logger()
    config = get_config(args.config)
    model_trainer = get_model_trainer()

    logger.info("Starting training pipeline")

    # Load processed data if not provided
    if df_processed is None:
        data_loader = get_data_loader()
        df_processed = data_loader.load_processed_data()

    # Split data
    preprocessor = get_preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.train_test_data_split(df_processed)

    # Configure models based on args
    if args.selected_models:
        for model_type in ['logistic_regression', 'random_forest', 'xgboost', 'neural_network']:
            enabled = model_type in args.selected_models
            model_config = config.get_section('models').get(model_type, {})
            model_config['enabled'] = enabled
            config.update_config({'models': {model_type: model_config}})

    # Train models
    models = model_trainer.train_all_models(X_train, y_train)

    # Evaluate models
    results = model_trainer.evaluate_models(X_test, y_test)

    # Save models
    saved_models = model_trainer.save_models()

    logger.info("Training pipeline completed")

    return models, results, X_test, y_test


def run_evaluation_pipeline(args):
    """Run only the model evaluation pipeline on already trained models."""
    logger = get_logger()
    config = get_config(args.config)

    logger.info("Starting evaluation-only pipeline")

    # Load processed data
    data_loader = get_data_loader()
    df_processed = data_loader.load_processed_data()

    # Split data
    preprocessor = get_preprocessor()
    _, X_test, _, y_test = preprocessor.train_test_data_split(df_processed)

    # Load saved models
    saved_models = {}
    model_dir = config.get_nested_value('model_saving.save_path')

    if os.path.exists(model_dir):
        for model_file in os.listdir(model_dir):
            if model_file.endswith('.pkl'):
                model_name = os.path.splitext(model_file)[0]
                model_path = os.path.join(model_dir, model_file)

                try:
                    logger.info(f"Loading model from {model_path}")
                    model = joblib.load(model_path)
                    saved_models[model_name] = model
                except Exception as e:
                    logger.error(f"Error loading model {model_path}: {e}")

    if not saved_models:
        logger.error("No saved models found for evaluation")
        return None

    # Evaluate models
    results = {}

    for model_name, model in saved_models.items():
        logger.info(f"Evaluating {model_name} on test set")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        test_results = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }

        results[model_name] = {'test_results': test_results}

        # Log the results
        for metric, value in test_results.items():
            logger.info(f"{model_name} test {metric}: {value:.4f}")

    # Save evaluation results
    results_path = config.get_nested_value('model_saving.evaluation_results_path')

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Find best model based on F1 score
    best_model = None
    best_score = -1

    for model_name, result in results.items():
        if 'test_results' in result and 'f1' in result['test_results']:
            f1 = result['test_results']['f1']

            if f1 > best_score:
                best_score = f1
                best_model = model_name

    if best_model:
        logger.info(f"Best model: {best_model} with F1 score: {best_score:.4f}")
        results['best_model'] = best_model

    try:
        logger.info(f"Saving evaluation results to {results_path}")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")

    logger.info("Evaluation pipeline completed")

    return results


def main():
    """Main entry point for the application."""
    # Parse arguments
    args = parse_arguments()

    # Initialize logger with configuration
    logger = get_logger(args.config)

    if args.verbose:
        logger.info("Verbose output enabled")

    # Initialize config with provided path
    config = get_config(args.config)

    try:
        # Run data pipeline only
        if args.data_only:
            logger.info("Running data pipeline only")
            df_processed = run_data_pipeline(args)
            return

        # Run evaluation pipeline only
        if args.evaluate_only:
            logger.info("Running evaluation pipeline only")
            results = run_evaluation_pipeline(args)
            return

        # Run training pipeline only
        if args.train_only:
            logger.info("Running training pipeline only")
            models, results, X_test, y_test = run_training_pipeline(args)
            return

        # Run full pipeline
        logger.info("Running full pipeline")

        # Data pipeline
        df_processed = run_data_pipeline(args)

        # Training pipeline
        models, results, X_test, y_test = run_training_pipeline(args, df_processed)

        logger.info("Full pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
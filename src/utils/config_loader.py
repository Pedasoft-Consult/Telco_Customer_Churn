import yaml
import os
import sys
from src.utils.logger import get_logger


def get_logger_without_config():
    """Simple logger that doesn't depend on configuration."""
    import logging
    logger = logging.getLogger("churn_prediction")
    logger.setLevel(logging.INFO)
    # Add console handler if it doesn't exist
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# Use a logger that doesn't depend on config
logger = get_logger_without_config()


class ConfigLoader:
    """
    Class to load and manage configuration from YAML files.
    """

    def __init__(self, config_path=None):
        """
        Initialize the ConfigLoader with the main configuration file.

        Args:
            config_path (str): Path to the main configuration file.
        """
        # If no path is provided, find the config relative to project root
        if config_path is None:
            current_file = os.path.abspath(__file__)
            utils_dir = os.path.dirname(current_file)
            src_dir = os.path.dirname(utils_dir)
            project_root = os.path.dirname(src_dir)
            config_path = os.path.join(project_root, 'config', 'config.yaml')

        self.config_path = config_path

        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        # Check if config file exists, create default if not
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}, creating default configuration")
            self._create_default_config()

        self.config = self._load_config()

    def _create_default_config(self):
        """Create a default configuration file if one doesn't exist."""
        current_file = os.path.abspath(__file__)
        utils_dir = os.path.dirname(current_file)
        src_dir = os.path.dirname(utils_dir)
        project_root = os.path.dirname(src_dir)

        default_config = {
            'data': {
                'raw_data_path': os.path.join(project_root, 'data', 'raw', 'WA_FnUseC_TelcoCustomerChurn.csv'),
                'processed_data_path': os.path.join(project_root, 'data', 'processed', 'processed_telco_data.csv'),
                'test_size': 0.2,
                'random_state': 42,
                'target_column': 'Churn'
            },
            'features': {
                'categorical_columns': [
                    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod'
                ],
                'numerical_columns': ['tenure', 'MonthlyCharges', 'TotalCharges'],
                'binary_columns': ['SeniorCitizen'],
                'id_column': 'customerID',
                'scaling': 'standard',
                'create_tenure_years': True
            },
            'models': {
                'logistic_regression': {
                    'enabled': True,
                    'hyperparameters': {
                        'C': [0.001, 0.01, 0.1, 1, 10, 100],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear'],
                        'max_iter': [1000]
                    }
                },
                'random_forest': {
                    'enabled': True,
                    'hyperparameters': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 5]
                    }
                },
                'xgboost': {
                    'enabled': True,
                    'hyperparameters': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    }
                }
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                'cross_validation': {
                    'enabled': True,
                    'n_splits': 5,
                    'shuffle': True,
                    'random_state': 42
                },
                'use_smote': False
            },
            'model_saving': {
                'save_path': os.path.join(project_root, 'models', 'trained'),
                'best_model_name': 'best_model.pkl',
                'evaluation_results_path': os.path.join(project_root, 'models', 'evaluation', 'model_comparison.json')
            },
            'logging': {
                'log_file': os.path.join(project_root, 'logs', 'app.log'),
                'log_level': 'INFO',
                'console_log': True
            },
            'webapp': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': True
            }
        }

        # Create necessary directories
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        os.makedirs(os.path.join(project_root, 'data', 'raw'), exist_ok=True)
        os.makedirs(os.path.join(project_root, 'data', 'processed'), exist_ok=True)
        os.makedirs(os.path.join(project_root, 'models', 'trained'), exist_ok=True)
        os.makedirs(os.path.join(project_root, 'models', 'evaluation'), exist_ok=True)
        os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)

        # Write default config
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        logger.info(f"Created default configuration at {self.config_path}")

    def _load_config(self):
        """
        Load configuration from YAML file.

        Returns:
            dict: Configuration dictionary.
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_path}: {e}")
            raise

    def get_config(self):
        """
        Get the entire configuration.

        Returns:
            dict: The complete configuration.
        """
        return self.config

    def get_section(self, section):
        """
        Get a specific section from the configuration.

        Args:
            section (str): The section to retrieve.

        Returns:
            dict: The specified section of the configuration.
        """
        if section in self.config:
            return self.config[section]
        else:
            logger.warning(f"Section '{section}' not found in configuration")
            return {}

    def get_nested_value(self, keys_path):
        """
        Get a nested value from the configuration using a dot-separated path.

        Args:
            keys_path (str): Dot-separated path to the desired value.
                             E.g., 'data.raw_data_path'

        Returns:
            The value at the specified path, or None if not found.
        """
        keys = keys_path.split('.')
        current = self.config

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            logger.warning(f"Path '{keys_path}' not found in configuration")
            return None

    def update_config(self, new_config):
        """
        Update the current configuration with new values.

        Args:
            new_config (dict): New configuration values to update.

        Returns:
            dict: The updated configuration.
        """

        def update_recursively(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_recursively(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.config = update_recursively(self.config, new_config)
        logger.info("Configuration updated successfully")
        return self.config

    def save_config(self, filepath=None):
        """
        Save the current configuration to a YAML file.

        Args:
            filepath (str, optional): Path to save the file.
                                     Defaults to the original config path.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        save_path = filepath or self.config_path

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)

            logger.info(f"Configuration saved successfully to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to {save_path}: {e}")
            return False


# Create a singleton instance for easy access
_config_instance = None


def get_config(config_path=None):
    """
    Get a singleton instance of the configuration.

    Args:
        config_path (str, optional): Path to the configuration file.

    Returns:
        ConfigLoader: A ConfigLoader instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance


if __name__ == "__main__":
    # Test the config loader
    config = get_config()
    print("Data configuration:")
    print(config.get_section('data'))

    print("\nRaw data path:")
    print(config.get_nested_value('data.raw_data_path'))
import logging
import os
import yaml
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime


class Logger:
    """
    Custom logger class for the Customer Churn Prediction project.

    This logger provides customized logging functionality with console and file handlers.
    """

    def __init__(self, config_path=None):
        """
        Initialize the logger with the configuration provided.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.log_file = self.config['logging']['log_file']
        self.log_level = self._get_log_level(self.config['logging']['log_level'])
        self.console_log = self.config['logging']['console_log']

        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # Configure logger
        self.logger = logging.getLogger("churn_prediction")
        self.logger.setLevel(self.log_level)
        self.logger.handlers = []  # Clear existing handlers

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add file handler
        file_handler = RotatingFileHandler(
            self.log_file, maxBytes=10485760, backupCount=10
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Add console handler if enabled
        if self.console_log:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Log initialization
        self.logger.info(f"Logger initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _load_config(self, config_path):
        """
        Load configuration from YAML file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            dict: Configuration dictionary.
        """
        # Find the project root if no config_path is provided
        if config_path is None:
            current_file = os.path.abspath(__file__)
            utils_dir = os.path.dirname(current_file)
            src_dir = os.path.dirname(utils_dir)
            project_root = os.path.dirname(src_dir)
            config_path = os.path.join(project_root, 'config', 'config.yaml')

        try:
            # Check if config directory exists, create if not
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            # Check if config file exists
            if not os.path.exists(config_path):
                # Create default config
                print(f"Warning: Config file not found at {config_path}. Creating default configuration.")
                default_config = {
                    'logging': {
                        'log_file': os.path.join(project_root, 'logs', 'app.log'),
                        'log_level': 'INFO',
                        'console_log': True
                    }
                }

                # Create directory
                os.makedirs(os.path.dirname(config_path), exist_ok=True)

                # Write default config
                with open(config_path, 'w') as file:
                    yaml.dump(default_config, file, default_flow_style=False)

                return default_config

            # Load existing config
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            # Create log directory
            log_file = config.get('logging', {}).get('log_file', os.path.join(project_root, 'logs', 'app.log'))
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            return config
        except Exception as e:
            # Default configuration if file not found
            print(f"Warning: Could not load config file. Using default configuration. Error: {e}")
            return {
                'logging': {
                    'log_file': os.path.join(project_root, 'logs', 'app.log'),
                    'log_level': 'INFO',
                    'console_log': True
                }
            }

    def _get_log_level(self, level_str):
        """
        Convert string log level to logging level.

        Args:
            level_str (str): Log level as string.

        Returns:
            int: Logging level.
        """
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return levels.get(level_str, logging.INFO)

    def get_logger(self):
        """
        Get the configured logger.

        Returns:
            Logger: Configured logger instance.
        """
        return self.logger


# Create a singleton instance
_logger_instance = None


def get_logger(config_path=None):
    """
    Get a configured logger instance.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Logger: Configured logger instance.
    """
    global _logger_instance
    if _logger_instance is None:
        logger_obj = Logger(config_path)
        _logger_instance = logger_obj.get_logger()
    return _logger_instance


if __name__ == "__main__":
    # Test the logger
    logger = get_logger()
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
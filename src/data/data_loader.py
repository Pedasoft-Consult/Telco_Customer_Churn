import pandas as pd
import os
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger()
config = get_config()


class DataLoader:
    """
    Class for loading data from various sources.
    """

    def __init__(self):
        """Initialize the DataLoader with configuration."""
        self.data_config = config.get_section('data')
        self.raw_data_path = self.data_config.get('raw_data_path')
        self.processed_data_path = self.data_config.get('processed_data_path')

    def load_raw_data(self):
        """
        Load the raw telco customer churn data.

        Returns:
            pandas.DataFrame: The loaded raw data.
        """
        try:
            logger.info(f"Loading raw data from {self.raw_data_path}")

            # Check if file exists
            if not os.path.exists(self.raw_data_path):
                logger.error(f"Raw data file not found at {self.raw_data_path}")
                raise FileNotFoundError(f"Data file not found: {self.raw_data_path}")

            # Load CSV data
            df = pd.read_csv(self.raw_data_path)

            logger.info(f"Raw data loaded successfully. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            raise

    def load_processed_data(self):
        """
        Load the processed telco customer churn data if available.

        Returns:
            pandas.DataFrame: The loaded processed data.
        """
        try:
            logger.info(f"Loading processed data from {self.processed_data_path}")

            # Check if file exists
            if not os.path.exists(self.processed_data_path):
                logger.warning(f"Processed data file not found at {self.processed_data_path}")
                logger.info("Loading raw data instead...")
                return self.load_raw_data()

            # Load CSV data
            df = pd.read_csv(self.processed_data_path)

            logger.info(f"Processed data loaded successfully. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise

    def save_processed_data(self, df):
        """
        Save processed data to CSV file.

        Args:
            df (pandas.DataFrame): Processed data to save.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)

            logger.info(f"Saving processed data to {self.processed_data_path}")
            df.to_csv(self.processed_data_path, index=False)

            logger.info(f"Processed data saved successfully. Shape: {df.shape}")
            return True

        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            return False

    def load_data_split(self):
        """
        Load processed data and split into features and target.

        Returns:
            tuple: (X, y) where X is features DataFrame and y is target Series.
        """
        try:
            df = self.load_processed_data()

            # Get target column from config
            target_column = self.data_config.get('target_column', 'Churn')

            # Split data
            if target_column in df.columns:
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Remove ID column if specified in config
                id_column = config.get_nested_value('features.id_column')
                if id_column and id_column in X.columns:
                    X = X.drop(columns=[id_column])

                logger.info(f"Data split into features and target. X shape: {X.shape}, y shape: {y.shape}")
                return X, y
            else:
                logger.error(f"Target column '{target_column}' not found in data")
                raise ValueError(f"Target column '{target_column}' not found in data")

        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise


# Singleton instance for easy access
_data_loader_instance = None


def get_data_loader():
    """
    Get a singleton instance of the DataLoader.

    Returns:
        DataLoader: A DataLoader instance.
    """
    global _data_loader_instance
    if _data_loader_instance is None:
        _data_loader_instance = DataLoader()
    return _data_loader_instance


if __name__ == "__main__":
    # Test the data loader
    data_loader = get_data_loader()

    # Try loading raw data
    df_raw = data_loader.load_raw_data()
    print(f"Raw data shape: {df_raw.shape}")
    print(f"Raw data columns: {df_raw.columns.tolist()}")
    print(f"Raw data head:\n{df_raw.head()}")
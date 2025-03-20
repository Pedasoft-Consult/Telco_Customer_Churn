import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger()
config = get_config()


class Preprocessor:
    """
    Class for preprocessing telco customer churn data.
    """

    def __init__(self):
        """Initialize the Preprocessor with configuration."""
        self.data_config = config.get_section('data')
        self.feature_config = config.get_section('features')
        self.eval_config = config.get_section('evaluation')

        # Get column lists from config
        self.categorical_columns = self.feature_config.get('categorical_columns', [])
        self.numerical_columns = self.feature_config.get('numerical_columns', [])
        self.binary_columns = self.feature_config.get('binary_columns', [])
        self.id_column = self.feature_config.get('id_column')
        self.target_column = self.data_config.get('target_column', 'Churn')

        # Get scaling method
        self.scaling_method = self.feature_config.get('scaling', 'standard')

        # Initialize scalers and imputers
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')

        # Initialize scaler based on config
        if self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

    def preprocess_data(self, df):
        """
        Preprocess the raw telco customer churn data.

        Args:
            df (pandas.DataFrame): Raw data to preprocess.

        Returns:
            pandas.DataFrame: Preprocessed data.
        """
        try:
            logger.info("Starting data preprocessing")

            # Make a copy to avoid modifying the original
            df_processed = df.copy()

            # Handle missing values
            df_processed = self._handle_missing_values(df_processed)

            # Convert data types
            df_processed = self._convert_data_types(df_processed)

            # DIRECT ENCODING OF ALL OBJECT COLUMNS
            # Get all object columns except ID and target
            object_columns = df_processed.select_dtypes(include=['object']).columns
            columns_to_encode = [col for col in object_columns if col != self.id_column and col != self.target_column]

            logger.info(f"Directly encoding {len(columns_to_encode)} object columns: {columns_to_encode}")

            # One-hot encode all categorical columns
            for col in columns_to_encode:
                logger.info(f"One-hot encoding column: {col}")
                dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed.drop(columns=[col], inplace=True)

            # Feature engineering
            if self.feature_config.get('create_tenure_years', False):
                df_processed = self._create_additional_features(df_processed)

            # Scale numerical features
            df_processed = self._scale_numerical_features(df_processed)

            # Encode target variable
            if self.target_column in df_processed.columns and df_processed[self.target_column].dtype == 'object':
                df_processed[self.target_column] = df_processed[self.target_column].map({'Yes': 1, 'No': 0})

            # Final verification of data types
            remaining_objects = df_processed.select_dtypes(include=['object']).columns.tolist()
            if remaining_objects and self.id_column and self.id_column in remaining_objects:
                # Remove ID column from the list if present
                remaining_objects.remove(self.id_column)

            if remaining_objects:
                logger.warning(f"Still found object columns after processing: {remaining_objects}")
                for col in remaining_objects:
                    if col != self.target_column:
                        logger.info(f"Converting remaining object column {col} to numeric using one-hot encoding")
                        dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                        df_processed = pd.concat([df_processed, dummies], axis=1)
                        df_processed.drop(columns=[col], inplace=True)

            logger.info(f"Final data types:\n{df_processed.dtypes}")
            logger.info(f"Data preprocessing completed. Final shape: {df_processed.shape}")
            return df_processed

        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise

    def _handle_missing_values(self, df):
        """
        Handle missing values in the dataset.

        Args:
            df (pandas.DataFrame): DataFrame with potentially missing values.

        Returns:
            pandas.DataFrame: DataFrame with handled missing values.
        """
        logger.info("Handling missing values")

        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0].index.tolist()

        if missing_columns:
            logger.info(f"Found missing values in columns: {missing_columns}")

            # Impute numerical columns
            numerical_missing = [col for col in missing_columns if col in self.numerical_columns]
            if numerical_missing:
                logger.info(f"Imputing numerical missing values in: {numerical_missing}")
                df[numerical_missing] = self.numerical_imputer.fit_transform(df[numerical_missing])

            # Impute categorical columns
            categorical_missing = [col for col in missing_columns if col in self.categorical_columns]
            if categorical_missing:
                logger.info(f"Imputing categorical missing values in: {categorical_missing}")
                df[categorical_missing] = self.categorical_imputer.fit_transform(df[categorical_missing])
        else:
            logger.info("No missing values found")

        return df

    def _convert_data_types(self, df):
        """
        Convert data types to appropriate types.

        Args:
            df (pandas.DataFrame): DataFrame with potentially incorrect types.

        Returns:
            pandas.DataFrame: DataFrame with corrected data types.
        """
        logger.info("Converting data types")

        # Convert TotalCharges from string to float if needed
        if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == 'object':
            logger.info("Converting TotalCharges to numeric")
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

            # Impute any resulting NaN values
            if df['TotalCharges'].isnull().sum() > 0:
                logger.info(f"Imputing {df['TotalCharges'].isnull().sum()} NaN values in TotalCharges")
                df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])

        # Convert binary columns to integers
        for col in self.binary_columns:
            if col in df.columns:
                logger.info(f"Converting binary column {col} to integer")
                df[col] = df[col].astype(int)

        return df

    def _create_additional_features(self, df):
        """
        Create additional features from existing data.

        Args:
            df (pandas.DataFrame): Original DataFrame.

        Returns:
            pandas.DataFrame: DataFrame with additional features.
        """
        logger.info("Creating additional features")

        # Create tenure_years from tenure (months)
        if 'tenure' in df.columns:
            logger.info("Creating tenure_years feature")
            df['tenure_years'] = df['tenure'] / 12.0

        # Create total services count
        # Look for encoded versions of service columns
        service_column_prefixes = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

        # Find any columns that start with these prefixes and end with a value (indicating one-hot encoding)
        service_cols = []
        for col in df.columns:
            for prefix in service_column_prefixes:
                if col.startswith(f"{prefix}_") and col.count('_') == 1:
                    service_cols.append(col)
                    break

        if service_cols:
            logger.info(f"Creating total_services feature from {len(service_cols)} encoded service columns")
            df['total_services'] = df[service_cols].sum(axis=1)

        return df

    def _scale_numerical_features(self, df):
        """
        Scale numerical features using the configured scaler.

        Args:
            df (pandas.DataFrame): DataFrame with numerical features.

        Returns:
            pandas.DataFrame: DataFrame with scaled numerical features.
        """
        logger.info(f"Scaling numerical features using {self.scaling_method} scaler")

        # Get numerical columns that exist in the dataframe
        num_cols = [col for col in self.numerical_columns if col in df.columns]

        if num_cols:
            # Make sure all numerical columns are actually numeric type
            for col in num_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    logger.warning(f"Converting numerical column {col} from {df[col].dtype} to numeric")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col].fillna(df[col].median(), inplace=True)

            # Scale the values
            scaled_values = self.scaler.fit_transform(df[num_cols])

            # Update the dataframe with scaled values
            df[num_cols] = scaled_values

            logger.info(f"Scaled {len(num_cols)} numerical features")
        else:
            logger.warning("No numerical features found for scaling")

        return df

    def train_test_data_split(self, df):
        """
        Split the preprocessed data into training and testing sets.

        Args:
            df (pandas.DataFrame): Preprocessed data.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train and test sets")

        # Make a copy to avoid modifying the original
        df_copy = df.copy()

        # Remove ID column if it exists
        if self.id_column in df_copy.columns:
            df_copy = df_copy.drop(columns=[self.id_column])

        # EMERGENCY FIX: Force encoding of any remaining object columns
        object_columns = df_copy.select_dtypes(include=['object']).columns.tolist()
        columns_to_encode = [col for col in object_columns if col != self.target_column]

        if columns_to_encode:
            logger.warning(f"Found unencoded columns before split: {columns_to_encode}")
            logger.info("Applying emergency encoding to convert all columns to numeric")

            # Force encode all remaining object columns except target
            for col in columns_to_encode:
                logger.info(f"Force encoding column: {col}")
                if df_copy[col].nunique() <= 2:  # Binary column
                    # Map to 0/1 for binary columns
                    unique_vals = df_copy[col].unique()
                    df_copy[col] = df_copy[col].map({unique_vals[0]: 0, unique_vals[1]: 1})
                    logger.info(f"Binary encoded {col}: {unique_vals[0]}=0, {unique_vals[1]}=1")
                else:
                    # One-hot encoding for multiple categories
                    dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
                    df_copy = pd.concat([df_copy, dummies], axis=1)
                    df_copy.drop(columns=[col], inplace=True)
                    logger.info(f"One-hot encoded {col}")

        # Check for 'TotalCharges' and ensure it's numeric
        if 'TotalCharges' in df_copy.columns and not pd.api.types.is_numeric_dtype(df_copy['TotalCharges']):
            logger.warning("TotalCharges is not numeric, converting...")
            df_copy['TotalCharges'] = pd.to_numeric(df_copy['TotalCharges'], errors='coerce')
            df_copy['TotalCharges'].fillna(df_copy['TotalCharges'].median(), inplace=True)

        # Split into features and target
        if self.target_column in df_copy.columns:
            # If target is still an object, encode it
            if pd.api.types.is_object_dtype(df_copy[self.target_column]):
                logger.info(f"Converting target column {self.target_column} to numeric")
                df_copy[self.target_column] = df_copy[self.target_column].map({'Yes': 1, 'No': 0})

            X = df_copy.drop(columns=[self.target_column])
            y = df_copy[self.target_column]

            # Final check - verify all data is numeric
            non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
            if non_numeric_cols:
                logger.error(f"Still found non-numeric columns after emergency encoding: {non_numeric_cols}")
                # Instead of raising an error, force convert these columns
                for col in non_numeric_cols:
                    logger.warning(f"Force dropping non-numeric column: {col}")
                    X = X.drop(columns=[col])

            # Get configuration values
            test_size = self.data_config.get('test_size', 0.2)
            random_state = self.data_config.get('random_state', 42)

            # Perform the split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            logger.info(f"Data split completed. Train size: {X_train.shape}, Test size: {X_test.shape}")

            # Apply SMOTE if configured
            if self.eval_config.get('use_smote', False):
                logger.info("Applying SMOTE to handle class imbalance")
                smote = SMOTE(random_state=random_state)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info(f"After SMOTE - Train size: {X_train.shape}")

            return X_train, X_test, y_train, y_test
        else:
            logger.error(f"Target column '{self.target_column}' not found in data")
            raise ValueError(f"Target column '{self.target_column}' not found in data")


# Singleton instance for easy access
_preprocessor_instance = None


def get_preprocessor():
    """
    Get a singleton instance of the Preprocessor.

    Returns:
        Preprocessor: A Preprocessor instance.
    """
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = Preprocessor()
    return _preprocessor_instance


if __name__ == "__main__":
    # Test the preprocessor
    from src.data.data_loader import get_data_loader

    data_loader = get_data_loader()
    preprocessor = get_preprocessor()

    # Load raw data
    df_raw = data_loader.load_raw_data()

    # Preprocess data
    df_processed = preprocessor.preprocess_data(df_raw)

    # Save processed data
    data_loader.save_processed_data(df_processed)

    # Split data
    X_train, X_test, y_train, y_test = preprocessor.train_test_data_split(df_processed)

    print(f"Processed data shape: {df_processed.shape}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
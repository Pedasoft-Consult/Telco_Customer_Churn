import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.decomposition import PCA

from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger()
config = get_config()


class FeatureEngineer:
    """
    Class for feature engineering on the telco customer churn dataset.
    """

    def __init__(self):
        """Initialize the FeatureEngineer with configuration."""
        self.feature_config = config.get_section('features')
        self.data_config = config.get_section('data')

        # Get column lists from config
        self.categorical_columns = self.feature_config.get('categorical_columns', [])
        self.numerical_columns = self.feature_config.get('numerical_columns', [])
        self.binary_columns = self.feature_config.get('binary_columns', [])
        self.id_column = self.feature_config.get('id_column')
        self.target_column = self.data_config.get('target_column', 'Churn')

    def create_interaction_features(self, df):
        """
        Create interaction features between numerical variables.

        Args:
            df (pandas.DataFrame): Input DataFrame.

        Returns:
            pandas.DataFrame: DataFrame with interaction features.
        """
        logger.info("Creating interaction features between numerical variables")

        # Get numerical columns that exist in the dataframe
        num_cols = [col for col in self.numerical_columns if col in df.columns]

        if len(num_cols) >= 2:
            # Create a copy of the dataframe
            df_with_interactions = df.copy()

            # Create pairwise interactions
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    col1 = num_cols[i]
                    col2 = num_cols[j]

                    # Create product feature
                    interaction_name = f"{col1}_x_{col2}"
                    df_with_interactions[interaction_name] = df[col1] * df[col2]

                    logger.info(f"Created interaction feature: {interaction_name}")

            return df_with_interactions
        else:
            logger.warning("Not enough numerical columns for interaction features")
            return df

    def create_polynomial_features(self, df, degree=2):
        """
        Create polynomial features for numerical variables.

        Args:
            df (pandas.DataFrame): Input DataFrame.
            degree (int): Degree of polynomial features.

        Returns:
            pandas.DataFrame: DataFrame with polynomial features.
        """
        logger.info(f"Creating polynomial features of degree {degree}")

        # Get numerical columns that exist in the dataframe
        num_cols = [col for col in self.numerical_columns if col in df.columns]

        if num_cols:
            # Create a copy of the dataframe
            df_with_poly = df.copy()

            # Initialize polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)

            # Fit and transform
            poly_features = poly.fit_transform(df[num_cols])

            # Get feature names
            poly_features_names = poly.get_feature_names_out(num_cols)

            # Add polynomial features to dataframe
            for i, name in enumerate(poly_features_names):
                # Skip the original features (they're already in the dataframe)
                if name in num_cols:
                    continue

                # Add the polynomial feature
                df_with_poly[name] = poly_features[:, i]

                logger.info(f"Created polynomial feature: {name}")

            return df_with_poly
        else:
            logger.warning("No numerical columns for polynomial features")
            return df

    def create_ratio_features(self, df):
        """
        Create ratio features between numerical variables.

        Args:
            df (pandas.DataFrame): Input DataFrame.

        Returns:
            pandas.DataFrame: DataFrame with ratio features.
        """
        logger.info("Creating ratio features between numerical variables")

        # Get numerical columns that exist in the dataframe
        num_cols = [col for col in self.numerical_columns if col in df.columns]

        if len(num_cols) >= 2:
            # Create a copy of the dataframe
            df_with_ratios = df.copy()

            # Create pairwise ratios
            for i in range(len(num_cols)):
                for j in range(len(num_cols)):
                    if i != j:  # Don't divide a column by itself
                        col1 = num_cols[i]
                        col2 = num_cols[j]

                        # Create ratio feature with safe division
                        ratio_name = f"{col1}_div_{col2}"
                        # Avoid division by zero
                        df_with_ratios[ratio_name] = df[col1] / (df[col2] + 1e-10)

                        logger.info(f"Created ratio feature: {ratio_name}")

            return df_with_ratios
        else:
            logger.warning("Not enough numerical columns for ratio features")
            return df

    def select_best_features(self, df, X, y, k=10):
        """
        Select the k best features based on statistical tests.

        Args:
            df (pandas.DataFrame): Original DataFrame.
            X (pandas.DataFrame): Features.
            y (pandas.Series): Target.
            k (int): Number of features to select.

        Returns:
            pandas.DataFrame: DataFrame with selected features.
        """
        logger.info(f"Selecting {k} best features using statistical tests")

        # Ensure k is not larger than the number of features
        k = min(k, X.shape[1])

        # Initialize feature selector
        selector = SelectKBest(score_func=f_classif, k=k)

        # Fit and transform
        X_selected = selector.fit_transform(X, y)

        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)

        # Get selected feature names
        selected_features = X.columns[selected_indices].tolist()

        logger.info(f"Selected features: {selected_features}")

        # Create a new dataframe with selected features and target
        selected_columns = selected_features.copy()
        if self.target_column in df.columns:
            selected_columns.append(self.target_column)

        # Add ID column if required
        if self.id_column and self.id_column in df.columns:
            selected_columns.append(self.id_column)

        df_selected = df[selected_columns]

        return df_selected

    def create_domain_specific_features(self, df):
        """
        Create domain-specific features for telecommunication customer churn.

        Args:
            df (pandas.DataFrame): Input DataFrame.

        Returns:
            pandas.DataFrame: DataFrame with domain-specific features.
        """
        logger.info("Creating domain-specific features for telco customer churn")

        # Create a copy of the dataframe
        df_with_domain = df.copy()

        # 1. Create a feature for number of services
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

        # Count services only if the columns exist and have been encoded
        existing_binary_service_cols = []
        for col in service_columns:
            # Check for original column or its encoded versions
            if col in df.columns:
                existing_binary_service_cols.append(col)
            else:
                # Look for one-hot encoded versions like "PhoneService_Yes"
                encoded_col = f"{col}_Yes"
                if encoded_col in df.columns:
                    existing_binary_service_cols.append(encoded_col)

        if existing_binary_service_cols:
            logger.info(f"Creating total_services feature from {len(existing_binary_service_cols)} service columns")
            df_with_domain['total_services'] = df[existing_binary_service_cols].sum(axis=1)

        # 2. Calculate average monthly cost per service
        if 'total_services' in df_with_domain.columns and 'MonthlyCharges' in df.columns:
            logger.info("Creating cost_per_service feature")
            # Avoid division by zero
            df_with_domain['cost_per_service'] = df['MonthlyCharges'] / (df_with_domain['total_services'] + 0.1)

        # 3. Create loyalty indicator (based on tenure and contract type)
        if 'tenure' in df.columns:
            logger.info("Creating loyalty_score feature")

            # Base loyalty on tenure
            df_with_domain['loyalty_score'] = df['tenure']

            # Adjust based on contract type if available
            if 'Contract_One year' in df.columns and 'Contract_Two year' in df.columns:
                # Add bonus for longer contracts
                df_with_domain['loyalty_score'] = df_with_domain['loyalty_score'] + \
                                                  df['Contract_One year'] * 12 + \
                                                  df['Contract_Two year'] * 24

            # Scale to 0-100 range for easier interpretation
            max_loyalty = df_with_domain['loyalty_score'].max()
            if max_loyalty > 0:
                df_with_domain['loyalty_score'] = 100 * df_with_domain['loyalty_score'] / max_loyalty

        # 4. Engagement level based on streaming services
        streaming_cols = ['StreamingTV_Yes', 'StreamingMovies_Yes']
        existing_streaming = [col for col in streaming_cols if col in df.columns]

        if existing_streaming:
            logger.info("Creating engagement_level feature")
            df_with_domain['streaming_engagement'] = df[existing_streaming].sum(axis=1)

        # 5. Risk factor based on payment method and billing
        payment_risk_cols = ['PaymentMethod_Electronic check', 'PaperlessBilling_Yes']
        existing_risk_cols = [col for col in payment_risk_cols if col in df.columns]

        if existing_risk_cols:
            logger.info("Creating payment_risk_factor feature")

            # Initialize risk factor
            df_with_domain['payment_risk_factor'] = 0

            # Increase risk for electronic check users (more likely to churn)
            if 'PaymentMethod_Electronic check' in existing_risk_cols:
                df_with_domain['payment_risk_factor'] += df['PaymentMethod_Electronic check']

            # Adjust based on paperless billing
            if 'PaperlessBilling_Yes' in existing_risk_cols:
                df_with_domain['payment_risk_factor'] += 0.5 * df['PaperlessBilling_Yes']

        return df_with_domain

    def apply_feature_engineering(self, df, include_interactions=False,
                                  include_polynomials=False, include_ratios=False,
                                  select_features=False, n_features=10):
        """
        Apply a series of feature engineering techniques to the data.

        Args:
            df (pandas.DataFrame): Input DataFrame.
            include_interactions (bool): Whether to include interaction features.
            include_polynomials (bool): Whether to include polynomial features.
            include_ratios (bool): Whether to include ratio features.
            select_features (bool): Whether to select best features.
            n_features (int): Number of features to select if select_features is True.

        Returns:
            pandas.DataFrame: DataFrame with engineered features.
        """
        logger.info("Applying feature engineering techniques")

        # Create domain-specific features
        df_engineered = self.create_domain_specific_features(df)

        # Create interaction features if enabled
        if include_interactions:
            df_engineered = self.create_interaction_features(df_engineered)

        # Create polynomial features if enabled
        if include_polynomials:
            df_engineered = self.create_polynomial_features(df_engineered, degree=2)

        # Create ratio features if enabled
        if include_ratios:
            df_engineered = self.create_ratio_features(df_engineered)

        # Select best features if enabled
        if select_features:
            # Split into features and target
            if self.target_column in df_engineered.columns:
                X = df_engineered.drop(columns=[self.target_column])
                if self.id_column and self.id_column in X.columns:
                    X = X.drop(columns=[self.id_column])
                y = df_engineered[self.target_column]

                df_engineered = self.select_best_features(df_engineered, X, y, k=n_features)

        logger.info(f"Feature engineering completed. Final shape: {df_engineered.shape}")
        return df_engineered


# Singleton instance for easy access
_feature_engineer_instance = None


def get_feature_engineer():
    """
    Get a singleton instance of the FeatureEngineer.

    Returns:
        FeatureEngineer: A FeatureEngineer instance.
    """
    global _feature_engineer_instance
    if _feature_engineer_instance is None:
        _feature_engineer_instance = FeatureEngineer()
    return _feature_engineer_instance


if __name__ == "__main__":
    # Test the feature engineer
    from src.data.data_loader import get_data_loader
    from src.data.preprocessor import get_preprocessor

    # Load and preprocess data
    data_loader = get_data_loader()
    preprocessor = get_preprocessor()

    df_raw = data_loader.load_raw_data()
    df_processed = preprocessor.preprocess_data(df_raw)

    # Apply feature engineering
    feature_engineer = get_feature_engineer()
    df_engineered = feature_engineer.apply_feature_engineering(
        df_processed,
        include_interactions=True,
        include_polynomials=True,
        include_ratios=False,
        select_features=True,
        n_features=15
    )

    print(f"Original shape: {df_processed.shape}")
    print(f"Engineered shape: {df_engineered.shape}")
    print(f"Engineered columns: {df_engineered.columns.tolist()}")
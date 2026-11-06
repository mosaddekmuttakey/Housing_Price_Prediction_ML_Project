"""Feature engineering utilities for the California Housing project."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Impute total bedrooms and add household-level ratio features.

    The median is learned during fit(), so cross-validation and the final
    train/test workflow do not use information from held-out observations.
    """

    engineered_features = (
        "rooms_per_household",
        "bedrooms_per_room",
        "population_per_household",
    )

    def fit(self, X, y=None):
        """Learn the training-set median for total_bedrooms."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if "total_bedrooms" not in X.columns:
            raise ValueError("Input data must contain 'total_bedrooms'.")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        bedrooms = pd.to_numeric(X["total_bedrooms"], errors="coerce")
        self.total_bedrooms_median_ = float(bedrooms.median())

        if not np.isfinite(self.total_bedrooms_median_):
            raise ValueError("Cannot calculate a valid median for total_bedrooms.")

        return self

    def transform(self, X):
        """Impute bedrooms and calculate the three original engineered features."""
        check_is_fitted(self, attributes=["total_bedrooms_median_", "feature_names_in_"])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        transformed = X.copy()

        transformed["total_bedrooms"] = (
            pd.to_numeric(transformed["total_bedrooms"], errors="coerce")
            .fillna(self.total_bedrooms_median_)
        )

        # Replace zero denominators with NaN; the downstream numeric imputer
        # handles any resulting missing ratio values safely.
        households = pd.to_numeric(transformed["households"], errors="coerce").replace(0, np.nan)
        rooms = pd.to_numeric(transformed["total_rooms"], errors="coerce").replace(0, np.nan)

        transformed["rooms_per_household"] = (
            pd.to_numeric(transformed["total_rooms"], errors="coerce") / households
        )
        transformed["bedrooms_per_room"] = (
            transformed["total_bedrooms"] / rooms
        )
        transformed["population_per_household"] = (
            pd.to_numeric(transformed["population"], errors="coerce") / households
        )

        return transformed

    def get_feature_names_out(self, input_features=None):
        """Return original and engineered feature names."""
        check_is_fitted(self, attributes=["feature_names_in_"])

        if input_features is None:
            input_features = self.feature_names_in_

        return np.asarray(
            list(input_features) + list(self.engineered_features),
            dtype=object,
        )

"""Column transformer: scaling for continuous, passthrough for binary/ordinal."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from models.config import CONTINUOUS_FEATURES, BINARY_FEATURES, ORDINAL_FEATURES


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer that scales continuous features
    and passes through binary/ordinal features.

    Only includes columns actually present in X.
    """
    available = set(X.columns)

    cont = [c for c in CONTINUOUS_FEATURES if c in available]
    binary_ord = [c for c in BINARY_FEATURES + ORDINAL_FEATURES if c in available]

    transformers = []
    if cont:
        transformers.append(("scaler", StandardScaler(), cont))
    if binary_ord:
        transformers.append(("passthrough", "passthrough", binary_ord))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

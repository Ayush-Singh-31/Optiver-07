# preprocessing_pipeline.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# -------------------
# Feature Engineering Transformer
# -------------------

class MakeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Feature engineering
        df['mid_price'] = (df['bid_price1'] + df['ask_price1']) / 2
        df['spread'] = df['ask_price1'] - df['bid_price1']
        df['rel_spread'] = df['spread'] / df['mid_price']
        df['imbalance'] = (df['bid_size1'] - df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
        df['book_pressure'] = ((df['bid_size1'] + df['bid_size2']) - (df['ask_size1'] + df['ask_size2'])) / (df['bid_size1'] + df['bid_size2'] + df['ask_size1'] + df['ask_size2'])
        df['microprice'] = ((df['ask_price1'] * df['bid_size1'] + df['bid_price1'] * df['ask_size1']) / (df['bid_size1'] + df['ask_size1']))
        df['normalized_spread'] = df['spread'] / df['mid_price']
        df['OBI_L2'] = (df['bid_size1'] + df['bid_size2']) / (df['bid_size1'] + df['bid_size2'] + df['ask_size1'] + df['ask_size2'])

        # Order book entropy features
        sizes = df[['bid_size1', 'bid_size2', 'ask_size1', 'ask_size2']].astype(float).values
        total = sizes.sum(axis=1, keepdims=True)
        p = np.divide(sizes, total, where=total != 0)
        entropy = -np.nansum(np.where(p > 0, p * np.log(p, where=p > 0), 0), axis=1)
        df['LOB_entropy'] = entropy
        df['LOB_entropy_normalized'] = entropy / np.log(4)

        # Price return + volatility related features
        df['log_return'] = df.groupby('time_id')['mid_price'].transform(lambda x: np.log(x / x.shift(1)))
        df['realized_volatility'] = df.groupby('time_id')['log_return'].transform(lambda x: np.sqrt(x.pow(2).rolling(window=60, min_periods=1).sum()))
        df['bipower_var'] = df.groupby('time_id')['log_return'].transform(
            lambda x: x.abs().rolling(2).apply(lambda r: r[0] * r[1], raw=True).rolling(60, min_periods=1).mean()
        )
        df['rolling_integrated_variance'] = df.groupby('time_id')['log_return'].transform(
            lambda x: x.pow(2).rolling(window=60, min_periods=1).sum()
        )

        return df

# -------------------
# Reindex and Forward Fill Transformer
# -------------------

class ReindexFill(BaseEstimator, TransformerMixin):
    def __init__(self, n_seconds=600):
        self.n_seconds = n_seconds

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Build full time_id × seconds_in_bucket index
        time_ids = df['time_id'].unique()
        full_index = pd.MultiIndex.from_product(
            [time_ids, range(self.n_seconds)],
            names=['time_id', 'seconds_in_bucket']
        )
        df = df.set_index(['time_id', 'seconds_in_bucket']).reindex(full_index)

        # Forward fill columns groupby time_id
        cols_to_fill = [c for c in df.columns if c not in ['time_id', 'seconds_in_bucket']]
        df[cols_to_fill] = df.groupby(level=0)[cols_to_fill].ffill()

        return df.reset_index()

# -------------------
# Handle Stock ID Mapping and Drop Any Stock ID Leftovers
# -------------------

class HandleStockID(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        if 'stock_id' in df.columns:
            mapping = (
                df.dropna(subset=['stock_id'])
                .drop_duplicates(subset=['time_id'])
                .set_index('time_id')['stock_id']
            )
            df['stock_id'] = df['time_id'].map(mapping)
            df = df.drop('stock_id', axis=1)

        df = df.dropna()

        return df

# -------------------
# Remap time_id to Sequential Integers
# -------------------

class MapTimeID(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        unique_time_ids = sorted(df['time_id'].unique())
        time_id_map = {old: new for new, old in enumerate(unique_time_ids, start=1)}
        df['time_id'] = df['time_id'].map(time_id_map)

        return df

# -------------------
# Assemble the Full Pipeline
# -------------------

pipeline = Pipeline([
    ('make_features', MakeFeatures()),
    ('reindex_fill', ReindexFill()),
    ('handle_stock_id', HandleStockID()),
    ('map_time_id', MapTimeID()),
])

"""Multi-method feature selection.

The design matrix contains many near-duplicates: nine indoor temperature sensors in one house
move together. Handing all of them to a recurrent network multiplies parameters without adding
information.

Each individual selector has a known blind spot - tree importance splits credit arbitrarily
among correlated columns, RFE assumes a linear relationship, univariate correlation ignores
interactions entirely - so a feature is kept only when at least ``min_votes`` of the three agree
on it. All three are fitted on training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

from energy_forecast.config import Config
from energy_forecast.exceptions import ConfigurationError
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of the vote.

    Attributes:
        selected: Names of the retained features, in a stable order.
        votes: Vote count per feature.
        importances: Random-forest importance per feature, for reporting.
        indices: Positions of the selected features in the original column order, so an
            already-scaled array can be sliced without being rebuilt.
    """

    selected: List[str]
    votes: pd.Series
    importances: pd.Series
    indices: List[int]

    @property
    def table(self) -> pd.DataFrame:
        """Per-feature detail, sorted by importance."""
        return pd.DataFrame({
            "votes": self.votes,
            "rf_importance": self.importances,
            "selected": self.votes.index.isin(self.selected),
        }).sort_values("rf_importance", ascending=False)


def select_features(X_train: pd.DataFrame, y_train_scaled: np.ndarray,
                    y_train_raw: pd.Series, config: Config) -> SelectionResult:
    """Run the three selectors and take a majority vote.

    Args:
        X_train: Unscaled training design matrix, used for the correlation screen and to
            recover column names.
        y_train_scaled: Training target in model space, used by the model-based selectors.
        y_train_raw: Training target in Wh, used for the correlation screen.
        config: Loaded configuration.

    Returns:
        A :class:`SelectionResult`.
    """
    settings = config.selection
    random_state = config.random_state

    forest = RandomForestRegressor(n_estimators=200, max_depth=18, min_samples_leaf=3,
                                   n_jobs=-1, random_state=random_state)
    forest.fit(X_train.values, y_train_scaled)
    importances = pd.Series(forest.feature_importances_, index=X_train.columns)

    rfe = RFE(Ridge(alpha=1.0), n_features_to_select=int(settings["n_features_rfe"]), step=5)
    rfe.fit(X_train.values, y_train_scaled)
    rfe_keep = set(X_train.columns[rfe.support_])

    correlation = X_train.corrwith(y_train_raw).abs()

    votes = pd.Series(0, index=X_train.columns, dtype=int)
    votes[importances.nlargest(int(settings["top_k_importance"])).index] += 1
    votes[list(rfe_keep)] += 1
    votes[correlation.nlargest(int(settings["top_k_correlation"])).index] += 1

    selected = sorted(votes[votes >= int(settings["min_votes"])].index)
    if not selected:
        raise ConfigurationError(
            "feature selection retained nothing; lower selection.min_votes")

    indices = [X_train.columns.get_loc(name) for name in selected]
    logger.info("feature selection: %s of %s features retained by majority vote",
                len(selected), X_train.shape[1])
    return SelectionResult(selected=selected, votes=votes, importances=importances,
                           indices=indices)

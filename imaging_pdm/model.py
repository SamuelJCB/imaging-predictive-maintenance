"""Failure timing (survival) and root-cause models, plus leave-one-site-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

from .features import BASE_FEATURES, SPATIAL_FEATURES, attach_spatial, site_failure_rates
from .fleet import MODES

ALERT_HORIZON_DAYS = 14

AFT_PARAMS = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution="normal",
                  aft_loss_distribution_scale=1.0, tree_method="hist", learning_rate=0.05, max_depth=4,
                  min_child_weight=20, subsample=0.8, colsample_bynode=0.8, seed=0)


class TimingModel:
    """XGBoost accelerated failure time model. Right-censored rows have an infinite upper bound."""

    def __init__(self, features: list[str], rounds: int = 400):
        self.features, self.rounds = features, rounds

    def fit(self, df: pd.DataFrame):
        # XGBoost's AFT objective takes bounds on the raw time scale and models log(time) internally.
        y = df["days_to_failure"].to_numpy(float)
        dtrain = xgb.DMatrix(df[self.features])
        dtrain.set_float_info("label_lower_bound", y)
        dtrain.set_float_info("label_upper_bound", np.where(df["event"] == 1, y, np.inf))
        self.booster = xgb.train(AFT_PARAMS, dtrain, num_boost_round=self.rounds)
        return self

    def predict_days(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(df[self.features]))


class RootCauseModel:
    """Predicts which component will fail, trained on the window just before observed failures."""

    def __init__(self, features: list[str]):
        self.features = features
        self.classes = list(MODES)

    def fit(self, df: pd.DataFrame):
        y = df["next_mode"].map({m: i for i, m in enumerate(self.classes)})
        self.model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                       colsample_bytree=0.8, objective="multi:softprob",
                                       num_class=len(self.classes), verbosity=0)
        self.model.fit(df[self.features], y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(self.classes)[self.model.predict_proba(df[self.features]).argmax(axis=1)]


def concordance_index(time: np.ndarray, pred: np.ndarray, event: np.ndarray, max_pairs: int = 400_000,
                      seed: int = 0) -> float:
    """Harrell's C on a random sample of comparable pairs (shorter observed failure vs longer time)."""
    rng = np.random.default_rng(seed)
    n = len(time)
    i, j = rng.integers(0, n, max_pairs), rng.integers(0, n, max_pairs)
    comparable = (event[i] == 1) & (time[i] < time[j])
    i, j = i[comparable], j[comparable]
    return float(((pred[i] < pred[j]) + 0.5 * (pred[i] == pred[j])).mean())


@dataclass
class FoldResult:
    site: str
    spatial: bool
    c_index: float
    alert_precision: float
    alert_recall: float
    alert_lift: float
    root_cause_accuracy: float
    root_cause_macro_f1: float


ALERT_BUDGET = 0.05  # engineers can inspect the riskiest 5% of machine-days


def _alert_metrics(df: pd.DataFrame, pred_days: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall, and lift when alerting on the shortest predicted times within the budget."""
    actual = ((df["event"] == 1) & (df["days_to_failure"] <= ALERT_HORIZON_DAYS)).to_numpy()
    flagged = pred_days <= np.quantile(pred_days, ALERT_BUDGET)
    tp = float((actual & flagged).sum())
    precision = tp / max(flagged.sum(), 1)
    recall = tp / max(actual.sum(), 1)
    return precision, recall, precision / max(actual.mean(), 1e-9)


def leave_one_site_out(data: pd.DataFrame, spatial: bool, sample_every: int = 2) -> list[FoldResult]:
    """Train on six sites, test on the seventh, for every site.

    This is the deployment question that matters for a new hospital: how well
    does the model work at a site it has never seen?
    """
    features = BASE_FEATURES + (SPATIAL_FEATURES if spatial else [])
    results = []
    for site in sorted(data["site"].unique()):
        train, test = data[data["site"] != site], data[data["site"] == site]
        if spatial:
            rates = site_failure_rates(train)  # held-out site's own failures are never used
            train, test = attach_spatial(train, rates), attach_spatial(test, rates)
        train = train.iloc[::sample_every]

        timing = TimingModel(features).fit(train)
        pred = timing.predict_days(test)
        precision, recall, lift = _alert_metrics(test, pred)

        near = lambda d: d[(d["event"] == 1) & (d["days_to_failure"] <= ALERT_HORIZON_DAYS)]
        rc_train, rc_test = near(train), near(test)
        rc = RootCauseModel(features).fit(rc_train)
        rc_pred = rc.predict(rc_test)

        results.append(FoldResult(
            site=site, spatial=spatial,
            c_index=concordance_index(test["days_to_failure"].to_numpy(), pred, test["event"].to_numpy()),
            alert_precision=precision, alert_recall=recall, alert_lift=lift,
            root_cause_accuracy=accuracy_score(rc_test["next_mode"], rc_pred),
            root_cause_macro_f1=f1_score(rc_test["next_mode"], rc_pred, average="macro"),
        ))
    return results


def summarize(results: list[FoldResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in results])
    return df.drop(columns="site").groupby("spatial").mean().round(3)

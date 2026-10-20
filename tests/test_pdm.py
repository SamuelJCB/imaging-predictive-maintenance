import numpy as np
import pandas as pd
import pytest

from imaging_pdm.features import (BASE_FEATURES, SPATIAL_FEATURES, add_labels, attach_spatial, build_dataset,
                                  haversine_km, site_failure_rates, spatial_features)
from imaging_pdm.fleet import MODALITY_MODES, generate_fleet
from imaging_pdm.model import RootCauseModel, TimingModel, concordance_index, leave_one_site_out


@pytest.fixture(scope="module")
def fleet():
    return generate_fleet(days=500, machines_per_site=4, seed=1)


@pytest.fixture(scope="module")
def data(fleet):
    return build_dataset(fleet)


def test_failure_modes_match_modality(fleet):
    failed = fleet.dropna(subset=["failure_mode"])
    assert len(failed) > 0
    for modality, modes in MODALITY_MODES.items():
        assert set(failed.loc[failed["modality"] == modality, "failure_mode"]) <= set(modes)


def test_labels_count_down_to_next_failure():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=6), "machine_id": "M1",
        "failure_mode": [None, None, "cooling", None, None, None],
    })
    out = add_labels(df)
    assert out["days_to_failure"].tolist()[:2] == [2, 1]
    assert out["next_mode"].tolist()[:2] == ["cooling", "cooling"]
    assert out["event"].tolist() == [1, 1, 0, 0, 0, 0]  # nothing observed after the failure: censored


def test_haversine_known_distance():
    assert haversine_km(19.08, 72.88, 28.61, 77.21) == pytest.approx(1150, rel=0.05)  # Mumbai-Delhi


def test_spatial_features_exclude_own_site(fleet):
    rates = site_failure_rates(fleet)
    rates.loc["Chennai", "cooling"] = 1e6  # if Chennai leaked into its own feature, this would dominate
    f = spatial_features(rates, rates.loc["Chennai", "lat"], rates.loc["Chennai", "lon"], exclude_site="Chennai")
    assert f["spatial_rate_cooling"] < 100


def test_concordance_index_bounds():
    t = np.arange(1, 101, dtype=float)
    e = np.ones(100, dtype=int)
    assert concordance_index(t, t, e) == pytest.approx(1.0)
    assert concordance_index(t, -t, e) == pytest.approx(0.0)


def test_models_fit_and_predict(data):
    d = attach_spatial(data, site_failure_rates(data))
    feats = BASE_FEATURES + SPATIAL_FEATURES
    timing = TimingModel(feats, rounds=50).fit(d.iloc[::3])
    assert (timing.predict_days(d.head(50)) > 0).all()
    near = d[(d["event"] == 1) & (d["days_to_failure"] <= 14)]
    preds = RootCauseModel(feats).fit(near).predict(near.head(20))
    assert len(preds) == 20


def test_leave_one_site_out_beats_random(data):
    results = leave_one_site_out(data, spatial=True, sample_every=3)
    assert len(results) == data["site"].nunique()
    assert np.mean([r.c_index for r in results]) > 0.6

"""Labels, rolling telemetry features, and the spatial layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fleet import MODES

SENSORS = ["ambient_temp_c", "humidity_pct", "voltage_sag_events", "scans", "tube_arc_count",
           "coolant_delta_t_c", "psu_ripple_mv", "helium_pct", "cold_head_temp_k", "gradient_error_count"]
BASE_FEATURES = (
    ["is_mri", "age_years", "days_since_service"]
    + [f"{s}_mean7" for s in SENSORS]
    + [f"{s}_mean30" for s in SENSORS]
    + [f"{s}_trend" for s in ("coolant_delta_t_c", "psu_ripple_mv", "helium_pct", "cold_head_temp_k",
                              "tube_arc_count", "gradient_error_count")]
)
SPATIAL_FEATURES = [f"spatial_rate_{m}" for m in MODES] + ["spatial_rate_all"]


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Days until the machine's next failure (strictly after today), whether it is observed, and its mode.

    Rows after a machine's last failure are right-censored at the end of the record.
    """
    out = []
    for _, g in df.sort_values("date").groupby("machine_id", sort=False):
        g = g.reset_index(drop=True).copy()
        dates = g["date"].to_numpy()
        modes = g["failure_mode"].to_numpy(dtype=object)
        n = len(g)
        days = np.empty(n, dtype=float)
        event = np.zeros(n, dtype=int)
        next_mode = np.full(n, None, dtype=object)
        nxt_date, nxt_mode = None, None
        for i in range(n - 1, -1, -1):  # walk backwards, carrying the next failure seen so far
            if nxt_date is not None:
                days[i], event[i], next_mode[i] = (nxt_date - dates[i]) / np.timedelta64(1, "D"), 1, nxt_mode
            else:
                days[i] = (dates[-1] - dates[i]) / np.timedelta64(1, "D")
            if modes[i] is not None and modes[i] == modes[i]:  # failure today (skip None and NaN)
                nxt_date, nxt_mode = dates[i], modes[i]
        g["event"] = event
        g["days_to_failure"] = np.clip(days, 1, None)
        g["next_mode"] = next_mode
        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machine_id", "date"]).copy()
    df["is_mri"] = (df["modality"] == "MRI").astype(int)
    grouped = df.groupby("machine_id")
    for s in SENSORS:
        df[f"{s}_mean7"] = grouped[s].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df[f"{s}_mean30"] = grouped[s].transform(lambda x: x.rolling(30, min_periods=1).mean())
    for s in ("coolant_delta_t_c", "psu_ripple_mv", "helium_pct", "cold_head_temp_k",
              "tube_arc_count", "gradient_error_count"):
        df[f"{s}_trend"] = df[f"{s}_mean7"] - df[f"{s}_mean30"]
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def site_failure_rates(history: pd.DataFrame) -> pd.DataFrame:
    """Failures per machine-year by site and mode, from a history window."""
    machine_years = history.groupby("site")["machine_id"].count() / 365.25
    counts = history.dropna(subset=["failure_mode"]).groupby(["site", "failure_mode"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=machine_years.index, columns=list(MODES), fill_value=0)
    rates = counts.div(machine_years, axis=0)
    rates["all"] = rates.sum(axis=1)
    coords = history.groupby("site")[["lat", "lon"]].first()
    return rates.join(coords)


def spatial_features(rates: pd.DataFrame, lat: float, lon: float, exclude_site: str | None,
                     power: float = 2.0, bandwidth_km: float = 600.0) -> dict:
    """Inverse-distance-weighted failure rates of *other* sites around a location.

    Excluding the site itself makes the feature an honest estimate of what the
    neighborhood implies, which is exactly the information available for a new
    site with no failure history of its own.
    """
    others = rates.drop(index=exclude_site, errors="ignore")
    d = haversine_km(lat, lon, others["lat"].to_numpy(), others["lon"].to_numpy())
    w = np.exp(-d / bandwidth_km) / np.maximum(d, 50) ** (power - 1)
    w = w / w.sum()
    cols = list(MODES) + ["all"]
    return {f"spatial_rate_{c}": float(np.dot(w, others[c].to_numpy())) for c in cols}


def attach_spatial(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    sites = df.groupby("site")[["lat", "lon"]].first()
    feats = {s: spatial_features(rates, r.lat, r.lon, exclude_site=s) for s, r in sites.iterrows()}
    return df.join(pd.DataFrame.from_dict(feats, orient="index"), on="site")


def build_dataset(fleet: pd.DataFrame) -> pd.DataFrame:
    return add_rolling(add_labels(fleet))

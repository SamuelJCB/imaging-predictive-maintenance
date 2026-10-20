"""Synthetic CT and MRI fleet telemetry across hospital sites in India and Singapore.

Each machine produces one row per day of operational and environmental
telemetry. Hidden component wear accumulates with use and environment, and
failures are drawn from a per-mode daily hazard. A repair resets the failed
component. Site locations are approximate city coordinates; no real hospital
or device data is used.

One driver is deliberately **unobserved**: an airborne dust and salt index
that varies smoothly over space (high in dense coastal cities). It raises the
hazard of cooling and RF coil failures (clogged filters, contaminated coil
contacts) without showing up in any sensor. A model can only learn it
indirectly, from what nearby sites experienced, which is what the spatial
layer is for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODES = ("xray_tube", "cooling", "power_supply", "cryogen", "rf_gradient_coil")
MODALITY_MODES = {
    "CT": ("xray_tube", "cooling", "power_supply"),
    "MRI": ("cooling", "power_supply", "cryogen", "rf_gradient_coil"),
}

# name, lat, lon, mean ambient temp in equipment room (C), humidity (%), grid sag rate (events/day)
SITES = [
    ("Mumbai", 19.08, 72.88, 27.5, 68, 0.35),
    ("Delhi", 28.61, 77.21, 26.0, 52, 0.55),
    ("Chennai", 13.08, 80.27, 28.5, 70, 0.40),
    ("Bengaluru", 12.97, 77.59, 24.0, 55, 0.30),
    ("Kolkata", 22.57, 88.36, 27.5, 72, 0.50),
    ("Kochi", 9.93, 76.27, 27.0, 75, 0.25),
    ("Singapore", 1.35, 103.82, 23.0, 60, 0.03),
]


def dust_index(lat: float, lon: float) -> float:
    """Hidden spatial field in [0, 1]: salt-laden coastal air centered on the southern peninsula."""
    return float(np.exp(-(((lat - 11.5) / 5) ** 2 + ((lon - 78.5) / 4.5) ** 2)))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_fleet(days: int = 1095, machines_per_site: int = 8, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    season = np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 100) / 365.25)
    rows = []
    machine_id = 0
    for site, lat, lon, t_mean, h_mean, sag_rate in SITES:
        dust = dust_index(lat, lon)
        for k in range(machines_per_site):
            machine_id += 1
            modality = "CT" if k % 2 == 0 else "MRI"
            age_years = rng.uniform(1, 12)
            wear = {m: rng.uniform(0, 0.4) for m in MODES}
            last_service = -int(rng.integers(0, 90))
            helium = 95.0
            for d in range(days):
                ambient = t_mean + 2.5 * season[d] + rng.normal(0, 1.2)
                humidity = np.clip(h_mean + 8 * season[d] + rng.normal(0, 5), 20, 98)
                sags = rng.poisson(sag_rate * (1 + 0.5 * max(season[d], 0)))
                scans = max(0, rng.normal(42 if modality == "CT" else 18, 6))

                # Hidden wear accumulates with use and heat. Sensors below reveal it only noisily.
                wear["xray_tube"] += 0.0018 * scans / 40
                wear["cooling"] += 0.0020 * (1 + 0.15 * max(ambient - 24, 0))
                wear["power_supply"] += 0.0012 * (1 + 1.5 * sags)
                wear["cryogen"] += 0.0020
                wear["rf_gradient_coil"] += 0.0016 * (scans / 18)

                if modality == "MRI":
                    helium = max(helium - (0.02 + 0.08 * wear["cryogen"]) + rng.normal(0, 0.01), 0)
                days_since_service = d - last_service

                # Noisy sensor readings that partially reveal wear.
                telemetry = dict(
                    ambient_temp_c=round(ambient, 2),
                    humidity_pct=round(humidity, 1),
                    voltage_sag_events=int(sags),
                    scans=round(scans, 1),
                    tube_arc_count=int(rng.poisson(0.2 + 6 * wear["xray_tube"] ** 2)) if modality == "CT" else 0,
                    coolant_delta_t_c=round(4 + 5 * wear["cooling"] + rng.normal(0, 0.4), 2),
                    psu_ripple_mv=round(12 + 25 * wear["power_supply"] + rng.normal(0, 2), 2),
                    helium_pct=round(helium, 2) if modality == "MRI" else np.nan,
                    cold_head_temp_k=round(4.0 + 1.5 * wear["cryogen"] + rng.normal(0, 0.08), 3) if modality == "MRI" else np.nan,
                    gradient_error_count=int(rng.poisson(0.1 + 3 * wear["rf_gradient_coil"] ** 2)) if modality == "MRI" else 0,
                    days_since_service=days_since_service,
                )

                hazards = {
                    "xray_tube": _sigmoid(16 * (wear["xray_tube"] - 1.0) - 4.5),
                    "cooling": _sigmoid(16 * (wear["cooling"] - 1.0) + 0.05 * (humidity - 60) + 4.0 * dust - 4.5),
                    "power_supply": _sigmoid(16 * (wear["power_supply"] - 1.0) + 0.6 * sags - 4.5),
                    "cryogen": _sigmoid(16 * (wear["cryogen"] - 1.0) - 0.03 * (helium - 60) - 4.5),
                    "rf_gradient_coil": _sigmoid(16 * (wear["rf_gradient_coil"] - 1.0) + 4.0 * dust - 4.5),
                }
                allowed = MODALITY_MODES[modality]
                failed = None
                for mode in allowed:
                    if rng.random() < hazards[mode]:
                        failed = mode
                        break

                rows.append(dict(date=dates[d], site=site, lat=lat, lon=lon, machine_id=f"M{machine_id:03d}",
                                 modality=modality, age_years=round(age_years + d / 365.25, 2), **telemetry,
                                 failure_mode=failed))
                if failed:
                    wear[failed] = rng.uniform(0, 0.1)
                    if failed == "cryogen":
                        helium = 95.0
                    last_service = d
                elif days_since_service > 180:
                    # Scheduled preventive maintenance partially restores every component.
                    for m in allowed:
                        wear[m] *= 0.85
                    last_service = d
    return pd.DataFrame(rows)

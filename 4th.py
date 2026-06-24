"""
Population-weighted zone aggregation and HDD/CDD computation.

Population weighting rationale: gas and power demand correlates with people,
not geography.  A hot Sahara adds nothing to European power demand; a cold
Berlin adds a great deal.  Each station's contribution is proportional to
its metro-area population.

HDD/CDD bases:
  15°C  — NW European gas industry standard (UK National Grid, TTF, etc.)
  18°C  — EU Eurostat / heating system design standard
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = ~np.isnan(values)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _circular_mean_wind(directions_deg: np.ndarray, weights: np.ndarray) -> float:
    """
    Population-weighted circular mean of wind directions.
    Avoids the naive arithmetic mean wrap-around error (e.g. 359° + 1° != 180°).
    """
    mask = ~np.isnan(directions_deg)
    if not mask.any():
        return np.nan
    d, w = directions_deg[mask], weights[mask]
    rad = np.radians(d)
    mean_deg = np.degrees(
        np.arctan2(
            np.average(np.sin(rad), weights=w),
            np.average(np.cos(rad), weights=w),
        )
    ) % 360
    return float(mean_deg)


def aggregate_zone(
    raw_df: pd.DataFrame,
    stations: List[Tuple],   # [(lat, lon, pop_millions, name), ...]
) -> pd.DataFrame:
    """
    Aggregate station-level daily weather to a single population-weighted
    zone-level daily time series.

    Input  raw_df: columns [date, station_idx, temperature_2m_max, ...]
    Output df    : one row per date.
    """
    pops = np.array([s[2] for s in stations], dtype=float)

    records = []
    for dt, grp in raw_df.groupby("date"):
        grp = grp.sort_values("station_idx").reset_index(drop=True)
        idxs = grp["station_idx"].values
        w = pops[idxs]

        def wav(col: str) -> float:
            return _weighted_mean(grp[col].values.astype(float), w)

        temp_mean = wav("temperature_2m_mean")

        rec = {
            "date":                    dt,
            "temp_max_c":              wav("temperature_2m_max"),
            "temp_min_c":              wav("temperature_2m_min"),
            "temp_mean_c":             temp_mean,
            "precipitation_mm":        wav("precipitation_sum"),
            "cloud_cover_pct":         wav("cloudcover_mean"),
            "solar_radiation_mj_m2":   wav("shortwave_radiation_sum"),
            "wind_speed_max_kmh":      wav("windspeed_10m_max"),
            "wind_direction_deg":      _circular_mean_wind(
                                           grp["winddirection_10m_dominant"].values.astype(float),
                                           w,
                                       ),
            "stations_used":           int((~grp["temperature_2m_mean"].isna()).sum()),
        }

        # HDD / CDD — two standard base temperatures
        if not np.isnan(temp_mean):
            rec["hdd_15c"] = max(0.0, 15.0 - temp_mean)
            rec["cdd_15c"] = max(0.0, temp_mean - 15.0)
            rec["hdd_18c"] = max(0.0, 18.0 - temp_mean)
            rec["cdd_18c"] = max(0.0, temp_mean - 18.0)
        else:
            rec.update(hdd_15c=np.nan, cdd_15c=np.nan, hdd_18c=np.nan, cdd_18c=np.nan)

        records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)
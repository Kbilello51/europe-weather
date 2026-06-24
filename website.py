"""
Open-Meteo API wrapper.

Historical data: ERA5 reanalysis via archive-api.open-meteo.com
  - Free, no API key required (rate-limited); or commercial with API key
  - ERA5 goes back to 1940-01-01
  - Archive lags ~5 days behind today

Forecast data: ECMWF IFS via api.open-meteo.com
  - Best-in-class European NWP for Europe, updated ~4x/day
  - Up to 16 days ahead; past_days=7 bridges the archive lag

Free-tier limits (open-meteo.com):
  - 10,000 requests/day  |  5,000/hour  |  600/minute
  - Data volume per request: ~1 year of daily data per call
  We chunk by year (one API call per year) and sleep 1 s between calls.

Commercial tier ($10/month at open-meteo.com):
  - No rate limits; unlimited data per call
  - Set OPENMETEO_API_KEY env var or pass --api-key to the CLI.
  - With a key, year-chunking is unnecessary but kept for cache efficiency.
"""

import logging
import os
import time
from datetime import date as date_type
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# API endpoints — commercial uses customer-api subdomain
_FREE_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
_FREE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_COM_ARCHIVE_URL   = "https://customer-archive-api.open-meteo.com/v1/archive"
_COM_FORECAST_URL  = "https://customer-api.open-meteo.com/v1/forecast"

# Seconds between API calls on the free tier (keeps us ~60 req/min)
_FREE_DELAY = 1.0
_COM_DELAY  = 0.0   # commercial has no meaningful rate limit

# Daily variables — index order must match Variables(i) in the response
DAILY_VARS = [
    "temperature_2m_max",          # 0  °C
    "temperature_2m_min",          # 1  °C
    "temperature_2m_mean",         # 2  °C
    "precipitation_sum",           # 3  mm
    "cloudcover_mean",             # 4  %
    "shortwave_radiation_sum",     # 5  MJ/m²
    "windspeed_10m_max",           # 6  km/h  (daily max; ERA5 daily doesn't include mean)
    "winddirection_10m_dominant",  # 7  degrees
]

# ─── module-level state ────────────────────────────────────────────────────────

_api_key: Optional[str] = None
_client  = None
_fresh_client = None   # for forecast (no cache)


def configure(api_key: Optional[str] = None) -> None:
    """
    Set the Open-Meteo API key.  Call before any fetch functions, or set the
    OPENMETEO_API_KEY environment variable.  Passing None uses the free tier.
    """
    global _api_key, _client, _fresh_client
    _api_key      = api_key or os.environ.get("OPENMETEO_API_KEY") or None
    _client       = None   # force rebuild
    _fresh_client = None


def _get_api_key() -> Optional[str]:
    if _api_key is not None:
        return _api_key
    return os.environ.get("OPENMETEO_API_KEY") or None


def _archive_url() -> str:
    return _COM_ARCHIVE_URL if _get_api_key() else _FREE_ARCHIVE_URL


def _forecast_url() -> str:
    return _COM_FORECAST_URL if _get_api_key() else _FREE_FORECAST_URL


def _call_delay() -> float:
    return _COM_DELAY if _get_api_key() else _FREE_DELAY


def _build_client(cache_dir: str = ".openmeteo_cache", cache_hours: float = 24.0,
                  expire_override: Optional[int] = None):
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    expire = expire_override if expire_override is not None else int(cache_hours * 3600)
    session = requests_cache.CachedSession(cache_dir, expire_after=expire)
    session = retry(session, retries=3, backoff_factor=2.0)
    return openmeteo_requests.Client(session=session)


def _get_client():
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def _get_fresh_client():
    global _fresh_client
    if _fresh_client is None:
        _fresh_client = _build_client(expire_override=0)
    return _fresh_client


def _inject_key(params: dict) -> dict:
    key = _get_api_key()
    if key:
        params = {**params, "apikey": key}
    return params


# ─── response parsing ─────────────────────────────────────────────────────────

def _parse_responses(responses) -> pd.DataFrame:
    """Convert openmeteo Response list into a tidy long DataFrame."""
    dfs = []
    for i, resp in enumerate(responses):
        daily    = resp.Daily()
        start_ts = pd.to_datetime(daily.Time(),    unit="s", utc=True)
        end_ts   = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True)
        interval = pd.Timedelta(seconds=daily.Interval())
        dates    = pd.date_range(
            start=start_ts, end=end_ts, freq=interval, inclusive="left"
        ).date

        data: dict = {"date": dates, "station_idx": i}
        for j, var in enumerate(DAILY_VARS):
            try:
                data[var] = daily.Variables(j).ValuesAsNumpy()
            except Exception:
                data[var] = np.full(len(dates), np.nan)

        dfs.append(pd.DataFrame(data))

    return pd.concat(dfs, ignore_index=True)


# ─── year-chunk helpers ───────────────────────────────────────────────────────

def _split_years(start: date_type, end: date_type) -> List[Tuple[str, str]]:
    chunks, year = [], start.year
    while year <= end.year:
        cs = max(start, date_type(year, 1, 1))
        ce = min(end,   date_type(year, 12, 31))
        chunks.append((cs.isoformat(), ce.isoformat()))
        year += 1
    return chunks


def year_chunks(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """Public helper: list of (start, end) strings, one per calendar year."""
    return _split_years(
        date_type.fromisoformat(start_date),
        date_type.fromisoformat(end_date),
    )


def inter_call_sleep() -> None:
    """Sleep between API calls; no-op on commercial tier."""
    delay = _call_delay()
    if delay > 0:
        time.sleep(delay)


# ─── public fetch functions ───────────────────────────────────────────────────

def fetch_historical_chunk(
    stations: List[Tuple],   # [(lat, lon, pop, name), ...]
    start: str,              # "YYYY-MM-DD"
    end:   str,              # "YYYY-MM-DD"
) -> pd.DataFrame:
    """
    Single ERA5 archive API call for one year's data across all zone stations.
    Results are cached 24 h (historical data never changes).
    """
    params = _inject_key({
        "latitude":   [s[0] for s in stations],
        "longitude":  [s[1] for s in stations],
        "start_date": start,
        "end_date":   end,
        "daily":      DAILY_VARS,
        "timezone":   "UTC",
    })
    responses = _get_client().weather_api(_archive_url(), params=params)
    return _parse_responses(responses)


def fetch_forecast(
    stations:     List[Tuple],
    past_days:    int = 7,
    forecast_days: int = 16,
) -> pd.DataFrame:
    """
    Fetch ECMWF-based forecast + recent observations.
    Always fetches fresh (bypasses cache) so each run reflects the latest model run.
    past_days=7 fills the ~5-day ERA5 archive lag so there is no gap in combined output.
    """
    params = _inject_key({
        "latitude":       [s[0] for s in stations],
        "longitude":      [s[1] for s in stations],
        "daily":          DAILY_VARS,
        "timezone":       "UTC",
        "past_days":      past_days,
        "forecast_days":  forecast_days,
    })
    responses = _get_fresh_client().weather_api(_forecast_url(), params=params)
    return _parse_responses(responses)


def using_commercial() -> bool:
    """True if an API key is active."""
    return bool(_get_api_key())
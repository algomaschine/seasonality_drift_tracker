# Copyright 2025 Eduard Samokhvalov
#
# This file is part of the Enhanced Cycle Scanner project.
#
# Licensed under the GNU General Public License v3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Additional Grant: This software is specifically granted for use by the
# "Foundation for the Study of Cycles".
"""
Enhanced Cycle Scanner
---------------------

This software implements advanced cycle detection and forecasting for financial time series, with a focus on Bitcoin price data. It features robust handling of trend, evolving seasonality (drift), multi-timeframe and multi-variant analysis, and interactive reporting. The code is released as open source under the GNU GPLv3 license. See the LICENSE file for details.

Author: Eduard Samokhvalov
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.structural import UnobservedComponents
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.statespace.tools as sstools
import os
import webbrowser
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import math
import time
import logging
import sys
import traceback
from io import StringIO
import hashlib
from collections import defaultdict
import contextlib
import json
from datetime import datetime as dt
import colorsys

# Configure logging to both console and file
def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"btc_cycle_scanner_{timestamp}.log")
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Log start message
    logging.info(f"Enhanced Cycle Scanner started. Logging to {log_file}")
    return log_file

# Ignore harmless warnings from statsmodels
warnings.filterwarnings("ignore")
# Set compatibility mode as an attribute rather than calling it as a function
sstools.compatibility_mode = True

# --- Module-level instruments list ---
INSTRUMENTS = ["BTC-USD.CC", "SOL-USD.CC"]

# --- Module-level constants and paths ---
API_KEY = "xxx" # you need to get a key from https://www.eodhd.com/
CACHE_FILE = "btc_1min_cache.csv"
CACHE_START_DATE = "2020-01-01"
REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

# --- Helper functions for file paths ---
def get_report_dir(instrument):
    d = os.path.join(REPORTS_DIR, instrument)
    os.makedirs(d, exist_ok=True)
    return d

def get_cache_file(instrument):
    return f"{instrument.replace('-', '_')}_1min_cache.csv"

# Hourly timeframes (in minutes) - from 1h to 24h
# Only use intervals officially supported by EODHD API
# HOURLY_TIMEFRAMES = [60, 240, 720]  # 1h, 4h, 12h (most likely to be supported)

REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

# Fibonacci-based timeframes (in minutes) from 1h to ~1mo
FIB_HOURS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
TIMEFRAME_MINUTES = [h * 60 for h in FIB_HOURS]  # [60, 120, 180, 300, 480, 780, 1260, 2040, 3300, 5340, 8640, 13980, 22620, 36600]
# Use pandas offset aliases for rules (if possible)
TIMEFRAME_RULES = {m: f'{m}T' for m in TIMEFRAME_MINUTES}
# Seasonality: for each timeframe, use 24 (daily), 7*24 (weekly), or 30*24 (monthly) as a rough guide
SEASONALITY_MAP = {m: max(2, int((1440*7)//m)) for m in TIMEFRAME_MINUTES}  # e.g. 7 days in units of timeframe

TIMING_STATS = defaultdict(float)

@contextlib.contextmanager
def timing(label):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    TIMING_STATS[label] += elapsed

# --- Step 0: Data Fetching and Caching Functions ---

def fetch_and_cache_1min_data(api_token=API_KEY, cache_file=None, start_date=CACHE_START_DATE, instrument="BTC-USD.CC"):
    """Download all 1-minute data for instrument from start_date, cache to CSV, and return as DataFrame.
       If cache exists, only download and append missing data.
    """
    if cache_file is None:
        # Try both new and old cache file formats
        cache_file = get_cache_file(instrument)  # New format: BTC_USD_CC_1min_cache.csv
        old_cache_file = "btc_1min_cache.csv"    # Old format
        
        # If old cache exists but new doesn't, use the old one
        if not os.path.exists(cache_file) and os.path.exists(old_cache_file):
            logging.info(f"[{instrument}] Found old format cache file, using: {old_cache_file}")
            cache_file = old_cache_file
        
    try:
        # First check if cache exists and try to load it
        if os.path.exists(cache_file):
            logging.info(f"[{instrument}] Loading 1-minute data from cache: {cache_file}")
            try:
                df = pd.read_csv(cache_file, parse_dates=['datetime'])
                if not df.empty:
                    latest_date = df['datetime'].max()
                    end_date = pd.to_datetime(datetime.utcnow())
                    
                    # Check if cache is up to date (within last minute)
                    if latest_date >= end_date - pd.Timedelta(minutes=1):
                        logging.info(f"[{instrument}] Cache is up to date. Latest data: {latest_date}")
                        return df
                        
                    # Only download new data from latest_date + 1 minute
                    logging.info(f"[{instrument}] Cache found with data up to {latest_date}. Will only download newer data.")
                    current_start = latest_date + pd.Timedelta(minutes=1)
                else:
                    logging.warning(f"[{instrument}] Cache file exists but is empty. Will download from {start_date}")
                    current_start = pd.to_datetime(start_date)
            except Exception as e:
                logging.error(f"[{instrument}] Error reading cache file: {e}. Will download from {start_date}")
                current_start = pd.to_datetime(start_date)
        else:
            logging.info(f"[{instrument}] No cache file found. Will download from {start_date}")
            current_start = pd.to_datetime(start_date)
            df = pd.DataFrame()  # Empty DataFrame for new data

        # Download only missing data
        end_date = pd.to_datetime(datetime.utcnow())
        all_data = []
        chunk_days = 30
        
        if current_start < end_date:
            logging.info(f"[{instrument}] Downloading missing data from {current_start} to {end_date}")
            while current_start < end_date:
                chunk_end = min(current_start + pd.Timedelta(days=chunk_days), end_date)
                from_ts = int(current_start.timestamp())
                to_ts = int(chunk_end.timestamp())
                url = f"https://eodhd.com/api/intraday/{instrument}?api_token={api_token}&interval=1m&from={from_ts}&to={to_ts}&fmt=json"
                logging.info(f"[{instrument}] Requesting {current_start.date()} to {chunk_end.date()}...")
                
                try:
                    r = requests.get(url, timeout=60)
                    r.raise_for_status()
                    data = r.json()
                    if isinstance(data, dict) and 'message' in data:
                        logging.error(f"[{instrument}] API returned an error message: {data['message']}")
                        break
                    if not data:
                        logging.warning(f"[{instrument}] No data returned for {current_start.date()} to {chunk_end.date()}")
                        break
                        
                    df_chunk = pd.DataFrame(data)
                    if 'datetime' in df_chunk.columns:
                        df_chunk['datetime'] = pd.to_datetime(df_chunk['datetime'])
                        all_data.append(df_chunk)
                    else:
                        logging.warning(f"[{instrument}] No 'datetime' column in chunk {current_start.date()} to {chunk_end.date()}")
                except Exception as e:
                    logging.error(f"[{instrument}] Error downloading chunk {current_start.date()} to {chunk_end.date()}: {e}")
                    break
                    
                current_start = chunk_end

            # Process and save new data if any was downloaded
            if all_data:
                df_new = pd.concat(all_data, ignore_index=True)
                df_new = df_new.drop_duplicates(subset=['datetime']).sort_values('datetime')
                
                # Combine with existing data if any
                if not df.empty:
                    df = pd.concat([df, df_new], ignore_index=True)
                    df = df.drop_duplicates(subset=['datetime']).sort_values('datetime')
                else:
                    df = df_new
                    
                # Save updated cache (always save to new format)
                new_cache_file = get_cache_file(instrument)
                df.to_csv(new_cache_file, index=False)
                logging.info(f"[{instrument}] Updated cache saved to: {new_cache_file}")
                
                # If we were using old format, save there too for backward compatibility
                if cache_file == old_cache_file:
                    df.to_csv(old_cache_file, index=False)
                    logging.info(f"[{instrument}] Also saved to old format cache: {old_cache_file}")
        else:
            logging.info(f"[{instrument}] No new data needed to be downloaded")
            
        return df
        
    except Exception as e:
        logging.error(f"[{instrument}] Unexpected error in fetch_and_cache_1min_data: {e}")
        if 'df' in locals() and not df.empty:
            return df  # Return whatever data we have
        raise  # Re-raise if we have no data at all

def resample_ohlc(df_1min, rule, price_col='close'):
    """Resample 1-minute OHLCV data to a new timeframe using pandas."""
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df = df_1min.set_index('datetime').sort_index()
    df_resampled = df.resample(rule).agg(ohlc_dict)
    df_resampled = df_resampled.dropna(subset=['close'])
    df_resampled = df_resampled.reset_index()
    return df_resampled

# --- Step 1: Helper Functions ---

def bartels_test(data_series, cycle_length):
    """
    Implements the Bartels Test to assess statistical significance of a cycle.
    
    The Bartels test evaluates whether a cycle is random or represents genuine periodicity.
    
    Args:
        data_series (pd.Series): The time series data (residuals).
        cycle_length (float): The length of the cycle to test.

    Returns:
        float: The Bartels score (0 to 1). Lower score means less likely due to chance.
    """
    if len(data_series) < 2 * cycle_length:
        return 1.0  # Not enough data for reliable test
    
    try:
        cycle_length = int(max(1, cycle_length)) # Ensure positive integer
        n_cycles = int(len(data_series) / cycle_length)
        if n_cycles == 0:
            return 1.0 # Not enough data for even one cycle
        effective_length = int(n_cycles * cycle_length)
        folded_data = data_series[:effective_length].values.reshape(n_cycles, cycle_length)
        
        phase_means = np.mean(folded_data, axis=0)
        phase_variance = np.var(phase_means)
        total_variance = np.var(data_series[:effective_length])
        
        if total_variance == 0:
            return 1.0  # Avoid division by zero
        
        R = phase_variance / total_variance
        score = max(0, min(1, 1 - (R * n_cycles))) # Simplified score logic
        return score
    except Exception as e:
        logging.error(f"Bartels test calculation failed for length {cycle_length}: {e}")
        return 1.0

def estimate_current_phase(series, cycle_length, window_size=None):
    """
    Estimates the phase of a cycle at the end of the series by fitting a sine wave
    to the last N points.
    """
    if cycle_length <= 0:
      return np.nan
    if window_size is None:
        window_size = int(max(10, 2 * cycle_length)) # Ensure reasonable window
    window_size = min(window_size, len(series)) # Cap window size at series length

    if len(series) < window_size or window_size < 3:
        # print(f"Warning: Series too short ({len(series)}) for phase estimation window ({window_size}).")
        return np.nan

    y_data = series.iloc[-window_size:].values
    x_data = np.arange(len(series))[-window_size:]

    frequency = 1.0 / cycle_length

    def sine_func(x, amplitude, phase_shift, mean_offset):
        return amplitude * np.sin(2 * np.pi * frequency * x + phase_shift) + mean_offset

    try:
        # Provide initial guesses
        initial_amplitude = np.std(y_data) * np.sqrt(2)
        initial_phase = 0
        initial_offset = np.mean(y_data)
        p0 = [initial_amplitude, initial_phase, initial_offset]

        params, _ = curve_fit(sine_func, x_data, y_data, p0=p0, maxfev=5000, bounds=([-np.inf, -np.pi, -np.inf], [np.inf, np.pi, np.inf]))

        amplitude, phase_shift, _ = params
        last_x = x_data[-1]
        current_phase = (2 * np.pi * frequency * last_x + phase_shift) % (2 * np.pi)
        return current_phase

    except Exception as e:
        # print(f"Debug: Phase estimation failed for cycle {cycle_length:.2f}. Error: {e}")
        return np.nan

def dynamic_harmonic_regression(series, period, max_order=3):
    """
    Implements Dynamic Harmonic Regression to model time-varying seasonality.
    Uses time-varying Fourier terms.
    """
    logging.info(f"  Fitting Dynamic Harmonic Regression (period={period}, max_order={max_order})...")
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    if period <= 0 or max_order <= 0:
        logging.warning("  DHR Warning: Invalid period or max_order.")
        return pd.Series(0, index=series.index), series

    time_idx = np.arange(len(series))
    time_trend = time_idx / len(series)  # Normalize to [0, 1] range

    # Create design matrix (exog)
    X_list = [np.ones(len(series)), time_idx] # Intercept and linear trend
    harmonic_indices = [] # To track indices of time-varying terms

    for i in range(1, max_order + 1):
        # Static terms
        sin_term_static = np.sin(2 * np.pi * i * time_idx / period)
        cos_term_static = np.cos(2 * np.pi * i * time_idx / period)
        X_list.append(sin_term_static)
        X_list.append(cos_term_static)

        # Time-varying terms (interaction with normalized time)
        X_list.append(sin_term_static * time_trend)
        harmonic_indices.append(len(X_list) - 1)
        X_list.append(cos_term_static * time_trend)
        harmonic_indices.append(len(X_list) - 1)

    X = np.column_stack(X_list)

    # Define SARIMAX model with external regressors
    model = sm.tsa.statespace.SARIMAX(
        series,
        exog=X,
        order=(1, 0, 1), # Basic ARIMA for residuals, can be tuned
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    try:
        results = model.fit(disp=False, maxiter=100)

        # Extract seasonal component using only the time-interaction parameters
        seasonal_params = results.params[harmonic_indices]
        X_seasonal_dynamic = X[:, harmonic_indices]

        seasonal_component = X_seasonal_dynamic @ seasonal_params
        residuals = results.resid # Use residuals from SARIMAX fit

        logging.info(f"  DHR model fitted successfully. AIC: {results.aic:.2f}")
        return pd.Series(seasonal_component, index=series.index), residuals
    except Exception as e:
        logging.error(f"  DHR model fitting failed: {e}")
        # Simple fallback within DHR: return 0 seasonality and original series as residuals
        return pd.Series(0, index=series.index), series

def save_plotly_html_report(log_price, trend, seasonality, residuals, cycles_df, dominant_cycle,
                             ssm_successful, dhr_successful, timeframe_name,
                             output_file=os.path.join(REPORTS_DIR, "bitcoin_cycle_analysis_{}.html"), seasonality_period=7, resample_rule=None):
    """Generates comprehensive HTML report with Plotly visualizations and explanations."""
    logging.info(f"  Generating Plotly HTML report for {timeframe_name}...")

    # --- Ensure DatetimeIndex for plotting and frequency inference ---
    if not isinstance(log_price.index, pd.DatetimeIndex):
        try:
            log_price.index = pd.to_datetime(log_price.index)
        except Exception:
            pass
    if trend is not None and not isinstance(trend.index, pd.DatetimeIndex):
        try:
            trend.index = pd.to_datetime(trend.index)
        except Exception:
            pass
    if seasonality is not None and not isinstance(seasonality.index, pd.DatetimeIndex):
        try:
            seasonality.index = pd.to_datetime(seasonality.index)
        except Exception:
            pass

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f"Bitcoin Price Decomposition ({timeframe_name}, Log Scale)",
            f"Extracted Seasonality ({timeframe_name}, Showing Drift)",
            f"Last 20% Data with Forward Cycle Projection ({timeframe_name})",
            f"Periodogram ({timeframe_name})"
        ),
        vertical_spacing=0.08,
        row_heights=[0.3, 0.2, 0.3, 0.2] # Adjust heights as needed
    )

    # --- Plot 1: Decomposition ---
    fig.add_trace(go.Scatter(x=log_price.index, y=log_price.values, name='Log Price', line=dict(color='navy', width=1), opacity=0.7), row=1, col=1)
    if trend is not None:
        fig.add_trace(go.Scatter(x=trend.index, y=trend.values, name='Trend', line=dict(color='red', width=2)), row=1, col=1)
    if trend is not None and seasonality is not None:
      fig.add_trace(go.Scatter(x=trend.index, y=trend.values + seasonality.values, name='Trend+Seasonality', line=dict(color='green', width=1.5, dash='dash')), row=1, col=1)

    # --- Plot 2: Seasonality Drift ---
    if seasonality is not None:
        fig.add_trace(go.Scatter(x=seasonality.index, y=seasonality.values, name='Seasonality', line=dict(color='purple', width=1.5)), row=2, col=1)
        # Add drift visualization (color-coded segments)
        if ssm_successful or dhr_successful:
            windows = min(6, len(log_price) // 30) # Up to 6 windows
            if windows > 0:
              window_size = len(log_price) // windows
              colors = ['#440154', '#3b528b', '#21908d', '#5dc863', '#fde725'] # Viridis subset
              for i in range(windows):
                  start_idx = i * window_size
                  end_idx = min(start_idx + window_size, len(log_price))
                  color = colors[i % len(colors)]
                  fig.add_trace(go.Scatter(x=log_price.index[start_idx:end_idx],
                                         y=seasonality.values[start_idx:end_idx],
                                         name=f'Period {i+1}', legendgroup="drift", showlegend=(i==0),
                                         line=dict(color=color, width=2), opacity=0.8), row=2, col=1)

    # --- Plot 3: Forecast with Forward Projection ---
    forecast_start_idx = int(len(log_price) * 0.8)
    forecast_data_actual = log_price.iloc[forecast_start_idx:]
    forecast_dates_actual = log_price.index[forecast_start_idx:]

    fig.add_trace(go.Scatter(x=forecast_dates_actual, y=forecast_data_actual.values, name='Actual Log Price', line=dict(color='blue', width=2)), row=3, col=1)

    forecast_baseline_actual = None
    if trend is not None:
        if isinstance(trend, pd.Series):
            forecast_trend_actual = trend.iloc[forecast_start_idx:]
        else:
            forecast_trend_actual = pd.Series(trend[forecast_start_idx:], index=forecast_data_actual.index)

        if seasonality is not None:
            if isinstance(seasonality, pd.Series):
                forecast_seasonality_actual = seasonality.iloc[forecast_start_idx:]
            else:
                forecast_seasonality_actual = pd.Series(seasonality[forecast_start_idx:], index=forecast_data_actual.index)
            forecast_baseline_actual = forecast_trend_actual + forecast_seasonality_actual
        else:
            forecast_baseline_actual = forecast_trend_actual # Case with trend but no seasonality
            forecast_seasonality_actual = pd.Series(0, index=forecast_dates_actual) # Placeholder

    if forecast_baseline_actual is not None:
       fig.add_trace(go.Scatter(x=forecast_dates_actual, y=forecast_baseline_actual, name='Baseline (Hist)', line=dict(color='green', width=1.5, dash='dot')), row=3, col=1)

    # --- Forward Projection Logic ---
    future_forecast_series = None
    if dominant_cycle is not None and forecast_baseline_actual is not None and residuals is not None and trend is not None and seasonality is not None:
        period = dominant_cycle['period']
        phase = dominant_cycle['current_phase']
        if np.isnan(phase):
            phase = 0

        # --- Calculate cycle for historical part ---
        amplitude = np.std(residuals.iloc[forecast_start_idx:]) * 1.5 # Estimate amplitude
        cycle_days_actual = np.arange(len(forecast_data_actual))
        cycle_component_actual = amplitude * np.sin(2 * np.pi * (cycle_days_actual / period) + phase)
        forecast_series_actual = forecast_baseline_actual + cycle_component_actual
        fig.add_trace(go.Scatter(x=forecast_dates_actual, y=forecast_series_actual, name=f'Cycle ({period:.1f} units, Hist)', line=dict(color='red', width=2, dash='dash')), row=3, col=1)

        # --- Calculate for future part ---
        forecast_horizon = int(max(10, period)) # Project forward for one cycle period, min 10 units
        last_date = log_price.index[-1]
        # Try to infer frequency, fallback to resample_rule
        try:
            inferred_freq = pd.infer_freq(log_price.index)
        except Exception:
            inferred_freq = None
        if not inferred_freq and resample_rule:
            inferred_freq = resample_rule
        if inferred_freq:
            # Use DateOffset which handles months correctly instead of Timedelta
            freq_map = {'1T': dict(minutes=1), '5T': dict(minutes=5), '15T': dict(minutes=15), '30T': dict(minutes=30),
                        '1H': dict(hours=1), '2H': dict(hours=2), '4H': dict(hours=4), '6H': dict(hours=6), '12H': dict(hours=12),
                        '1D': dict(days=1), '1W': dict(weeks=1), '1M': dict(months=1)}
            offset_kwargs = freq_map.get(inferred_freq, dict(days=1))
            date_step = pd.DateOffset(**offset_kwargs)
            future_dates = pd.date_range(start=last_date + date_step, periods=forecast_horizon, freq=inferred_freq)
        else: # Cannot infer freq, use numeric index continuation
            future_dates = pd.Index(np.arange(len(log_price), len(log_price) + forecast_horizon))

        # 1. Extrapolate Trend (linear based on last few points)
        if len(trend) >= 2:
            trend_slope = (trend.iloc[-1] - trend.iloc[-2]) # Simple slope from last 2 points
            last_trend_value = trend.iloc[-1]
            trend_future = last_trend_value + trend_slope * np.arange(1, forecast_horizon + 1)
        else:
            trend_future = np.full(forecast_horizon, trend.iloc[-1] if len(trend)>0 else 0) # Constant if < 2 points

        # 2. Extrapolate Seasonality (repeat last full cycle)
        if len(seasonality) >= seasonality_period:
            last_cycle_seasonality = seasonality.iloc[-int(seasonality_period):]
            # Tile the last cycle pattern to cover the forecast horizon
            seasonality_future = np.tile(last_cycle_seasonality.values, 
                                        int(np.ceil(forecast_horizon / seasonality_period)))[:forecast_horizon]
        else:
             seasonality_future = np.zeros(forecast_horizon)

        forecast_baseline_future = trend_future + seasonality_future

        # 3. Calculate Future Cycle Component
        cycle_start_point = len(log_price) # Starting point for future cycle calculation
        cycle_days_future = np.arange(cycle_start_point, cycle_start_point + forecast_horizon)
        cycle_component_future = amplitude * np.sin(2 * np.pi * (cycle_days_future / period) + phase)

        # 4. Combine for Future Forecast
        future_forecast_series = forecast_baseline_future + cycle_component_future

        # 5. Add future traces to plot
        fig.add_trace(go.Scatter(x=future_dates, y=forecast_baseline_future, name='Baseline (Proj)', line=dict(color='orange', width=1.5, dash='dot')), row=3, col=1)
        fig.add_trace(go.Scatter(x=future_dates, y=future_forecast_series, name=f'Cycle ({period:.1f} units, Proj)', line=dict(color='magenta', width=2, dash='dash')), row=3, col=1)

    elif dominant_cycle is None:
       pass # No forecast line if no dominant cycle

    # Update x-axis range for plot 3 to include projection
    if future_forecast_series is not None:
      fig.update_xaxes(range=[forecast_dates_actual[0], future_dates[-1]], row=3, col=1)

    # --- Plot 4: Periodogram ---
    if residuals is not None:
        residuals_values = residuals.dropna().values
        n = len(residuals_values)
        if n > 1:
            fft_values = fft(residuals_values)
            sample_freq = fftfreq(n, d=1.0)
            positive_freq_indices = np.where((sample_freq > 1/n) & (sample_freq <= 0.5))[0] # Exclude 0 freq
            if len(positive_freq_indices) > 0:
                power = np.abs(fft_values[positive_freq_indices])**2
                periods = 1.0 / sample_freq[positive_freq_indices]
                # Sort by period for cleaner plotting
                sort_idx = np.argsort(periods)
                periods = periods[sort_idx]
                power = power[sort_idx]
                fig.add_trace(go.Scatter(x=periods, y=power, name='Power Spectrum', line=dict(color='blue'), yaxis="y4"), row=4, col=1)

                # Add vertical lines for cycles
                if dominant_cycle is not None:
                    period_val = dominant_cycle['period']
                    fig.add_vline(x=period_val, line_width=2, line_dash="dash", line_color="red", annotation_text=f"Dom: {period_val:.1f}", annotation_position="top left", row=4, col=1)
                    # Other cycles
                    top_cycle_periods = cycles_df.head(5)['period'].values
                    for i, p in enumerate(top_cycle_periods):
                        if p != period_val and i < 3: # Show top 2 others
                           fig.add_vline(x=p, line_width=1, line_dash="dot", line_color="green", annotation_text=f"#{i+1}: {p:.1f}", annotation_position="bottom left" if i%2==0 else "bottom right", row=4, col=1)

    # --- Update Layout for Interactivity ---
    fig.update_layout(
        height=1600, # Increased height for better readability
        hovermode='x unified', # Show crosshair and unified hover labels
        legend_traceorder="reversed",
        template="plotly_white"
    )
    fig.update_xaxes(showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor="gray", spikethickness=1)
    fig.update_yaxes(type="log", row=4, col=1, title_text="Power (log)", secondary_y=False, rangemode='tozero') # Ensure periodogram uses log scale
    # Update x-axis range for periodogram based on actual detected periods
    if 'periods' in locals() and len(periods)>0:
         fig.update_xaxes(range=[max(2, periods.min()), min(n/2, periods.max())], row=4, col=1)
    else: # Default range if no periods
        fig.update_xaxes(range=[2, min(365, n/2 if n>4 else 365)], row=4, col=1)

    # --- Generate HTML Parts ---
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Methodology Description (dynamic timeframe)
    method_description = f"""
    <h2>Enhanced Cycle Scanner ({timeframe_name}): Methodology</h2>
    <p>This analysis uses advanced time series methods to find cycles in {timeframe_name} Bitcoin price data, specifically handling trend and evolving seasonality ("drift").</p>
    <h3>Key Components</h3>
    <ol>
        <li><strong>State Space Modeling (SSM) / Dynamic Harmonic Regression (DHR)</strong>: These models separate the data into trend, seasonal, and residual components. Importantly, they allow the seasonal pattern to change over time (drift handling).</li>
        <li><strong>Fast Fourier Transform (FFT)</strong>: Applied to the residuals (data after removing trend/seasonality) to identify potential cycle frequencies.</li>
        <li><strong>Bartels Test</strong>: Statistically validates if detected cycles are likely real patterns or just random noise.</li>
    </ol>
    <h3>Why This Works</h3>
    <p>Financial data is complex. Standard methods fail because of non-stationarity (trends) and seasonality drift. This approach first isolates these factors, then searches for cycles in the cleaner residual data, leading to more reliable cycle detection.</p>
    """

    # Forecast Guide (dynamic timeframe)
    forecast_guide = f"""
    <h2>How to Read the {timeframe_name} Forecast Visualization</h2>
    <h3>Panel 1: Decomposition</h3>
    <p>Shows the original log price (navy) broken down into its trend (red) and the combined trend+seasonality (green dashed).</p>
    <h3>Panel 2: Seasonality Drift</h3>
    <p>Focuses on the extracted seasonal pattern (purple). Color segments show how this pattern might change over different time periods. For {timeframe_name} data, we model a {seasonality_period}-unit seasonality pattern.</p>
    <h3>Panel 3: Forecast and Forward Projection</h3>
    <p>Zooms into the last 20% of data and projects forward. Shows:</p>
    <ul>
        <li>Actual Price (blue line)</li>
        <li>Historical Baseline (Trend+Seasonality, green dotted line)</li>
        <li>Historical Cycle Forecast (Baseline + Dominant Cycle, red dashed line)</li>
        <li><strong>Projected Baseline</strong> (Extrapolated Trend+Seasonality, orange dotted line)</li>
        <li><strong>Projected Cycle Forecast</strong> (Projected Baseline + Dominant Cycle, magenta dashed line)</li>
    </ul>
    <p>The magenta line shows the projected path if the dominant cycle ({dominant_cycle['period']:.1f} units) continues. This is separate from the modeled seasonality ({seasonality_period} units).</p>
    <h3>Panel 4: Periodogram</h3>
    <p>Visualizes cycle strength (power) vs. cycle length (period). Peaks indicate potential cycles. Vertical lines mark the dominant cycle (red) and other significant ones (green).</p>
    """

    # Results Section (dynamic timeframe)
    results_section = f"<h2>Cycle Analysis Results ({timeframe_name})</h2>"
    if dominant_cycle is not None:
        period = dominant_cycle['period']
        bartels = dominant_cycle['bartels_score']
        phase = dominant_cycle['current_phase']
        days_to_peak = "Unknown"
        days_to_trough = "Unknown"
        if not np.isnan(phase):
           days_to_peak = (period * (1 - (phase / (2 * np.pi))))
           days_to_trough = (period * (0.5 - (phase / (2 * np.pi))) % period)

        confidence = max(0, min(100, 100 * (1 - bartels)))
        results_section += f"""
        <div class="results-box">
            <h3>Dominant Cycle Detected</h3>
            <ul>
                <li><strong>Period</strong>: {period:.2f} ({timeframe_name} units)</li>
                <li><strong>Significance</strong>: {(1-bartels)*100:.1f}% (Bartels Test)</li>
                <li><strong>Current Position</strong>: {f'{(phase/(2*np.pi)*period):.2f}' if not np.isnan(phase) else 'Unknown'} units into cycle</li>
                <li><strong>Est. Time to Next Peak</strong>: {f'{days_to_peak:.2f} units' if isinstance(days_to_peak, float) else days_to_peak}</li>
                <li><strong>Est. Time to Next Trough</strong>: {f'{days_to_trough:.2f} units' if isinstance(days_to_trough, float) else days_to_trough}</li>
                <li><strong>Confidence Level</strong>: {confidence:.1f}%</li>
            </ul>
        </div>
        <h3>Other Significant Cycles</h3>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><th>Rank</th><th>Period (units)</th><th>Significance</th><th>Relative Strength</th></tr>
        """
        for i, (idx, cycle) in enumerate(cycles_df.head(5).iterrows()):
            if i > 0:
                results_section += f"<tr><td>{i+1}</td><td>{cycle['period']:.2f}</td><td>{(1-cycle['bartels_score'])*100:.1f}%</td><td>{(cycle['strength']/dominant_cycle['strength']*100 if dominant_cycle['strength'] > 0 else 0):.1f}%</td></tr>"
        results_section += "</table>"
    else:
        results_section += "<p>No statistically significant dominant cycle was detected.</p>"

    # Assemble HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bitcoin Cycle Analysis ({timeframe_name})</title>
        <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; color: #333; max-width: 1400px; margin: auto; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #2980b9; margin-top: 30px; }}
            h3 {{ color: #3498db; }}
            .results-box {{ background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th {{ background-color: #3498db; color: white; text-align: left; padding: 8px; }}
            td {{ padding: 8px; border: 1px solid #ddd; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .plotly-graph-div {{ margin: 30px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .footer {{ margin-top: 50px; border-top: 1px solid #ddd; padding-top: 20px; font-size: 0.9em; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <h1>Bitcoin Cycle Analysis ({timeframe_name}) with Seasonality Drift Handling</h1>
        <p>Analysis performed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}.</p>
        {fig_html}
        {results_section}
        {method_description}
        {forecast_guide}
        <div class="footer">Generated by Enhanced Cycle Scanner Algorithm.</div>
    </body>
    </html>
    """

    with open(output_file, "w", encoding='utf-8') as f:
        f.write(html_content)
    logging.info(f"  Report saved as '{output_file}'")

# --- Step 2: Main Analysis Pipeline Function ---

def run_analysis(df_raw, timeframe_name, price_col, seasonality_period, resample_rule=None):
    """Runs the full analysis pipeline for a given dataframe and timeframe.
       Returns key metrics for the dominant cycle.
    """
    logging.info(f"Processing {timeframe_name} data...")
    dominant_cycle_metrics = {
        'timeframe': timeframe_name,
        'period': np.nan,
        'significance': 0,
        'strength': 0,
        'confidence': 0
    }

    # --- Data Preparation ---
    with timing(f"{timeframe_name}_data_preparation"):
        if df_raw is None or df_raw.empty:
            logging.error("  Error: Input dataframe is empty. Skipping.")
            return dominant_cycle_metrics # Return default empty metrics

        df = df_raw.copy()
        if price_col not in df.columns:
            logging.error(f"  Error: Price column '{price_col}' not found. Skipping.")
            return dominant_cycle_metrics
        df = df[[price_col]].dropna()
        if len(df) < seasonality_period * 2: # Need at least 2 seasons
            logging.error(f"  Error: Insufficient data ({len(df)} points) for period {seasonality_period}. Need at least {seasonality_period*2}. Skipping.")
            return dominant_cycle_metrics

        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df = df.dropna()

        # Use log-transformed price
        if (df[price_col] <= 0).any():
            logging.warning("  Warning: Non-positive price values found. Adding small constant before log transform.")
            df[price_col] = df[price_col] + 0.0001
        log_price = np.log(df[price_col])
        # --- Ensure DatetimeIndex for log_price ---
        if 'datetime' in df_raw.columns:
            log_price.index = pd.to_datetime(df_raw['datetime'])
        elif not isinstance(log_price.index, pd.DatetimeIndex):
            try:
                log_price.index = pd.to_datetime(log_price.index)
            except Exception:
                pass
        logging.info(f"  Prepared log_price series. Length: {len(log_price)}")

    # --- Model Selection & Fitting (Seasonality Drift) ---
    with timing(f"{timeframe_name}_decomposition"):
        logging.info("  Decomposing series to handle trend and seasonality drift...")
        trend, seasonality, residuals = None, None, None
        ssm_successful, dhr_successful = False, False

        # Approach 1: State Space Model (SSM) with stochastic seasonality
        try:
            logging.info("  Attempting State Space Model (SSM)...")
            # Ensure frequency is set for SSM if possible
            inferred_freq = pd.infer_freq(log_price.index)
            ssm_model = UnobservedComponents(
                log_price,
                level='local linear trend', # Stochastic level and slope
                seasonal=int(seasonality_period),
                stochastic_seasonal=True,
                freq=inferred_freq # Use inferred frequency
            )
            ssm_results = ssm_model.fit(maxiter=200, disp=False) # Increased maxiter
            logging.info(f"  SSM fitted successfully. AIC: {ssm_results.aic:.2f}")
            trend = ssm_results.states.smoothed['level']
            # Correctly extract seasonality based on how statsmodels names states
            seasonal_components = [col for col in ssm_results.states.smoothed.columns if 'seasonal' in col]
            if seasonal_components:
              seasonality = ssm_results.states.smoothed[seasonal_components].sum(axis=1)
            else:
                seasonality = pd.Series(0, index=log_price.index) # No seasonality found
            residuals = ssm_results.resid
            ssm_successful = True
        except Exception as e:
            logging.error(f"  SSM fitting failed: {e}. Trying DHR...")
            # Approach 2: Dynamic Harmonic Regression (DHR)
            try:
                seasonality, residuals = dynamic_harmonic_regression(log_price, period=seasonality_period, max_order=3)
                # Estimate trend from the DHR residuals (price - DHR seasonality)
                cycle_trend_resid, trend_hp = sm.tsa.filters.hpfilter(log_price - seasonality, lamb=1600) # lambda typical for daily, adjust if needed
                trend = pd.Series(trend_hp, index=log_price.index)
                dhr_successful = True
            except Exception as e_dhr:
                logging.error(f"  DHR fitting also failed: {e_dhr}. Using simple detrending.")
                # Fallback: Simple Detrending
                x = np.arange(len(log_price))
                trend_params = np.polyfit(x, log_price, 1)
                trend = pd.Series(trend_params[0] * x + trend_params[1], index=log_price.index)
                residuals = log_price - trend
                seasonality = pd.Series(0, index=log_price.index) # Assume no seasonality in fallback

        if residuals is None:
          residuals = log_price - (trend if trend is not None else 0) - (seasonality if seasonality is not None else 0)

        logging.info(f"  Decomposition complete. Mean absolute residual: {np.abs(residuals).mean():.6f}")

    # --- Cycle Detection (FFT) ---
    with timing(f"{timeframe_name}_fft_cycle_detection"):
        logging.info("  Detecting cycles in residuals using FFT...")
        cycles_df = pd.DataFrame()
        dominant_cycle = None

        residuals_values = residuals.dropna().values
        n = len(residuals_values)
        if n < 30:
            logging.error("  Error: Not enough residual data points for FFT analysis.")
        else:
            fft_values = fft(residuals_values)
            sample_freq = fftfreq(n, d=1.0)
            positive_freq_indices = np.where((sample_freq > 1/n) & (sample_freq <= 0.5))[0] # Exclude 0 freq
            if len(positive_freq_indices) == 0:
                logging.warning("  No valid positive frequencies found for FFT.")
            else:
              power = np.abs(fft_values[positive_freq_indices])**2
              freq = sample_freq[positive_freq_indices]
              periods = 1.0 / freq
              # Filter periods: > 2 units and < N/2 (half the series length)
              valid_period_mask = (periods > 2) & (periods < n / 2)
              if not np.any(valid_period_mask):
                 logging.warning("  No cycles found within valid period range (2 to N/2).")
              else:
                periods = periods[valid_period_mask]
                power = power[valid_period_mask]
                freq = freq[valid_period_mask]
                peaks, _ = find_peaks(power, height=np.mean(power), distance=max(1, int(len(power)*0.01)) ) # Dynamic distance
                if len(peaks) == 0:
                    logging.warning("  No significant peaks found in the power spectrum.")
                else:
                    cycles_df = pd.DataFrame({'period': periods[peaks], 'power': power[peaks], 'frequency': freq[peaks]})
                    cycles_df = cycles_df.sort_values('power', ascending=False)
                    logging.info(f"  Found {len(cycles_df)} potential cycles.")

                    # --- Cycle Validation & Ranking ---
                    logging.info("  Validating and ranking cycles...")
                    bartels_scores = []
                    phases = []
                    valid_cycles = []
                    for period_val in cycles_df['period']:
                        bartels_score = bartels_test(residuals, period_val)
                        phase = estimate_current_phase(residuals, period_val)
                        # Keep cycle if Bartels score is reasonably low (e.g., < 0.8)
                        if bartels_score < 0.8:
                            valid_cycles.append(period_val)
                            bartels_scores.append(bartels_score)
                            phases.append(phase)

                    if not valid_cycles:
                        logging.warning("  No cycles passed Bartels validation (score < 0.8).")
                    else:
                        cycles_df = cycles_df[cycles_df['period'].isin(valid_cycles)].copy()
                        cycles_df['bartels_score'] = bartels_scores
                        cycles_df['current_phase'] = phases
                        cycles_df['strength'] = cycles_df['power'] * (1 - cycles_df['bartels_score'])
                        cycles_df = cycles_df.sort_values('strength', ascending=False).reset_index(drop=True)
                        dominant_cycle = cycles_df.iloc[0]
                        logging.info(f"  Dominant cycle identified: {dominant_cycle['period']:.2f} {timeframe_name} units")
                        
                        # Update metrics to return
                        dominant_cycle_metrics['period'] = dominant_cycle['period']
                        dominant_cycle_metrics['significance'] = (1 - dominant_cycle['bartels_score'])
                        dominant_cycle_metrics['strength'] = dominant_cycle['strength']
                        dominant_cycle_metrics['confidence'] = (1 - dominant_cycle['bartels_score']) * 100

    # --- Generate HTML Report ---
    with timing(f"{timeframe_name}_report_generation"):
        output_filename = os.path.join(REPORTS_DIR, f"bitcoin_cycle_analysis_{timeframe_name.replace(' ', '_')}.html")
        save_plotly_html_report(
            log_price=log_price,
            trend=trend,
            seasonality=seasonality,
            residuals=residuals,
            cycles_df=cycles_df,
            dominant_cycle=dominant_cycle,
            ssm_successful=ssm_successful,
            dhr_successful=dhr_successful,
            timeframe_name=timeframe_name,
            output_file=output_filename,
            seasonality_period=seasonality_period,
            resample_rule=resample_rule
        )
    
    return dominant_cycle_metrics

# --- Step 3: Summary Report Function ---

def generate_summary_report(results_list, output_file=os.path.join(REPORTS_DIR, "cycle_summary_report.html")):
    """Generates an HTML report with a heatmap summarizing cycle characteristics across timeframes, ranked by sustainability."""
    logging.info("\n--- Generating Summary Report ---")
    if not results_list:
        logging.warning("No results to summarize.")
        return

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    results_df = results_df.set_index('timeframe')
    
    # Normalize strength for better heatmap visualization (0 to 1 scale)
    max_strength = results_df['strength'].max()
    if max_strength > 0:
      results_df['normalized_strength'] = results_df['strength'] / max_strength
    else:
      results_df['normalized_strength'] = 0
      
    # Composite sustainability score
    results_df['sustainability_score'] = results_df['significance'] * results_df['normalized_strength']
    # Sort by sustainability
    results_df = results_df.sort_values('sustainability_score', ascending=False)
    
    # Select metrics for heatmap
    heatmap_data = results_df[['significance', 'normalized_strength', 'confidence']].fillna(0)
    # Round for display
    heatmap_data_rounded = heatmap_data.round(2)
    
    # Create Plotly Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis', # Choose a colorscale
        text=heatmap_data_rounded.values, # Show rounded values on cells
        texttemplate="%{text}",
        hoverongaps=False,
        colorbar_title='Metric Value'
    ))

    fig.update_layout(
        title='Cycle Characteristics Summary Across Timeframes (Ranked by Sustainability)',
        xaxis_title='Metric',
        yaxis_title='Timeframe (Most Sustainable at Top)',
        yaxis_autorange='reversed', # Show timeframes top to bottom
        height=400 + len(results_df)*30 # Adjust height based on number of timeframes
    )

    # --- Generate HTML Parts ---
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    summary_explanation = """
    <h2>How to Read the Summary Heatmap</h2>
    <p>This heatmap compares the characteristics of the <strong>dominant cycle</strong> found in each analyzed timeframe.</p>
    <ul>
        <li><strong>Timeframe</strong>: The data resolution (e.g., Daily, Weekly).</li>
        <li><strong>Significance</strong>: Based on the Bartels Test (0 to 1). Higher values (brighter colors) indicate the cycle is less likely due to random chance (more statistically significant / "sustainable").</li>
        <li><strong>Normalized Strength</strong>: Cycle strength (Power * Significance), scaled from 0 to 1 across all timeframes. Higher values (brighter colors) indicate a more pronounced cycle relative to others, potentially having a larger impact (more "predictable" impact).</li>
        <li><strong>Confidence</strong>: Same as Significance, but expressed as a percentage (0% to 100%).</li>
        <li><strong>Sustainability Score</strong>: Composite score (Significance × Normalized Strength). The higher, the more robust and sustainable the detected cycles are for that timeframe.</li>
    </ul>
    <p>Timeframes are <b>ranked from most to least sustainable</b> (top to bottom). Look for timeframes with bright colors across both 'Significance' and 'Normalized Strength' as these indicate potentially more robust and impactful cycles.</p>
    """

    # Add a table of top timeframes
    top_table = """
    <h2>Top Ranked Timeframes by Sustainability</h2>
    <table border="1" cellpadding="5" cellspacing="0">
      <tr><th>Rank</th><th>Timeframe</th><th>Sustainability Score</th><th>Significance</th><th>Strength</th><th>Confidence</th></tr>
    """
    for i, (tf, row) in enumerate(results_df.head(10).iterrows()):
        top_table += f"<tr><td>{i+1}</td><td>{tf}</td><td>{row['sustainability_score']:.3f}</td><td>{row['significance']:.2f}</td><td>{row['normalized_strength']:.2f}</td><td>{row['confidence']:.1f}%</td></tr>"
    top_table += "</table>"

    # Assemble HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bitcoin Cycle Analysis Summary</title>
        <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; color: #333; max-width: 1000px; margin: auto; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #2980b9; margin-top: 30px; }}
            .plotly-graph-div {{ margin: 30px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .footer {{ margin-top: 50px; border-top: 1px solid #ddd; padding-top: 20px; font-size: 0.9em; color: #7f8c8d; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th {{ background-color: #3498db; color: white; text-align: left; padding: 8px; }}
            td {{ padding: 8px; border: 1px solid #ddd; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Bitcoin Cycle Analysis Summary Report</h1>
        <p>Summary generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}.</p>
        {fig_html}
        {top_table}
        {summary_explanation}
        <div class="footer">Generated by Enhanced Cycle Scanner Algorithm.</div>
    </body>
    </html>
    """

    with open(output_file, "w", encoding='utf-8') as f:
        f.write(html_content)
    logging.info(f"Summary report saved as '{output_file}'")

def fetch_all_data(timeframes):
    """Download all data sequentially and return a dict of DataFrames."""
    data_dict = {}
    for tf_config in timeframes:
        try:
            logging.info(f"Fetching data for {tf_config['name']}...")
            if tf_config['type'] == 'intraday':
                df = fetch_intraday_btc(
                    interval=tf_config['interval'],
                    days_back=tf_config['days_back'],
                    api_token=API_KEY
                )
            else:
                df = fetch_eod_btc(
                    period=tf_config['period_param'],
                    start_date=tf_config['start_date'],
                    api_token=API_KEY
                )
            if df is not None and not df.empty:
                data_dict[tf_config['name']] = df
                logging.info(f"  Downloaded {len(df)} rows for {tf_config['name']}")
            else:
                logging.error(f"  No data for {tf_config['name']}")
        except Exception as e:
            logging.error(f"  Exception while downloading {tf_config['name']}: {e}")
            logging.error(traceback.format_exc())
    return data_dict

# --- Step 4: Main Execution Block ---

def process_timeframe_with_data(args):
    tf_config, df = args
    try:
        label = f"analyze_timeframe_{tf_config['name']}"
        with timing(label):
            return process_timeframe_with_df(tf_config, df)
    except Exception as e:
        logging.error(f"  Analysis failed for {tf_config['name']}: {e}")
        logging.error(traceback.format_exc())
        return None

def process_timeframe_with_df(tf_config, df):
    # This is the same as process_timeframe, but uses the already-downloaded df
    logging.info(f"\n--- Processing Timeframe: {tf_config['name']} ---")
    try:
        # Data quality check
        if df is None or df.empty:
            logging.error(f"  Error: Empty dataframe for {tf_config['name']}. Skipping.")
            return None
        if len(df) < 20:
            logging.error(f"  Error: Insufficient data points ({len(df)}) for {tf_config['name']}. Skipping.")
            return None
        # Select price column, handle potential missing adjusted_close
        price_col = tf_config['price_col']
        if price_col not in df.columns and 'close' in df.columns:
            logging.warning(f"  Warning: '{price_col}' not found, using 'close'.")
            price_col = 'close'
        elif price_col not in df.columns:
            logging.error(f"  Error: Price column '{price_col}' or 'close' not found. Skipping.")
            return None
        # Get seasonality period
        seasonality_period = tf_config.get('period') or tf_config.get('period_val')
        if seasonality_period is None or seasonality_period <= 1:
            logging.error(f"  Error: Invalid seasonality period ({seasonality_period}). Skipping.")
            return None
        resample_rule = tf_config.get('rule')
        return run_analysis(df, tf_config['name'], price_col, seasonality_period, resample_rule=resample_rule)
    except Exception as e:
        logging.error(f"  Error processing {tf_config['name']} timeframe: {e}")
        logging.error(traceback.format_exc())
        return None

def resample_timeframe(args):
    tf, df_1min = args
    try:
        logging.info(f"Resampling 1m data to {tf['name']}...")
        df_tf = resample_ohlc(df_1min, tf['rule'])
        if not df_tf.empty:
            logging.info(f"  Resampled {len(df_tf)} rows for {tf['name']}")
            return (tf['name'], df_tf)
        else:
            logging.warning(f"  No data after resampling for {tf['name']}")
            return (tf['name'], None)
    except Exception as e:
        logging.error(f"  Error resampling to {tf['name']}: {e}")
        logging.error(traceback.format_exc())
        return (tf['name'], None)

# --- New: Helper to find latest major high/low and extract segment ---
def extract_latest_segment(df, price_col, min_length=300, mode='high', prominence=1.0):
    """
    Extracts the latest segment from the last major high or low.
    mode: 'high' or 'low'
    prominence: controls how major the peak/trough is
    Returns: DataFrame segment (at least min_length rows, or as many as possible)
    """
    series = df[price_col].values
    if mode == 'high':
        peaks, props = find_peaks(series, prominence=prominence)
    else:
        peaks, props = find_peaks(-series, prominence=prominence)
    if len(peaks) == 0:
        # No peaks/troughs found, fallback to last min_length
        return df.iloc[-min_length:].copy()
    last_peak = peaks[-1]
    # Ensure at least min_length
    start_idx = max(0, last_peak - (min_length - 1))
    segment = df.iloc[start_idx:].copy()
    if len(segment) < min_length:
        # Not enough data, fallback to last min_length
        segment = df.iloc[-min_length:].copy()
    return segment

# --- New: Uber-report generator ---
def generate_uber_report(variant_results, output_file=os.path.join(REPORTS_DIR, "uber_report.html")):
    """
    Summarizes and ranks all variants, grouped by instrument, with links to their summary and deep reports.
    variant_results: list of dicts with keys: 'instrument', 'timeframe', 'variant', 'score', 'summary_report', 'deep_report', ...
    """
    import operator
    from collections import defaultdict
    # Group by instrument
    grouped = defaultdict(list)
    for v in variant_results:
        grouped[v.get('instrument', 'UNKNOWN')].append(v)
    html = """
    <html><head><title>Uber Report</title></head><body>
    <h1>Uber Report: All Variants</h1>
    <div style='background:#f8f9fa;border-left:4px solid #3498db;padding:12px;margin:18px 0 24px 0;'>
    <b>How to read the directory structure:</b><br>
    <ul>
      <li>All reports are under the <b>reports/</b> directory.</li>
      <li>For each instrument, reports are in <b>reports/&lt;instrument&gt;/</b></li>
      <li>Each variant summary: <b>reports/&lt;instrument&gt;/&lt;timeframe&gt;/&lt;variant&gt;/summary.html</b></li>
      <li>Each deep report: <b>reports/&lt;instrument&gt;/bitcoin_cycle_analysis_&lt;timeframe&gt;.html</b></li>
      <li>The Uber report and visual graph are in <b>reports/uber_report.html</b> and <b>reports/uber_graph.html</b></li>
    </ul>
    </div>
    """
    for instrument, variants in grouped.items():
        html += f"<h2>{instrument}</h2>"
        ranked = sorted(variants, key=lambda x: -x.get('score', 0))
        html += "<table border='1' cellpadding='5'><tr><th>Rank</th><th>Timeframe</th><th>Variant</th><th>Score</th><th>Summary</th><th>Deep Report</th></tr>"
        for i, res in enumerate(ranked):
            link = os.path.relpath(res['summary_report'], REPORTS_DIR).replace('\\', '/')
            deep_link = os.path.relpath(res['deep_report'], REPORTS_DIR).replace('\\', '/') if 'deep_report' in res else ''
            html += f"<tr><td>{i+1}</td><td>{res['timeframe']}</td><td>{res['variant']}</td><td>{res['score']:.3f}</td><td><a href='{link}' target='_blank'>Summary</a></td><td><a href='{deep_link}' target='_blank'>Deep</a></td></tr>"
        html += "</table>"
    html += "<p>Each link opens the summary or deep report for that variant.</p></body></html>"
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(html)
    logging.info(f"Uber report saved as '{output_file}'")

# --- Module-level variable for recalc threshold (in seconds) ---
RECALC_THRESHOLD_SECONDS = 5 * 3600  # 5 hours

def is_file_stale(filepath, threshold_seconds=RECALC_THRESHOLD_SECONDS):
    """Return True if file does not exist or is older than threshold_seconds."""
    if not os.path.exists(filepath):
        return True
    mtime = os.path.getmtime(filepath)
    age = time.time() - mtime
    return age > threshold_seconds

def generate_visual_uber_graph(variant_results, output_file=os.path.join(REPORTS_DIR, "uber_graph.html")):
    """
    Generates an interactive visual graph (network) of all variants, with nodes linking to their summary reports.
    Each node represents a variant, colored/ranked by score, and links to the deep report.
    Uses vis-network (vis.js) for physics-based layout and interactivity.
    """
    # Prepare node and edge data
    # Normalize scores for color
    scores = [v.get('score', 0) for v in variant_results]
    max_score = max(scores) if scores else 1
    min_score = min(scores) if scores else 0
    def score_to_color(score):
        # Map score to a color (Viridis-like, but simple HSV for now)
        norm = (score - min_score) / (max_score - min_score + 1e-9)
        rgb = colorsys.hsv_to_rgb(0.7 - 0.7*norm, 0.7, 0.95)
        return 'rgb({},{},{})'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    nodes = []
    edges = []
    node_ids = {}
    for idx, v in enumerate(variant_results):
        node_id = idx
        label = f"{v['instrument']}\n{v['timeframe']}\n{v['variant']}"
        url = os.path.relpath(v['summary_report'], REPORTS_DIR).replace('\\', '/')
        color = score_to_color(v.get('score', 0))
        nodes.append({
            'id': node_id,
            'label': label,
            'title': f"Score: {v.get('score', 0):.2f}<br>Click to open report",
            'color': color,
            'url': url
        })
        node_ids[(v['instrument'], v['timeframe'], v['variant'])] = node_id
    # Edges: connect nodes of same instrument in order
    instruments = set(v['instrument'] for v in variant_results)
    for inst in instruments:
        inst_nodes = [v for v in variant_results if v['instrument'] == inst]
        inst_nodes_sorted = sorted(inst_nodes, key=lambda x: (x['timeframe'], x['variant']))
        for i in range(len(inst_nodes_sorted)-1):
            n1 = node_ids[(inst_nodes_sorted[i]['instrument'], inst_nodes_sorted[i]['timeframe'], inst_nodes_sorted[i]['variant'])]
            n2 = node_ids[(inst_nodes_sorted[i+1]['instrument'], inst_nodes_sorted[i+1]['timeframe'], inst_nodes_sorted[i+1]['variant'])]
            edges.append({'from': n1, 'to': n2})
    # HTML/JS for vis-network
    html = f"""
    <html>
    <head>
      <title>Uber Variant Graph</title>
      <script type="text/javascript" src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
      <link href="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.css" rel="stylesheet" type="text/css" />
      <style>
        #mynetwork {{ width: 100%; height: 800px; border: 1px solid lightgray; }}
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: auto; }}
      </style>
    </head>
    <body>
      <h1>Cycle Variant Network (Physics-enabled)</h1>
      <div style='background:#f8f9fa;border-left:4px solid #3498db;padding:12px;margin:18px 0 24px 0;'>
        <b>How to use:</b> Nodes can be <b>dragged</b> and the network will rearrange dynamically. Click a node to open its summary report.<br>
      </div>
      <div id="mynetwork"></div>
      <script type="text/javascript">
        var nodes = {json.dumps(nodes)};
        var edges = {json.dumps(edges)};
        var container = document.getElementById('mynetwork');
        var data = {{ nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) }};
        var options = {{
          nodes: {{ shape: 'dot', size: 22, font: {{ size: 16 }} }},
          edges: {{ width: 2, color: {{ color: '#888', highlight: '#3498db' }} }},
          physics: {{ enabled: true, barnesHut: {{ gravitationalConstant: -30000, springLength: 180, springConstant: 0.04 }} }},
          interaction: {{ hover: true, tooltipDelay: 100, dragNodes: true }}
        }};
        var network = new vis.Network(container, data, options);
        network.on('click', function(params) {{
          if(params.nodes.length > 0) {{
            var node = nodes.find(n => n.id === params.nodes[0]);
            if(node && node.url) {{ window.open(node.url, '_blank'); }}
          }}
        }});
      </script>
      <p>Each node is a variant. Drag to rearrange. Click to open its report.</p>
    </body>
    </html>
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    logging.info(f"Uber graph saved as '{output_file}'")

# --- Standalone runner for visual uber graph ---
def run_visual_uber_graph():
    uber_graph_path = os.path.join(REPORTS_DIR, "uber_graph.html")
    if not is_file_stale(uber_graph_path):
        logging.info(f"Uber graph is fresh (<5h), not regenerating.")
        return
    variant_results = []
    for instrument in INSTRUMENTS:
        inst_dir = get_report_dir(instrument)
        for root, dirs, files in os.walk(inst_dir):
            for file in files:
                if file == 'summary.html':
                    summary_path = os.path.join(root, file)
                    rel = os.path.relpath(summary_path, REPORTS_DIR)
                    parts = rel.split(os.sep)
                    if len(parts) >= 4:
                        instrument, timeframe, variant = parts[0], parts[1], parts[2]
                    else:
                        instrument, timeframe, variant = 'unknown', 'unknown', 'unknown'
                    # Try to extract score from HTML (look for Sustainability Score)
                    score = 0
                    try:
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            html = f.read()
                            import re
                            m = re.search(r'Sustainability Score.*?<td>([0-9.]+)</td>', html, re.DOTALL)
                            if m:
                                score = float(m.group(1))
                    except Exception:
                        pass
                    # Deep report path
                    deep_report = os.path.join(inst_dir, f"bitcoin_cycle_analysis_{timeframe}.html")
                    variant_results.append({
                        'instrument': instrument,
                        'timeframe': timeframe,
                        'variant': variant,
                        'score': score,
                        'summary_report': summary_path,
                        'deep_report': deep_report
                    })
    if not variant_results:
        logging.warning("No variant summary reports found for uber graph.")
        return
    generate_visual_uber_graph(variant_results, output_file=uber_graph_path)

# --- Optionally, run this at the end if desired ---
if __name__ == "__main__":
    # Set up logging
    log_file = setup_logging()
    logging.info("Enhanced Cycle Scanner Algorithm starting")
    
    # Download or load 1-minute data
    with timing("fetch_and_cache_1min_data"):
        df_1min = fetch_and_cache_1min_data()
    if df_1min is None or df_1min.empty:
        logging.error("No 1-minute BTC data available. Exiting.")
        sys.exit(1)
    
    # Define all timeframes to analyze (in minutes)
    TIMEFRAME_MINUTES = [1, 5, 15, 30, 60, 120, 240, 360, 720, 1440, 10080, 43200] # 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w, 1mo
    TIMEFRAME_RULES = {
        1: '1T', 5: '5T', 15: '15T', 30: '30T', 60: '1H', 120: '2H', 240: '4H', 360: '6H', 720: '12H', 1440: '1D', 10080: '1W', 43200: '1M'
    }
    # Map for seasonality period (e.g. 24 for 1h, 7 for 1d, etc.)
    SEASONALITY_MAP = {
        1: 1440, 5: 288, 15: 96, 30: 48, 60: 24, 120: 12, 240: 6, 360: 4, 720: 2, 1440: 7, 10080: 52, 43200: 12
    }
    timeframes = []
    for mins in TIMEFRAME_MINUTES:
        rule = TIMEFRAME_RULES[mins]
        name = rule
        period = SEASONALITY_MAP[mins]
        timeframes.append({
            'name': name,
            'rule': rule,
            'price_col': 'close',
            'period': period
        })
    # Parallel resampling
    data_dict = {}
    max_workers = max(1, multiprocessing.cpu_count() - 4)
    logging.info(f"Parallel resampling with {max_workers} worker processes")
    with timing("resample_all_timeframes"):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            resample_args = [(tf, df_1min) for tf in timeframes]
            for tf_name, df_tf in executor.map(resample_timeframe, resample_args):
                if df_tf is not None:
                    data_dict[tf_name] = df_tf
    # Prepare args for multiprocessing: only those with data
    # --- New: For each timeframe, generate variants (high/low, lengths) ---
    segment_lengths = [300, 500, 1000]
    modes = ['high', 'low']
    variant_args = []
    variant_meta = []
    for tf in timeframes:
        tf_name = tf['name']
        if tf_name not in data_dict:
            continue
        df = data_dict[tf_name]
        for mode in modes:
            for seglen in segment_lengths:
                segment = extract_latest_segment(df, tf['price_col'], min_length=seglen, mode=mode, prominence=1.0)
                variant = f"{mode}_from_last_{seglen}"
                variant_args.append((tf, segment))
                variant_meta.append({'timeframe': tf_name, 'variant': variant, 'seglen': seglen, 'mode': mode})
    all_variant_results = []
    executor = None
    logging.info(f"Running analysis for all variants with {max_workers} worker processes")
    try:
        with timing("analyze_all_variants"):
            executor = ProcessPoolExecutor(max_workers=max_workers)
            future_to_meta = {executor.submit(process_timeframe_with_data, args): meta for args, meta in zip(variant_args, variant_meta)}
            for future in future_to_meta:
                meta = future_to_meta[future]
                try:
                    result = future.result()
                    if result is not None:
                        # Save summary report for this variant in its own directory
                        variant_dir = os.path.join(REPORTS_DIR, meta['timeframe'], meta['variant'])
                        os.makedirs(variant_dir, exist_ok=True)
                        summary_report = os.path.join(variant_dir, "summary.html")
                        # Generate summary report for this single variant
                        generate_summary_report([result], output_file=summary_report)
                        meta['score'] = result.get('sustainability_score', result.get('strength', 0))
                        meta['summary_report'] = summary_report
                        all_variant_results.append(meta)
                        logging.info(f"Completed analysis for {meta['timeframe']} {meta['variant']}")
                    else:
                        logging.warning(f"Analysis failed for {meta['timeframe']} {meta['variant']}")
                except Exception as exc:
                    logging.error(f"Analysis for {meta['timeframe']} {meta['variant']} generated an exception: {exc}")
                    logging.error(traceback.format_exc())
    except Exception as e:
        logging.error(f"Error in multiprocessing: {e}")
        logging.error(traceback.format_exc())
    finally:
        if executor:
            executor.shutdown(wait=True)
    # Generate the uber-report after processing all variants
    if all_variant_results:
        try:
            with timing("generate_uber_report"):
                generate_uber_report(all_variant_results)
        except Exception as e:
            logging.error(f"Error generating uber report: {e}")
            logging.error(traceback.format_exc())
    else:
        logging.warning("No valid results to generate uber report")
    
    # Log timing report
    logging.info("\n--- Timing Report (seconds) ---")
    for label, elapsed in sorted(TIMING_STATS.items(), key=lambda x: -x[1]):
        logging.info(f"{label}: {elapsed:.2f}s")
    
    logging.info("\n--- All analyses complete ---")
    logging.info(f"Log file saved at: {log_file}")

    # After all analyses complete, generate the visual uber graph (if not stale)
    run_visual_uber_graph()

# Helper class to capture stdout and stderr to a log file
class LoggingTee:
    def __init__(self, file_path, mode='a'):
        self.file = open(file_path, mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
    
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close() 

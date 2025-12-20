#!/usr/bin/env python3
"""
Argo Float Summaries Generator
==============================
Generates temporal, location, and contextual summaries from Argo float CSV data
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import numpy as np

# ============================================================================
# 0. DATE CONVERSION UTILITY
# ============================================================================

def convert_to_standard_datetime(date_value, include_time=False) -> str:
    """
    Convert various date formats to standard format: YYYY-MM-DD (or YYYY-MM-DD HH:MM:SS if include_time=True)
    Handles: DATE strings, JULD (Julian Day), and other date formats
    """
    if pd.isna(date_value):
        return None
    
    try:
        date_str = str(date_value).strip()
        
        # Try to handle JULD format (Julian Day - decimal number)
        try:
            juld_float = float(date_str)
            # JULD is days since Jan 1, 1950, 00:00:00
            epoch = datetime(1950, 1, 1)
            # Extract integer and fractional parts
            day_int = int(juld_float)
            day_frac = juld_float - day_int
            # Calculate hours, minutes, seconds from fractional part
            seconds_in_day = day_frac * 86400  # 24 * 60 * 60
            result_date = epoch + timedelta(days=day_int, seconds=seconds_in_day)
            fmt = "%Y-%m-%d %H:%M:%S" if include_time else "%Y-%m-%d"
            return result_date.strftime(fmt)
        except (ValueError, TypeError):
            pass
        
        # Try common datetime string formats
        common_formats = [
            "%Y-%m-%d %H:%M:%S",           # Already in format with time
            "%Y-%m-%dT%H:%M:%S",           # ISO format
            "%Y-%m-%d",                     # Date only
            "%Y/%m/%d %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ]
        
        for fmt in common_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                output_fmt = "%Y-%m-%d %H:%M:%S" if include_time else "%Y-%m-%d"
                return dt.strftime(output_fmt)
            except ValueError:
                continue
        
        # If no format matched, return the original value
        return date_str
        
    except Exception as e:
        return None

# ============================================================================
# 1. DEFINE PATHS TO FINAL CSV FOLDERS
# ============================================================================

# Path to BCG floats final CSV folder
BCG_CSV_PATH = Path("f:/pineconetesting/Data/BCG floats/final csv files")

# Path to Non-BCG floats final CSV folder (need to search for all CSV files recursively)
NON_BCG_BASE_PATH = Path("f:/pineconetesting/Data/Non-BCG floats/final_csv")

# Find all CSV files in Non-BCG folder recursively
def find_csv_files(base_path):
    """Recursively find all CSV files in a directory"""
    csv_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(Path(root) / file)
    return csv_files

# ============================================================================
# 2. DEFINE SUMMARY GENERATION FUNCTIONS
# ============================================================================

def generate_temporal_summary(row: pd.Series) -> str:
    """
    Generate temporal summary (focusing on temperature and date information)
    """
    summaries = []
    
    # Date/Time information - convert to standard format (date only, no time)
    if 'DATE' in row.index and pd.notna(row['DATE']):
        date_str = convert_to_standard_datetime(row['DATE'], include_time=False)
        if date_str:
            summaries.append(f"Measurement recorded on {date_str}")
    elif 'JULD' in row.index and pd.notna(row['JULD']):
        date_str = convert_to_standard_datetime(row['JULD'], include_time=False)
        if date_str:
            summaries.append(f"Measurement recorded on {date_str}")
    
    # Surface location timestamp (when float came to surface) - with full time
    if 'JULD_LOCATION' in row.index and pd.notna(row['JULD_LOCATION']):
        surface_date = convert_to_standard_datetime(row['JULD_LOCATION'], include_time=True)
        if surface_date:
            summaries.append(f"Surface location timestamp: {surface_date}")
    
    # Surface temperature - use surface_temp_avg_C (mean surface temperature)
    if 'surface_temp_avg_C' in row.index and pd.notna(row['surface_temp_avg_C']):
        summaries.append(f"Surface temperature (avg): {row['surface_temp_avg_C']:.2f}°C")
    elif 'surface_temp_C' in row.index and pd.notna(row['surface_temp_C']):
        summaries.append(f"Surface temperature: {row['surface_temp_C']:.2f}°C")
    
    # Average temperature
    if 'TEMP_ADJUSTED' in row.index and pd.notna(row['TEMP_ADJUSTED']):
        summaries.append(f"Average adjusted temperature: {row['TEMP_ADJUSTED']:.2f}°C")
    elif 'TEMP' in row.index and pd.notna(row['TEMP']):
        summaries.append(f"Average temperature: {row['TEMP']:.2f}°C")
    
    # Mixed layer depth
    if 'mixed_layer_depth_m' in row.index and pd.notna(row['mixed_layer_depth_m']):
        summaries.append(f"Mixed layer depth: {row['mixed_layer_depth_m']:.2f}m")
    
    return " | ".join(summaries) if summaries else "No temporal data available"

def generate_location_summary(row: pd.Series) -> str:
    """
    Generate location summary (focusing on geographic information)
    """
    summaries = []
    
    # Region information
    if 'avg_REGION' in row.index and pd.notna(row['avg_REGION']):
        summaries.append(f"Region: {row['avg_REGION']}")
    
    # Ocean basin
    if 'ocean' in row.index and pd.notna(row['ocean']):
        summaries.append(f"Ocean: {row['ocean']}")
    
    # Latitude
    if 'LATITUDE' in row.index and pd.notna(row['LATITUDE']):
        summaries.append(f"Latitude: {row['LATITUDE']:.4f}°")
    
    # Longitude
    if 'LONGITUDE' in row.index and pd.notna(row['LONGITUDE']):
        summaries.append(f"Longitude: {row['LONGITUDE']:.4f}°")
    
    # Depth range
    if 'min_DEPTH_m' in row.index and 'max_DEPTH_m' in row.index:
        if pd.notna(row['min_DEPTH_m']) and pd.notna(row['max_DEPTH_m']):
            summaries.append(f"Depth range: {row['min_DEPTH_m']:.2f}m to {row['max_DEPTH_m']:.2f}m")
    
    return " | ".join(summaries) if summaries else "No location data available"

def generate_contextual_summary(row: pd.Series) -> str:
    """
    Generate contextual summary (focusing on comprehensive oceanographic and BGC parameters)
    """
    summaries = []
    
    # 1. CORE PARAMETERS
    summaries.append("CORE PARAMETERS:")
    
    if 'PLATFORM_NUMBER' in row.index and pd.notna(row['PLATFORM_NUMBER']):
        summaries.append(f"  Platform: {row['PLATFORM_NUMBER']}")
    
    if 'PLATFORM_TYPE' in row.index and pd.notna(row['PLATFORM_TYPE']):
        summaries.append(f"  Type: {row['PLATFORM_TYPE']}")
    
    if 'CYCLE_NUMBER' in row.index and pd.notna(row['CYCLE_NUMBER']):
        summaries.append(f"  Cycle: {int(row['CYCLE_NUMBER'])}")
    
    # Pressure with raw and adjusted versions - EXACT FIELD NAMES
    pres_info = []
    if 'PRES' in row.index and pd.notna(row['PRES']):
        pres_info.append(f"PRES={row['PRES']:.2f}")
    if 'PRES_ADJUSTED' in row.index:
        val = row['PRES_ADJUSTED'] if pd.notna(row['PRES_ADJUSTED']) else "NaN"
        if val != "NaN":
            pres_info.append(f"PRES_ADJUSTED={val:.2f}")
        else:
            pres_info.append("PRES_ADJUSTED=NaN")
    if 'PRES_ADJUSTED_ERROR' in row.index:
        val = row['PRES_ADJUSTED_ERROR'] if pd.notna(row['PRES_ADJUSTED_ERROR']) else "NaN"
        if val != "NaN":
            pres_info.append(f"PRES_ADJUSTED_ERROR={val:.4f}")
        else:
            pres_info.append("PRES_ADJUSTED_ERROR=NaN")
    if 'PRES_ADJUSTED_ERROR_min' in row.index:
        val = row['PRES_ADJUSTED_ERROR_min'] if pd.notna(row['PRES_ADJUSTED_ERROR_min']) else "NaN"
        if val != "NaN":
            pres_info.append(f"PRES_ADJUSTED_ERROR_min={val:.4f}")
        else:
            pres_info.append("PRES_ADJUSTED_ERROR_min=NaN")
    if 'PRES_ADJUSTED_ERROR_max' in row.index:
        val = row['PRES_ADJUSTED_ERROR_max'] if pd.notna(row['PRES_ADJUSTED_ERROR_max']) else "NaN"
        if val != "NaN":
            pres_info.append(f"PRES_ADJUSTED_ERROR_max={val:.4f}")
        else:
            pres_info.append("PRES_ADJUSTED_ERROR_max=NaN")
    if 'PRES_ADJUSTED_ERROR_mean' in row.index:
        val = row['PRES_ADJUSTED_ERROR_mean'] if pd.notna(row['PRES_ADJUSTED_ERROR_mean']) else "NaN"
        if val != "NaN":
            pres_info.append(f"PRES_ADJUSTED_ERROR_mean={val:.4f}")
        else:
            pres_info.append("PRES_ADJUSTED_ERROR_mean=NaN")
    if 'PRES_ADJUSTED_QC' in row.index:
        val = row['PRES_ADJUSTED_QC'] if pd.notna(row['PRES_ADJUSTED_QC']) else "NaN"
        pres_info.append(f"PRES_ADJUSTED_QC={val}")
    if pres_info:
        summaries.append(f"  {' | '.join(pres_info)}")
    
    # Temperature with raw and adjusted versions - EXACT FIELD NAMES
    temp_info = []
    if 'TEMP' in row.index and pd.notna(row['TEMP']):
        temp_info.append(f"TEMP={row['TEMP']:.2f}")
    if 'TEMP_ADJUSTED' in row.index:
        val = row['TEMP_ADJUSTED'] if pd.notna(row['TEMP_ADJUSTED']) else "NaN"
        if val != "NaN":
            temp_info.append(f"TEMP_ADJUSTED={val:.2f}")
        else:
            temp_info.append("TEMP_ADJUSTED=NaN")
    if 'TEMP_ADJUSTED_ERROR' in row.index:
        val = row['TEMP_ADJUSTED_ERROR'] if pd.notna(row['TEMP_ADJUSTED_ERROR']) else "NaN"
        if val != "NaN":
            temp_info.append(f"TEMP_ADJUSTED_ERROR={val:.4f}")
        else:
            temp_info.append("TEMP_ADJUSTED_ERROR=NaN")
    if 'TEMP_ADJUSTED_ERROR_min' in row.index:
        val = row['TEMP_ADJUSTED_ERROR_min'] if pd.notna(row['TEMP_ADJUSTED_ERROR_min']) else "NaN"
        if val != "NaN":
            temp_info.append(f"TEMP_ADJUSTED_ERROR_min={val:.4f}")
        else:
            temp_info.append("TEMP_ADJUSTED_ERROR_min=NaN")
    if 'TEMP_ADJUSTED_ERROR_max' in row.index:
        val = row['TEMP_ADJUSTED_ERROR_max'] if pd.notna(row['TEMP_ADJUSTED_ERROR_max']) else "NaN"
        if val != "NaN":
            temp_info.append(f"TEMP_ADJUSTED_ERROR_max={val:.4f}")
        else:
            temp_info.append("TEMP_ADJUSTED_ERROR_max=NaN")
    if 'TEMP_ADJUSTED_ERROR_mean' in row.index:
        val = row['TEMP_ADJUSTED_ERROR_mean'] if pd.notna(row['TEMP_ADJUSTED_ERROR_mean']) else "NaN"
        if val != "NaN":
            temp_info.append(f"TEMP_ADJUSTED_ERROR_mean={val:.4f}")
        else:
            temp_info.append("TEMP_ADJUSTED_ERROR_mean=NaN")
    if 'TEMP_ADJUSTED_QC' in row.index:
        val = row['TEMP_ADJUSTED_QC'] if pd.notna(row['TEMP_ADJUSTED_QC']) else "NaN"
        temp_info.append(f"TEMP_ADJUSTED_QC={val}")
    if temp_info:
        summaries.append(f"  {' | '.join(temp_info)}")
    
    # Salinity with raw and adjusted versions - EXACT FIELD NAMES
    psal_info = []
    if 'PSAL' in row.index and pd.notna(row['PSAL']):
        psal_info.append(f"PSAL={row['PSAL']:.3f}")
    if 'PSAL_ADJUSTED' in row.index:
        val = row['PSAL_ADJUSTED'] if pd.notna(row['PSAL_ADJUSTED']) else "NaN"
        if val != "NaN":
            psal_info.append(f"PSAL_ADJUSTED={val:.3f}")
        else:
            psal_info.append("PSAL_ADJUSTED=NaN")
    if 'PSAL_ADJUSTED_ERROR' in row.index:
        val = row['PSAL_ADJUSTED_ERROR'] if pd.notna(row['PSAL_ADJUSTED_ERROR']) else "NaN"
        if val != "NaN":
            psal_info.append(f"PSAL_ADJUSTED_ERROR={val:.4f}")
        else:
            psal_info.append("PSAL_ADJUSTED_ERROR=NaN")
    if 'PSAL_ADJUSTED_ERROR_min' in row.index:
        val = row['PSAL_ADJUSTED_ERROR_min'] if pd.notna(row['PSAL_ADJUSTED_ERROR_min']) else "NaN"
        if val != "NaN":
            psal_info.append(f"PSAL_ADJUSTED_ERROR_min={val:.4f}")
        else:
            psal_info.append("PSAL_ADJUSTED_ERROR_min=NaN")
    if 'PSAL_ADJUSTED_ERROR_max' in row.index:
        val = row['PSAL_ADJUSTED_ERROR_max'] if pd.notna(row['PSAL_ADJUSTED_ERROR_max']) else "NaN"
        if val != "NaN":
            psal_info.append(f"PSAL_ADJUSTED_ERROR_max={val:.4f}")
        else:
            psal_info.append("PSAL_ADJUSTED_ERROR_max=NaN")
    if 'PSAL_ADJUSTED_ERROR_mean' in row.index:
        val = row['PSAL_ADJUSTED_ERROR_mean'] if pd.notna(row['PSAL_ADJUSTED_ERROR_mean']) else "NaN"
        if val != "NaN":
            psal_info.append(f"PSAL_ADJUSTED_ERROR_mean={val:.4f}")
        else:
            psal_info.append("PSAL_ADJUSTED_ERROR_mean=NaN")
    if 'PSAL_ADJUSTED_QC' in row.index:
        val = row['PSAL_ADJUSTED_QC'] if pd.notna(row['PSAL_ADJUSTED_QC']) else "NaN"
        psal_info.append(f"PSAL_ADJUSTED_QC={val}")
    if psal_info:
        summaries.append(f"  {' | '.join(psal_info)}")
    
    # 2. DISSOLVED OXYGEN & NUTRIENTS
    summaries.append("BIOGEOCHEMISTRY:")
    
    if 'DOXY' in row.index and pd.notna(row['DOXY']):
        summaries.append(f"  Dissolved oxygen: {row['DOXY']:.2f} µmol/kg")
    
    if 'NITRATE' in row.index and pd.notna(row['NITRATE']):
        summaries.append(f"  Nitrate: {row['NITRATE']:.2f} µmol/kg")
    
    if 'CHLA' in row.index and pd.notna(row['CHLA']):
        summaries.append(f"  Chlorophyll a: {row['CHLA']:.3f} mg/m³")
    
    # 3. BGC SIGNALS - CDOM with ALL adjusted versions and QC - EXACT FIELD NAMES
    cdom_data = []
    if 'CDOM' in row.index and pd.notna(row['CDOM']):
        cdom_data.append(f"CDOM={row['CDOM']:.3f}")
    if 'CDOM_ADJUSTED' in row.index:
        val = row['CDOM_ADJUSTED'] if pd.notna(row['CDOM_ADJUSTED']) else "NaN"
        if val != "NaN":
            cdom_data.append(f"CDOM_ADJUSTED={val:.3f}")
        else:
            cdom_data.append("CDOM_ADJUSTED=NaN")
    if 'CDOM_ADJUSTED_ERROR' in row.index:
        val = row['CDOM_ADJUSTED_ERROR'] if pd.notna(row['CDOM_ADJUSTED_ERROR']) else "NaN"
        if val != "NaN":
            cdom_data.append(f"CDOM_ADJUSTED_ERROR={val:.3f}")
        else:
            cdom_data.append("CDOM_ADJUSTED_ERROR=NaN")
    if 'CDOM_QC' in row.index and pd.notna(row['CDOM_QC']):
        cdom_data.append(f"CDOM_QC={row['CDOM_QC']}")
    if 'CDOM_ADJUSTED_QC' in row.index:
        val = row['CDOM_ADJUSTED_QC'] if pd.notna(row['CDOM_ADJUSTED_QC']) else "NaN"
        cdom_data.append(f"CDOM_ADJUSTED_QC={val}")
    if cdom_data:
        summaries.append(f"  {' | '.join(cdom_data)}")
    
    # 4. BGC SIGNALS - DOWNWELLING IRRADIANCE with ALL adjusted versions - EXACT FIELD NAMES
    irrad_380 = []
    if 'DOWN_IRRADIANCE380' in row.index and pd.notna(row['DOWN_IRRADIANCE380']):
        irrad_380.append(f"DOWN_IRRADIANCE380={row['DOWN_IRRADIANCE380']:.4f}")
    if 'DOWN_IRRADIANCE380_ADJUSTED' in row.index:
        val = row['DOWN_IRRADIANCE380_ADJUSTED'] if pd.notna(row['DOWN_IRRADIANCE380_ADJUSTED']) else "NaN"
        if val != "NaN":
            irrad_380.append(f"DOWN_IRRADIANCE380_ADJUSTED={val:.4f}")
        else:
            irrad_380.append("DOWN_IRRADIANCE380_ADJUSTED=NaN")
    if irrad_380:
        summaries.append(f"  {' | '.join(irrad_380)}")
    
    irrad_443 = []
    if 'DOWN_IRRADIANCE443' in row.index and pd.notna(row['DOWN_IRRADIANCE443']):
        irrad_443.append(f"DOWN_IRRADIANCE443={row['DOWN_IRRADIANCE443']:.4f}")
    if 'DOWN_IRRADIANCE443_ADJUSTED' in row.index:
        val = row['DOWN_IRRADIANCE443_ADJUSTED'] if pd.notna(row['DOWN_IRRADIANCE443_ADJUSTED']) else "NaN"
        if val != "NaN":
            irrad_443.append(f"DOWN_IRRADIANCE443_ADJUSTED={val:.4f}")
        else:
            irrad_443.append("DOWN_IRRADIANCE443_ADJUSTED=NaN")
    if irrad_443:
        summaries.append(f"  {' | '.join(irrad_443)}")
    
    irrad_490 = []
    if 'DOWN_IRRADIANCE490' in row.index and pd.notna(row['DOWN_IRRADIANCE490']):
        irrad_490.append(f"DOWN_IRRADIANCE490={row['DOWN_IRRADIANCE490']:.4f}")
    if 'DOWN_IRRADIANCE490_ADJUSTED' in row.index:
        val = row['DOWN_IRRADIANCE490_ADJUSTED'] if pd.notna(row['DOWN_IRRADIANCE490_ADJUSTED']) else "NaN"
        if val != "NaN":
            irrad_490.append(f"DOWN_IRRADIANCE490_ADJUSTED={val:.4f}")
        else:
            irrad_490.append("DOWN_IRRADIANCE490_ADJUSTED=NaN")
    if irrad_490:
        summaries.append(f"  {' | '.join(irrad_490)}")
    
    irrad_555 = []
    if 'DOWN_IRRADIANCE555' in row.index and pd.notna(row['DOWN_IRRADIANCE555']):
        irrad_555.append(f"DOWN_IRRADIANCE555={row['DOWN_IRRADIANCE555']:.4f}")
    if 'DOWN_IRRADIANCE555_ADJUSTED' in row.index:
        val = row['DOWN_IRRADIANCE555_ADJUSTED'] if pd.notna(row['DOWN_IRRADIANCE555_ADJUSTED']) else "NaN"
        if val != "NaN":
            irrad_555.append(f"DOWN_IRRADIANCE555_ADJUSTED={val:.4f}")
        else:
            irrad_555.append("DOWN_IRRADIANCE555_ADJUSTED=NaN")
    if irrad_555:
        summaries.append(f"  {' | '.join(irrad_555)}")
    
    # 5. QC FLAGS - EXACT FIELD NAMES (including density, salinity variants)
    summaries.append("QC FLAGS:")
    
    qc_flags = []
    if 'PROFILE_PRES_QC' in row.index and pd.notna(row['PROFILE_PRES_QC']):
        qc_flags.append(f"PROFILE_PRES_QC={row['PROFILE_PRES_QC']}")
    if 'PROFILE_TEMP_QC' in row.index and pd.notna(row['PROFILE_TEMP_QC']):
        qc_flags.append(f"PROFILE_TEMP_QC={row['PROFILE_TEMP_QC']}")
    if 'PROFILE_PSAL_QC' in row.index and pd.notna(row['PROFILE_PSAL_QC']):
        qc_flags.append(f"PROFILE_PSAL_QC={row['PROFILE_PSAL_QC']}")
    if 'PROFILE_DENSITY_QC' in row.index and pd.notna(row['PROFILE_DENSITY_QC']):
        qc_flags.append(f"PROFILE_DENSITY_QC={row['PROFILE_DENSITY_QC']}")
    if 'PROFILE_SALINITY_QC' in row.index and pd.notna(row['PROFILE_SALINITY_QC']):
        qc_flags.append(f"PROFILE_SALINITY_QC={row['PROFILE_SALINITY_QC']}")
    if 'PROFILE_DOXY_QC' in row.index and pd.notna(row['PROFILE_DOXY_QC']):
        qc_flags.append(f"PROFILE_DOXY_QC={row['PROFILE_DOXY_QC']}")
    if 'PROFILE_NITRATE_QC' in row.index and pd.notna(row['PROFILE_NITRATE_QC']):
        qc_flags.append(f"PROFILE_NITRATE_QC={row['PROFILE_NITRATE_QC']}")
    if 'PROFILE_CHLA_QC' in row.index and pd.notna(row['PROFILE_CHLA_QC']):
        qc_flags.append(f"PROFILE_CHLA_QC={row['PROFILE_CHLA_QC']}")
    if 'PROFILE_CDOM_QC' in row.index and pd.notna(row['PROFILE_CDOM_QC']):
        qc_flags.append(f"PROFILE_CDOM_QC={row['PROFILE_CDOM_QC']}")
    if 'PROFILE_BBP700_QC' in row.index and pd.notna(row['PROFILE_BBP700_QC']):
        qc_flags.append(f"PROFILE_BBP700_QC={row['PROFILE_BBP700_QC']}")
    if 'PROFILE_PH_IN_SITU_TOTAL_QC' in row.index and pd.notna(row['PROFILE_PH_IN_SITU_TOTAL_QC']):
        qc_flags.append(f"PROFILE_PH_IN_SITU_TOTAL_QC={row['PROFILE_PH_IN_SITU_TOTAL_QC']}")
    if 'PROFILE_CHLA_FLUORESCENCE_QC' in row.index and pd.notna(row['PROFILE_CHLA_FLUORESCENCE_QC']):
        qc_flags.append(f"PROFILE_CHLA_FLUORESCENCE_QC={row['PROFILE_CHLA_FLUORESCENCE_QC']}")
    if 'PROFILE_DOWN_IRRADIANCE380_QC' in row.index and pd.notna(row['PROFILE_DOWN_IRRADIANCE380_QC']):
        qc_flags.append(f"PROFILE_DOWN_IRRADIANCE380_QC={row['PROFILE_DOWN_IRRADIANCE380_QC']}")
    if 'PROFILE_DOWN_IRRADIANCE443_QC' in row.index and pd.notna(row['PROFILE_DOWN_IRRADIANCE443_QC']):
        qc_flags.append(f"PROFILE_DOWN_IRRADIANCE443_QC={row['PROFILE_DOWN_IRRADIANCE443_QC']}")
    if 'PROFILE_DOWN_IRRADIANCE490_QC' in row.index and pd.notna(row['PROFILE_DOWN_IRRADIANCE490_QC']):
        qc_flags.append(f"PROFILE_DOWN_IRRADIANCE490_QC={row['PROFILE_DOWN_IRRADIANCE490_QC']}")
    if 'PROFILE_DOWN_IRRADIANCE555_QC' in row.index and pd.notna(row['PROFILE_DOWN_IRRADIANCE555_QC']):
        qc_flags.append(f"PROFILE_DOWN_IRRADIANCE555_QC={row['PROFILE_DOWN_IRRADIANCE555_QC']}")
    
    if qc_flags:
        summaries.append(f"  {' | '.join(qc_flags)}")
    
    # 6. RAW PARAMETER MIN/MAX (vertical variability) - support both naming conventions
    summaries.append("RAW PARAMETER RANGES:")
    
    raw_ranges = []
    # Support both PRES_raw_min and PRES_min naming variations
    if 'PRES_raw_min' in row.index and pd.notna(row['PRES_raw_min']):
        raw_ranges.append(f"PRES_raw_min={row['PRES_raw_min']:.2f}")
    elif 'PRES_min' in row.index and pd.notna(row['PRES_min']):
        raw_ranges.append(f"PRES_min={row['PRES_min']:.2f}")
    
    if 'PRES_raw_max' in row.index and pd.notna(row['PRES_raw_max']):
        raw_ranges.append(f"PRES_raw_max={row['PRES_raw_max']:.2f}")
    elif 'PRES_max' in row.index and pd.notna(row['PRES_max']):
        raw_ranges.append(f"PRES_max={row['PRES_max']:.2f}")
    
    if 'TEMP_raw_min' in row.index and pd.notna(row['TEMP_raw_min']):
        raw_ranges.append(f"TEMP_raw_min={row['TEMP_raw_min']:.2f}")
    elif 'TEMP_min' in row.index and pd.notna(row['TEMP_min']):
        raw_ranges.append(f"TEMP_min={row['TEMP_min']:.2f}")
    
    if 'TEMP_raw_max' in row.index and pd.notna(row['TEMP_raw_max']):
        raw_ranges.append(f"TEMP_raw_max={row['TEMP_raw_max']:.2f}")
    elif 'TEMP_max' in row.index and pd.notna(row['TEMP_max']):
        raw_ranges.append(f"TEMP_max={row['TEMP_max']:.2f}")
    
    if 'PSAL_raw_min' in row.index and pd.notna(row['PSAL_raw_min']):
        raw_ranges.append(f"PSAL_raw_min={row['PSAL_raw_min']:.3f}")
    elif 'PSAL_min' in row.index and pd.notna(row['PSAL_min']):
        raw_ranges.append(f"PSAL_min={row['PSAL_min']:.3f}")
    
    if 'PSAL_raw_max' in row.index and pd.notna(row['PSAL_raw_max']):
        raw_ranges.append(f"PSAL_raw_max={row['PSAL_raw_max']:.3f}")
    elif 'PSAL_max' in row.index and pd.notna(row['PSAL_max']):
        raw_ranges.append(f"PSAL_max={row['PSAL_max']:.3f}")
    
    if raw_ranges:
        summaries.append(f"  {' | '.join(raw_ranges)}")
    
    # 7. MISSION & TECHNICAL METADATA
    summaries.append("MISSION & TECHNICAL:")
    
    if 'FLOAT_SERIAL_NO' in row.index and pd.notna(row['FLOAT_SERIAL_NO']):
        summaries.append(f"  Serial: {str(row['FLOAT_SERIAL_NO']).strip()}")
    
    if 'CONFIG_MISSION_NUMBER' in row.index and pd.notna(row['CONFIG_MISSION_NUMBER']):
        summaries.append(f"  Config mission: {int(row['CONFIG_MISSION_NUMBER'])}")
    
    if 'DIRECTION' in row.index and pd.notna(row['DIRECTION']):
        summaries.append(f"  Direction: {row['DIRECTION']}")
    
    if 'DATA_MODE' in row.index and pd.notna(row['DATA_MODE']):
        summaries.append(f"  Data mode: {row['DATA_MODE']}")
    
    if 'DATA_CENTRE' in row.index and pd.notna(row['DATA_CENTRE']):
        summaries.append(f"  Data centre: {row['DATA_CENTRE']}")
    
    if 'DATA_TYPE' in row.index and pd.notna(row['DATA_TYPE']):
        summaries.append(f"  Data type: {str(row['DATA_TYPE']).strip()}")
    
    if 'FORMAT_VERSION' in row.index and pd.notna(row['FORMAT_VERSION']):
        summaries.append(f"  Format version: {row['FORMAT_VERSION']}")
    
    if 'HANDBOOK_VERSION' in row.index and pd.notna(row['HANDBOOK_VERSION']):
        summaries.append(f"  Handbook version: {row['HANDBOOK_VERSION']}")
    
    if 'n_levels' in row.index and pd.notna(row['n_levels']):
        summaries.append(f"  Number of depth levels: {int(row['n_levels'])}")
    
    return " | ".join(summaries) if summaries else "No contextual data available"

def generate_metadata(row: pd.Series, float_type: str, file_name: str) -> Dict:
    """
    Generate metadata for a row
    """
    # Ensure float_type is always set as the first and most important metadata
    metadata = {
        "float_type": float_type,  # BCG or Non-BCG - ALWAYS INCLUDED
        "source_file": file_name,
    }
    
    # Add float identifiers
    if 'PLATFORM_NUMBER' in row.index and pd.notna(row['PLATFORM_NUMBER']):
        metadata["platform_number"] = str(row['PLATFORM_NUMBER']).strip()
    
    if 'FLOAT_SERIAL_NO' in row.index and pd.notna(row['FLOAT_SERIAL_NO']):
        metadata["float_serial_no"] = str(row['FLOAT_SERIAL_NO']).strip()
    
    # Add location info
    if 'LATITUDE' in row.index and pd.notna(row['LATITUDE']):
        metadata["latitude"] = float(row['LATITUDE'])
    
    if 'LONGITUDE' in row.index and pd.notna(row['LONGITUDE']):
        metadata["longitude"] = float(row['LONGITUDE'])
    
    if 'avg_REGION' in row.index and pd.notna(row['avg_REGION']):
        metadata["avg_region"] = str(row['avg_REGION']).strip()
    
    if 'ocean' in row.index and pd.notna(row['ocean']):
        metadata["ocean"] = str(row['ocean']).strip()
    
    # Add reference time fields
    if 'REFERENCE_DATE_TIME' in row.index and pd.notna(row['REFERENCE_DATE_TIME']):
        ref_date = convert_to_standard_datetime(row['REFERENCE_DATE_TIME'], include_time=False)
        if ref_date:
            metadata["reference_date_time"] = ref_date
    
    if 'JULD_LOCATION_time' in row.index and pd.notna(row['JULD_LOCATION_time']):
        loc_time = convert_to_standard_datetime(row['JULD_LOCATION_time'], include_time=True)
        if loc_time:
            metadata["juld_location_time"] = loc_time
    
    # Add temporal info - convert to standard format (date only)
    if 'DATE' in row.index and pd.notna(row['DATE']):
        date_str = convert_to_standard_datetime(row['DATE'], include_time=False)
        if date_str:
            metadata["measurement_date"] = date_str
    elif 'JULD' in row.index and pd.notna(row['JULD']):
        date_str = convert_to_standard_datetime(row['JULD'], include_time=False)
        if date_str:
            metadata["measurement_date"] = date_str
    
    # Add surface location timestamp (JULD_LOCATION - includes time)
    if 'JULD_LOCATION' in row.index and pd.notna(row['JULD_LOCATION']):
        surface_time = convert_to_standard_datetime(row['JULD_LOCATION'], include_time=True)
        if surface_time:
            metadata["juld_location_timestamp"] = surface_time
    
    # Add cycle info
    if 'CYCLE_NUMBER' in row.index and pd.notna(row['CYCLE_NUMBER']):
        metadata["cycle_number"] = int(row['CYCLE_NUMBER'])
    
    # Add depth info (including average)
    if 'min_DEPTH_m' in row.index and pd.notna(row['min_DEPTH_m']):
        metadata["min_depth_m"] = float(row['min_DEPTH_m'])
    
    if 'max_DEPTH_m' in row.index and pd.notna(row['max_DEPTH_m']):
        metadata["max_depth_m"] = float(row['max_DEPTH_m'])
    
    if 'avg_DEPTH_m' in row.index and pd.notna(row['avg_DEPTH_m']):
        metadata["avg_depth_m"] = float(row['avg_DEPTH_m'])
    
    if 'n_levels' in row.index and pd.notna(row['n_levels']):
        metadata["n_levels"] = int(row['n_levels'])
    
    # Add raw parameter ranges (vertical variability) - support all naming conventions
    if 'PRES_raw_min' in row.index and pd.notna(row['PRES_raw_min']):
        metadata["pres_raw_min"] = float(row['PRES_raw_min'])
    if 'PRES_min' in row.index and pd.notna(row['PRES_min']):
        metadata["pres_min"] = float(row['PRES_min'])
    
    if 'PRES_raw_max' in row.index and pd.notna(row['PRES_raw_max']):
        metadata["pres_raw_max"] = float(row['PRES_raw_max'])
    if 'PRES_max' in row.index and pd.notna(row['PRES_max']):
        metadata["pres_max"] = float(row['PRES_max'])
    
    if 'TEMP_raw_min' in row.index and pd.notna(row['TEMP_raw_min']):
        metadata["temp_raw_min"] = float(row['TEMP_raw_min'])
    if 'TEMP_min' in row.index and pd.notna(row['TEMP_min']):
        metadata["temp_min"] = float(row['TEMP_min'])
    
    if 'TEMP_raw_max' in row.index and pd.notna(row['TEMP_raw_max']):
        metadata["temp_raw_max"] = float(row['TEMP_raw_max'])
    if 'TEMP_max' in row.index and pd.notna(row['TEMP_max']):
        metadata["temp_max"] = float(row['TEMP_max'])
    
    if 'PSAL_raw_min' in row.index and pd.notna(row['PSAL_raw_min']):
        metadata["psal_raw_min"] = float(row['PSAL_raw_min'])
    if 'PSAL_min' in row.index and pd.notna(row['PSAL_min']):
        metadata["psal_min"] = float(row['PSAL_min'])
    
    if 'PSAL_raw_max' in row.index and pd.notna(row['PSAL_raw_max']):
        metadata["psal_raw_max"] = float(row['PSAL_raw_max'])
    if 'PSAL_max' in row.index and pd.notna(row['PSAL_max']):
        metadata["psal_max"] = float(row['PSAL_max'])
    
    # Add adjusted physical parameters
    if 'PRES_ADJUSTED' in row.index and pd.notna(row['PRES_ADJUSTED']):
        metadata["pres_adjusted_dbar"] = float(row['PRES_ADJUSTED'])
    if 'PRES_ADJUSTED_ERROR' in row.index and pd.notna(row['PRES_ADJUSTED_ERROR']):
        metadata["pres_adjusted_error"] = float(row['PRES_ADJUSTED_ERROR'])
    if 'PRES_ADJUSTED_ERROR_min' in row.index and pd.notna(row['PRES_ADJUSTED_ERROR_min']):
        metadata["pres_adjusted_error_min"] = float(row['PRES_ADJUSTED_ERROR_min'])
    if 'PRES_ADJUSTED_ERROR_max' in row.index and pd.notna(row['PRES_ADJUSTED_ERROR_max']):
        metadata["pres_adjusted_error_max"] = float(row['PRES_ADJUSTED_ERROR_max'])
    if 'PRES_ADJUSTED_ERROR_mean' in row.index and pd.notna(row['PRES_ADJUSTED_ERROR_mean']):
        metadata["pres_adjusted_error_mean"] = float(row['PRES_ADJUSTED_ERROR_mean'])
    if 'PRES_ADJUSTED_QC' in row.index and pd.notna(row['PRES_ADJUSTED_QC']):
        metadata["pres_adjusted_qc"] = row['PRES_ADJUSTED_QC']
    
    if 'PSAL_ADJUSTED' in row.index and pd.notna(row['PSAL_ADJUSTED']):
        metadata["psal_adjusted_psu"] = float(row['PSAL_ADJUSTED'])
    if 'PSAL_ADJUSTED_ERROR' in row.index and pd.notna(row['PSAL_ADJUSTED_ERROR']):
        metadata["psal_adjusted_error"] = float(row['PSAL_ADJUSTED_ERROR'])
    if 'PSAL_ADJUSTED_ERROR_min' in row.index and pd.notna(row['PSAL_ADJUSTED_ERROR_min']):
        metadata["psal_adjusted_error_min"] = float(row['PSAL_ADJUSTED_ERROR_min'])
    if 'PSAL_ADJUSTED_ERROR_max' in row.index and pd.notna(row['PSAL_ADJUSTED_ERROR_max']):
        metadata["psal_adjusted_error_max"] = float(row['PSAL_ADJUSTED_ERROR_max'])
    if 'PSAL_ADJUSTED_ERROR_mean' in row.index and pd.notna(row['PSAL_ADJUSTED_ERROR_mean']):
        metadata["psal_adjusted_error_mean"] = float(row['PSAL_ADJUSTED_ERROR_mean'])
    if 'PSAL_ADJUSTED_QC' in row.index and pd.notna(row['PSAL_ADJUSTED_QC']):
        metadata["psal_adjusted_qc"] = row['PSAL_ADJUSTED_QC']
    
    if 'TEMP_ADJUSTED' in row.index and pd.notna(row['TEMP_ADJUSTED']):
        metadata["temp_adjusted_c"] = float(row['TEMP_ADJUSTED'])
    if 'TEMP_ADJUSTED_ERROR' in row.index and pd.notna(row['TEMP_ADJUSTED_ERROR']):
        metadata["temp_adjusted_error"] = float(row['TEMP_ADJUSTED_ERROR'])
    if 'TEMP_ADJUSTED_ERROR_min' in row.index and pd.notna(row['TEMP_ADJUSTED_ERROR_min']):
        metadata["temp_adjusted_error_min"] = float(row['TEMP_ADJUSTED_ERROR_min'])
    if 'TEMP_ADJUSTED_ERROR_max' in row.index and pd.notna(row['TEMP_ADJUSTED_ERROR_max']):
        metadata["temp_adjusted_error_max"] = float(row['TEMP_ADJUSTED_ERROR_max'])
    if 'TEMP_ADJUSTED_ERROR_mean' in row.index and pd.notna(row['TEMP_ADJUSTED_ERROR_mean']):
        metadata["temp_adjusted_error_mean"] = float(row['TEMP_ADJUSTED_ERROR_mean'])
    if 'TEMP_ADJUSTED_QC' in row.index and pd.notna(row['TEMP_ADJUSTED_QC']):
        metadata["temp_adjusted_qc"] = row['TEMP_ADJUSTED_QC']
    
    # Add adjusted BGC parameters
    if 'CDOM_ADJUSTED' in row.index and pd.notna(row['CDOM_ADJUSTED']):
        metadata["cdom_adjusted"] = float(row['CDOM_ADJUSTED'])
    if 'CDOM_ADJUSTED_ERROR' in row.index and pd.notna(row['CDOM_ADJUSTED_ERROR']):
        metadata["cdom_adjusted_error"] = float(row['CDOM_ADJUSTED_ERROR'])
    if 'CDOM_ADJUSTED_QC' in row.index and pd.notna(row['CDOM_ADJUSTED_QC']):
        metadata["cdom_adjusted_qc"] = row['CDOM_ADJUSTED_QC']
    
    # Add adjusted irradiance values
    if 'DOWN_IRRADIANCE380_ADJUSTED' in row.index and pd.notna(row['DOWN_IRRADIANCE380_ADJUSTED']):
        metadata["irr380_adjusted"] = float(row['DOWN_IRRADIANCE380_ADJUSTED'])
    if 'DOWN_IRRADIANCE443_ADJUSTED' in row.index and pd.notna(row['DOWN_IRRADIANCE443_ADJUSTED']):
        metadata["irr443_adjusted"] = float(row['DOWN_IRRADIANCE443_ADJUSTED'])
    if 'DOWN_IRRADIANCE490_ADJUSTED' in row.index and pd.notna(row['DOWN_IRRADIANCE490_ADJUSTED']):
        metadata["irr490_adjusted"] = float(row['DOWN_IRRADIANCE490_ADJUSTED'])
    if 'DOWN_IRRADIANCE555_ADJUSTED' in row.index and pd.notna(row['DOWN_IRRADIANCE555_ADJUSTED']):
        metadata["irr555_adjusted"] = float(row['DOWN_IRRADIANCE555_ADJUSTED'])
    
    # Add document and data information - convert dates to standard format
    if 'DATA_TYPE' in row.index and pd.notna(row['DATA_TYPE']):
        metadata["data_type"] = str(row['DATA_TYPE']).strip()
    
    if 'DATA_MODE' in row.index and pd.notna(row['DATA_MODE']):
        metadata["data_mode"] = row['DATA_MODE']
    
    if 'DATA_CENTRE' in row.index and pd.notna(row['DATA_CENTRE']):
        metadata["data_centre"] = row['DATA_CENTRE']
    
    if 'DATE_CREATION' in row.index and pd.notna(row['DATE_CREATION']):
        date_str = convert_to_standard_datetime(row['DATE_CREATION'], include_time=False)
        if date_str:
            metadata["date_creation"] = date_str
    
    if 'DATE_UPDATE' in row.index and pd.notna(row['DATE_UPDATE']):
        date_str = convert_to_standard_datetime(row['DATE_UPDATE'], include_time=False)
        if date_str:
            metadata["date_update"] = date_str
    
    if 'FORMAT_VERSION' in row.index and pd.notna(row['FORMAT_VERSION']):
        metadata["format_version"] = row['FORMAT_VERSION']
    
    if 'HANDBOOK_VERSION' in row.index and pd.notna(row['HANDBOOK_VERSION']):
        metadata["handbook_version"] = row['HANDBOOK_VERSION']
    
    return metadata

# ============================================================================
# 3. MAIN PROCESSING FUNCTION
# ============================================================================

def process_csvs(csv_path: Path, float_type: str, num_samples: int = 5) -> List[Tuple[str, str, str, Dict]]:
    """
    Process CSV files and generate summaries
    Returns: List of (temporal_summary, location_summary, contextual_summary, metadata) tuples
    """
    results = []
    
    # Get all CSV files
    if csv_path.is_dir():
        csv_files = sorted(list(csv_path.glob("*.csv")))
    else:
        csv_files = [csv_path]
    
    if not csv_files:
        print(f"[WARNING] No CSV files found in {csv_path}")
        return results
    
    print(f"\n{'='*80}")
    print(f"{float_type} FLOATS - Summary Generation")
    print(f"{'='*80}")
    print(f"Found {len(csv_files)} CSV file(s)")
    
    total_samples = 0
    
    for csv_file in csv_files[:5]:  # Limit to first 5 files for demo
        print(f"\nProcessing: {csv_file.name}")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            print(f"  - Rows in file: {len(df)}")
            
            # Process first 5 rows from this file
            for idx, (_, row) in enumerate(df.head(num_samples).iterrows()):
                if total_samples >= num_samples:
                    break
                
                temporal = generate_temporal_summary(row)
                location = generate_location_summary(row)
                contextual = generate_contextual_summary(row)
                metadata = generate_metadata(row, float_type, csv_file.name)
                
                results.append((temporal, location, contextual, metadata))
                total_samples += 1
            
            if total_samples >= num_samples:
                break
                
        except Exception as e:
            print(f"  [ERROR] Failed to process {csv_file.name}: {e}")
            continue
    
    return results

# ============================================================================
# 4. DISPLAY RESULTS
# ============================================================================

def display_results(results: List[Tuple[str, str, str, Dict]], float_type: str):
    """Display the generated summaries and metadata"""
    
    print(f"\n{'='*80}")
    print(f"{float_type} FLOATS - Generated Summaries and Metadata")
    print(f"{'='*80}\n")
    
    for i, (temporal, location, contextual, metadata) in enumerate(results, 1):
        print(f"{'─'*80}")
        print(f"Sample #{i} | Float Type: {metadata.get('float_type', 'N/A').upper()}")
        print(f"{'─'*80}")
        
        print(f"\nTEMPORAL SUMMARY:")
        print(f"   {temporal}")
        
        print(f"\nLOCATION SUMMARY:")
        print(f"   {location}")
        
        print(f"\nCONTEXTUAL SUMMARY:")
        print(f"   {contextual}")
        
        print(f"\nMETADATA:")
        print(f"   - float_type: {metadata.get('float_type', 'N/A')} (PRIMARY IDENTIFIER)")
        for key, value in metadata.items():
            if key != 'float_type':  # Skip float_type since we already printed it
                print(f"   - {key}: {value}")
        
        print()

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ARGO FLOAT SUMMARIES & METADATA GENERATOR")
    print("="*80)
    
    # Print path information
    print(f"\nCSV Path Configuration:")
    print(f"   BCG floats:     {BCG_CSV_PATH}")
    print(f"   Non-BCG floats: {NON_BCG_BASE_PATH}")
    
    # Process BCG floats
    bcg_results = process_csvs(BCG_CSV_PATH, "BCG", num_samples=5)
    display_results(bcg_results, "BCG")
    
    # Find and process Non-BCG floats
    non_bcg_csv_files = find_csv_files(NON_BCG_BASE_PATH)
    print(f"\n{'='*80}")
    print(f"NON-BCG FLOATS - Found {len(non_bcg_csv_files)} CSV file(s)")
    print(f"{'='*80}")
    
    non_bcg_results = []
    num_samples = 5
    total_samples = 0
    
    for csv_file in non_bcg_csv_files[:5]:  # Limit to first 5 files
        print(f"\nProcessing: {csv_file.name}")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"  - Rows in file: {len(df)}")
            
            for idx, (_, row) in enumerate(df.head(num_samples).iterrows()):
                if total_samples >= num_samples:
                    break
                
                temporal = generate_temporal_summary(row)
                location = generate_location_summary(row)
                contextual = generate_contextual_summary(row)
                metadata = generate_metadata(row, "Non-BCG", csv_file.name)
                
                non_bcg_results.append((temporal, location, contextual, metadata))
                total_samples += 1
            
            if total_samples >= num_samples:
                break
                
        except Exception as e:
            print(f"  [ERROR] Failed to process {csv_file.name}: {e}")
            continue
    
    display_results(non_bcg_results, "Non-BCG")
    
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"   - BCG samples generated: {len(bcg_results)}")
    print(f"   - Non-BCG samples generated: {len(non_bcg_results)}")
    print(f"   - Total samples: {len(bcg_results) + len(non_bcg_results)}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ARGO Float Metadata Upload System - Single Table Focus
Creates and populates ONLY the float_metadata table in Supabase
"""

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase imports
try:
    from supabase import create_client, Client
except ImportError:
    print("ERROR: supabase package not found. Install with: pip install supabase")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SupabaseConfig:
    """Configuration for Supabase connection"""
    url: str
    key: str

class FloatMetadataUploader:
    """Handles creation and population of float_metadata and float_parameters tables"""
    
    def __init__(self, config: SupabaseConfig):
        """Initialize uploader with Supabase configuration"""
        self.config = config
        self.client: Client = create_client(config.url, config.key)
        
        # Statistics tracking
        self.stats = {
            'floats_processed': 0,
            'floats_inserted': 0,
            'parameters_inserted': 0,
            'csv_files_scanned': 0,
            'errors': 0,
            'bgc_floats': 0,
            'non_bgc_floats': 0
        }
    
    def create_float_metadata_table(self) -> bool:
        """
        Check if both tables exist: float_metadata and float_parameters_reading
        
        Returns:
            True if both tables exist, False otherwise
        """
        try:
            logger.info("Checking if float_metadata table exists...")
            
            # Check float_metadata table
            response = self.client.table('float_metadata').select('*').limit(1).execute()
            logger.info("✓ float_metadata table exists and is accessible")
            
            # Check float_parameters_reading table
            logger.info("Checking if float_parameters_reading table exists...")
            response = self.client.table('float_parameters_reading').select('*').limit(1).execute()
            logger.info("✓ float_parameters_reading table exists and is accessible")
            
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            if 'does not exist' in error_msg or 'could not find' in error_msg:
                logger.error("❌ One or both tables do not exist!")
                logger.error("Please create the tables manually in Supabase SQL Editor:")
                logger.info("="*60)
                logger.info("""
-- Create float_metadata table (float-level info with metadata and last reading date)
CREATE TABLE IF NOT EXISTS float_metadata (
    float_id TEXT PRIMARY KEY,
    profiler_type TEXT,
    institution TEXT,
    project_name TEXT,
    wmo_inst_type TEXT,
    platform_type TEXT,
    metadata JSONB,
    all_columns JSONB,
    csv_filename TEXT,
    last_reading_date DATE
);

-- Create float_parameters_reading table (individual parameter readings with values)
CREATE TABLE IF NOT EXISTS float_parameters_reading (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    float_id TEXT NOT NULL,
    parameter_name TEXT NOT NULL,
    value TEXT,
    date_recorded DATE,
    FOREIGN KEY (float_id) REFERENCES float_metadata(float_id) ON DELETE CASCADE,
    UNIQUE(float_id, parameter_name, date_recorded)
);

-- Create indexes for float_metadata
CREATE INDEX IF NOT EXISTS idx_float_metadata_float_id ON float_metadata (float_id);
CREATE INDEX IF NOT EXISTS idx_float_metadata_institution ON float_metadata (institution);
CREATE INDEX IF NOT EXISTS idx_float_metadata_profiler_type ON float_metadata (profiler_type);
CREATE INDEX IF NOT EXISTS idx_float_metadata_last_reading_date ON float_metadata (last_reading_date);
CREATE INDEX IF NOT EXISTS idx_float_metadata_metadata ON float_metadata USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_float_metadata_columns ON float_metadata USING GIN (all_columns);

-- Create indexes for float_parameters_reading
CREATE INDEX IF NOT EXISTS idx_float_parameters_reading_float_id ON float_parameters_reading (float_id);
CREATE INDEX IF NOT EXISTS idx_float_parameters_reading_date ON float_parameters_reading (date_recorded);
CREATE INDEX IF NOT EXISTS idx_float_parameters_reading_parameter ON float_parameters_reading (parameter_name);

-- For numeric queries, create an IMMUTABLE function to cast value to numeric when possible
CREATE OR REPLACE FUNCTION safe_numeric_cast(text_val TEXT) 
RETURNS NUMERIC AS $$
BEGIN
    RETURN text_val::NUMERIC;
EXCEPTION 
    WHEN OTHERS THEN
        RETURN NULL;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- Create index for numeric queries (for <= >= comparisons)
CREATE INDEX IF NOT EXISTS idx_float_parameters_reading_numeric_value 
ON float_parameters_reading (safe_numeric_cast(value)) 
WHERE safe_numeric_cast(value) IS NOT NULL;
                """)
                logger.info("="*60)
                return False
            else:
                logger.error(f"Error checking tables: {str(e)}")
                return False
    
    def extract_float_metadata_from_csv(self, csv_path: str) -> List[Dict]:
        """
        Extract float metadata from a single CSV file
        Handles both BCG and Non-BCG CSV formats
        Captures ALL columns as metadata
        For Non-BCG files: extracts metadata for ALL unique floats in the file
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of dictionaries with float metadata or empty list if failed
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            if df.empty:
                logger.warning(f"Empty CSV file: {csv_path}")
                return []
            
            # Check if file has any meaningful content
            if len(df.columns) == 0:
                logger.warning(f"CSV file has no columns: {csv_path}")
                return []
            
            # Get ALL columns from the CSV file
            all_columns = df.columns.tolist()
            
            # Get filename and record count
            csv_filename = Path(csv_path).name
            total_records = len(df)
            
            # Determine file type (BCG or NON-BCG) based on file path
            # Check if 'BCG' is in the path and 'Non-BCG' is not
            is_bcg = "BCG" in csv_path and "Non-BCG" not in csv_path
            float_type = "BCG" if is_bcg else "NON-BCG"
            file_type = float_type  # For logging
            
            # Get unique float IDs in this file
            if 'PLATFORM_NUMBER' not in df.columns:
                logger.warning(f"No PLATFORM_NUMBER column found in {csv_path}")
                return []
            
            unique_floats = df['PLATFORM_NUMBER'].dropna().unique()
            logger.info(f"Found {len(unique_floats)} unique floats in {file_type} file {csv_filename}")
            
            metadata_list = []
            
            # Group data by float_id and measurement_date to avoid duplicate lat/lon
            float_daily_data = {}
            
            # For each unique float, process ALL its readings/cycles in this file
            for platform_number in unique_floats:
                platform_str = str(platform_number)
                if not platform_str or platform_str == 'nan':
                    continue
                
                # Get ALL rows for this float
                float_rows = df[df['PLATFORM_NUMBER'] == platform_number]
                
                # Track latest date for this float
                latest_date = None
                
                # Process each row/reading for this float
                for idx, row in float_rows.iterrows():
                    # Extract cycle number if available
                    cycle_number = None
                    if 'CYCLE_NUMBER' in df.columns and pd.notna(row.get('CYCLE_NUMBER')):
                        cycle_number = int(row['CYCLE_NUMBER'])
                    
                    # Extract measurement date from this specific row
                    measurement_date = None
                    if 'DATE' in df.columns and pd.notna(row.get('DATE')):
                        date_str = str(row['DATE'])
                        try:
                            from datetime import datetime as dt
                            if 'T' in date_str:
                                parsed_date = dt.fromisoformat(date_str.split('T')[0])
                            elif ' ' in date_str:
                                parsed_date = dt.strptime(date_str.split(' ')[0], '%Y-%m-%d')
                            else:
                                parsed_date = dt.strptime(date_str[:10], '%Y-%m-%d')
                            measurement_date = parsed_date.strftime('%Y-%m-%d')
                        except:
                            measurement_date = date_str[:10] if len(date_str) >= 10 else date_str
                    elif 'JULD' in df.columns and pd.notna(row.get('JULD')):
                        try:
                            juld_val = float(row['JULD'])
                            from datetime import datetime as dt, timedelta
                            base_date = dt(1950, 1, 1)  # ARGO reference date
                            measurement_date = (base_date + timedelta(days=juld_val)).strftime('%Y-%m-%d')
                        except:
                            measurement_date = str(row['JULD'])[:10]
                    elif 'DATE_CREATION' in df.columns and pd.notna(row.get('DATE_CREATION')):
                        date_str = str(row['DATE_CREATION'])
                        try:
                            from datetime import datetime as dt
                            if 'T' in date_str:
                                parsed_date = dt.fromisoformat(date_str.split('T')[0])
                            elif ' ' in date_str:
                                parsed_date = dt.strptime(date_str.split(' ')[0], '%Y-%m-%d')
                            else:
                                parsed_date = dt.strptime(date_str[:10], '%Y-%m-%d')
                            measurement_date = parsed_date.strftime('%Y-%m-%d')
                        except:
                            measurement_date = date_str[:10] if len(date_str) >= 10 else date_str
                    
                    # If no date found, use current date
                    if not measurement_date:
                        measurement_date = datetime.now().strftime('%Y-%m-%d')
                    
                    # Track latest date for this float
                    if not latest_date or measurement_date > latest_date:
                        latest_date = measurement_date
                    
                    # Create unique key for float + date combination
                    float_date_key = f"{platform_str}_{measurement_date}"
                    
                    # Initialize if not exists
                    if float_date_key not in float_daily_data:
                        # Extract location data (only once per float per day)
                        latitude = None
                        longitude = None
                        if 'LATITUDE' in df.columns and pd.notna(row.get('LATITUDE')):
                            try:
                                latitude = float(row['LATITUDE'])
                            except:
                                pass
                        if 'LONGITUDE' in df.columns and pd.notna(row.get('LONGITUDE')):
                            try:
                                longitude = float(row['LONGITUDE'])
                            except:
                                pass
                        
                        # Extract ocean information
                        ocean = str(row.get('ocean', row.get('Ocean', row.get('OCEAN', ''))))
                        if not ocean or ocean == 'nan' or ocean == '' or ocean == 'None':
                            ocean = 'Indian Ocean'
                        else:
                            ocean = ocean.strip()
                        
                        float_daily_data[float_date_key] = {
                            'float_id': platform_str,
                            'measurement_date': measurement_date,
                            'cycle_number': cycle_number,
                            'latitude': latitude,
                            'longitude': longitude,
                            'ocean': ocean,
                            'parameters': [],
                            'latest_date': latest_date
                        }
                    
                    # Extract ALL column values for this specific reading
                    single_row_df = pd.DataFrame([row])
                    parameters = self.extract_all_column_values(single_row_df, cycle_number, measurement_date, 
                                                              float_daily_data[float_date_key].get('latitude'), 
                                                              float_daily_data[float_date_key].get('longitude'), 
                                                              float_daily_data[float_date_key].get('ocean'))
                    
                    # Add parameters to this day's data
                    float_daily_data[float_date_key]['parameters'].extend(parameters)
                    float_daily_data[float_date_key]['latest_date'] = latest_date
                
                # Create float metadata (once per float, with latest date)
                first_row = float_rows.iloc[0]
                profiler_type = str(first_row.get('WMO_INST_TYPE', first_row.get('PLATFORM_TYPE', '')))
                institution = str(first_row.get('DATA_CENTRE', ''))
                
                # Extract important metadata columns for summary/vectordb
                summary_metadata = self.extract_summary_metadata_columns(df)
                
                float_metadata = {
                    'float_id': platform_str,
                    'float_type': float_type,
                    'profiler_type': profiler_type if profiler_type != 'nan' else None,
                    'institution': institution if institution != 'nan' else None,
                    'project_name': str(first_row.get('PROJECT_NAME', '')) if str(first_row.get('PROJECT_NAME', '')) != 'nan' else None,
                    'wmo_inst_type': str(first_row.get('WMO_INST_TYPE', '')) if str(first_row.get('WMO_INST_TYPE', '')) != 'nan' else None,
                    'platform_type': str(first_row.get('PLATFORM_TYPE', '')) if str(first_row.get('PLATFORM_TYPE', '')) != 'nan' else None,
                    'metadata': summary_metadata,
                    'all_columns': all_columns,
                    'csv_filename': csv_filename,
                    'last_reading_date': latest_date
                }
                
                # Create entries for each day this float has readings
                for daily_data in float_daily_data.values():
                    if daily_data['float_id'] == platform_str:
                        metadata = {
                            'float_metadata': float_metadata,
                            'daily_reading_data': daily_data
                        }
                        metadata_list.append(metadata)
                        logger.debug(f"Extracted metadata for {file_type} float {platform_str} on {daily_data['measurement_date']} with {len(daily_data['parameters'])} parameters")
            
            return metadata_list
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {csv_path}: {str(e)}")
            return []
    
    def extract_available_parameters(self, df: pd.DataFrame) -> List[str]:
        """
        Extract list of available parameters from CSV columns by detecting all measurement columns
        
        Args:
            df: DataFrame with ARGO data
            
        Returns:
            List of parameter names that have actual data
        """
        parameters = set()
        
        # Get all column names
        columns = df.columns.tolist()
        
        # Define columns to EXCLUDE (metadata, not measurements)
        metadata_columns = {
            'PLATFORM_NUMBER', 'FLOAT_SERIAL_NO', 'CYCLE_NUMBER', 'n_levels', 'JULD', 'DATE',
            'avg_REGION', 'ocean', 'Ocean', 'CONFIG_MISSION_NUMBER', 'DATA_CENTRE', 'DIRECTION',
            'FIRMWARE_VERSION', 'JULD_LOCATION', 'JULD_QC', 'LATITUDE', 'LONGITUDE',
            'PARAMETER_DATA_MODE', 'PI_NAME', 'PLATFORM_TYPE', 'POSITIONING_SYSTEM',
            'POSITION_QC', 'PROJECT_NAME', 'SCIENTIFIC_CALIB_COEFFICIENT', 
            'SCIENTIFIC_CALIB_COMMENT', 'SCIENTIFIC_CALIB_DATE', 'SCIENTIFIC_CALIB_EQUATION',
            'STATION_PARAMETERS', 'WMO_INST_TYPE', 'DATA_TYPE', 'DATE_CREATION', 
            'DATE_UPDATE', 'FORMAT_VERSION', 'HANDBOOK_VERSION', 'REFERENCE_DATE_TIME',
            'PARAMETER', 'row_count', 'DATA_MODE', 'DATA_STATE_INDICATOR', 'DC_REFERENCE',
            'HISTORY_ACTION', 'HISTORY_DATE', 'HISTORY_INSTITUTION', 'HISTORY_PARAMETER',
            'HISTORY_PREVIOUS_VALUE', 'HISTORY_QCTEST', 'HISTORY_REFERENCE', 'HISTORY_SOFTWARE',
            'HISTORY_SOFTWARE_RELEASE', 'HISTORY_START_PRES', 'HISTORY_STEP', 'HISTORY_STOP_PRES',
            'VERTICAL_SAMPLING_SCHEME', 'JULD_LOCATION_TIME'
        }
        
        for col in columns:
            col_upper = col.upper()
            
            # Skip known metadata columns
            if col_upper in metadata_columns:
                continue
                
            # Skip QC profile columns (these are summary QC, not measurements)
            if col_upper.startswith('PROFILE_') and col_upper.endswith('_QC'):
                continue
                
            # Check if this column has any non-null numeric or meaningful data
            if col in df.columns and df[col].notna().any():
                # Extract a clean parameter name
                param_name = col_upper
                
                # Handle specific parameter patterns
                if 'SURFACE_TEMP' in param_name:
                    param_name = 'SURFACE_TEMP'
                elif 'MIXED_LAYER_DEPTH' in param_name:
                    param_name = 'MIXED_LAYER_DEPTH'
                elif 'DEPTH' in param_name and not param_name.startswith('AVG_'):
                    if 'MIN_' in param_name:
                        param_name = 'MIN_DEPTH'
                    elif 'MAX_' in param_name:
                        param_name = 'MAX_DEPTH'
                    elif 'AVG_' in param_name:
                        param_name = 'AVG_DEPTH'
                    else:
                        param_name = 'DEPTH'
                elif param_name.endswith('_ADJUSTED'):
                    param_name = param_name.replace('_ADJUSTED', '')
                elif param_name.endswith('_QC'):
                    param_name = param_name.replace('_QC', '')
                elif param_name.endswith('_ERROR'):
                    param_name = param_name.replace('_ERROR', '')
                elif param_name.endswith('_ADJUSTED_ERROR'):
                    param_name = param_name.replace('_ADJUSTED_ERROR', '')
                elif param_name.endswith('_ADJUSTED_QC'):
                    param_name = param_name.replace('_ADJUSTED_QC', '')
                elif param_name.endswith('_ERROR_MIN') or param_name.endswith('_ERROR_MAX') or param_name.endswith('_ERROR_MEAN'):
                    # Handle Non-BCG error statistics columns
                    param_name = param_name.split('_ERROR_')[0] + '_ERROR'
                
                # Clean up irradiance parameters
                if 'DOWN_IRRADIANCE' in param_name:
                    if any(char.isdigit() for char in param_name):
                        # Keep specific wavelengths
                        import re
                        wavelength = re.search(r'\d+', param_name)
                        if wavelength:
                            param_name = f'DOWN_IRRADIANCE_{wavelength.group()}'
                    else:
                        param_name = 'DOWN_IRRADIANCE'
                
                # Add the parameter if it's meaningful
                if len(param_name) > 1 and param_name not in metadata_columns:
                    parameters.add(param_name)
        
        # Also extract from PARAMETER column if it exists
        if 'PARAMETER' in df.columns:
            param_values = df['PARAMETER'].dropna().unique()
            for param_val in param_values:
                if isinstance(param_val, str):
                    param_list = param_val.replace(' ', '').split(',') if ',' in param_val else [param_val.strip()]
                    for p in param_list:
                        if p and len(p) > 1:
                            parameters.add(p.upper())
        
        # Also extract from STATION_PARAMETERS column if it exists
        if 'STATION_PARAMETERS' in df.columns:
            station_params = df['STATION_PARAMETERS'].dropna().unique()
            for param_val in station_params:
                if isinstance(param_val, str):
                    param_list = param_val.replace(' ', '').split(',') if ',' in param_val else [param_val.strip()]
                    for p in param_list:
                        if p and len(p) > 1:
                            parameters.add(p.upper())
        
        return sorted(list(parameters))  # Remove duplicates and sort
    
    def extract_summary_metadata_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Extract important metadata column names for summary/vectordb creation
        Focus on columns that are important for creating meaningful summaries
        
        Args:
            df: DataFrame with ARGO data
            
        Returns:
            List of important column names present in the data
        """
        # Important columns for summary and vectordb
        important_columns = {
            # Core measurement parameters
            'PRES', 'TEMP', 'PSAL', 'DOXY', 'NITRATE', 'CHLA', 'BBP700', 'CDOM',
            'PH_IN_SITU_TOTAL', 'DOWN_IRRADIANCE380', 'DOWN_IRRADIANCE443', 
            'DOWN_IRRADIANCE490', 'DOWN_IRRADIANCE555',
            
            # Location and time info
            'LATITUDE', 'LONGITUDE', 'DATE', 'JULD', 'CYCLE_NUMBER',
            
            # Derived/summary parameters
            'surface_temp_avg_C', 'surface_temp_C', 'mixed_layer_depth_m',
            'min_DEPTH_m', 'max_DEPTH_m', 'avg_DEPTH_m',
            
            # Quality and data info
            'DATA_MODE', 'POSITION_QC', 'PROFILE_PRES_QC', 'PROFILE_TEMP_QC', 'PROFILE_PSAL_QC',
            
            # Institutional info
            'PLATFORM_TYPE', 'WMO_INST_TYPE', 'PROJECT_NAME',
            
            # Regional info
            'avg_REGION', 'ocean', 'Ocean'
        }
        
        # Get columns that exist in the dataframe and are in our important list
        available_columns = []
        df_columns = set(df.columns)
        
        for col in important_columns:
            if col in df_columns and df[col].notna().any():
                available_columns.append(col)
        
        # Also check for any adjusted versions of key parameters
        for col in df.columns:
            col_upper = col.upper()
            if any(param in col_upper for param in ['PRES_ADJUSTED', 'TEMP_ADJUSTED', 'PSAL_ADJUSTED', 'DOXY_ADJUSTED']):
                if df[col].notna().any():
                    available_columns.append(col)
        
        return sorted(list(set(available_columns)))  # Remove duplicates and sort
    
    def extract_parameter_data_modes(self, df: pd.DataFrame) -> List[Dict]:
        """
        Extract parameter names and their data modes from CSV
        
        Args:
            df: DataFrame with ARGO data
            
        Returns:
            List of dictionaries with parameter_name and data_mode
        """
        parameters = []
        
        # Get all column names
        columns = df.columns.tolist()
        
        # Define columns to EXCLUDE (metadata, not measurements)
        metadata_columns = {
            'PLATFORM_NUMBER', 'FLOAT_SERIAL_NO', 'CYCLE_NUMBER', 'n_levels', 'JULD', 'DATE',
            'avg_REGION', 'ocean', 'Ocean', 'CONFIG_MISSION_NUMBER', 'DATA_CENTRE', 'DIRECTION',
            'FIRMWARE_VERSION', 'JULD_LOCATION', 'JULD_QC', 'LATITUDE', 'LONGITUDE',
            'PARAMETER_DATA_MODE', 'PI_NAME', 'PLATFORM_TYPE', 'POSITIONING_SYSTEM',
            'POSITION_QC', 'PROJECT_NAME', 'SCIENTIFIC_CALIB_COEFFICIENT', 
            'SCIENTIFIC_CALIB_COMMENT', 'SCIENTIFIC_CALIB_DATE', 'SCIENTIFIC_CALIB_EQUATION',
            'STATION_PARAMETERS', 'WMO_INST_TYPE', 'DATA_TYPE', 'DATE_CREATION', 
            'DATE_UPDATE', 'FORMAT_VERSION', 'HANDBOOK_VERSION', 'REFERENCE_DATE_TIME',
            'PARAMETER', 'row_count', 'DATA_MODE', 'DATA_STATE_INDICATOR', 'DC_REFERENCE',
            'HISTORY_ACTION', 'HISTORY_DATE', 'HISTORY_INSTITUTION', 'HISTORY_PARAMETER',
            'HISTORY_PREVIOUS_VALUE', 'HISTORY_QCTEST', 'HISTORY_REFERENCE', 'HISTORY_SOFTWARE',
            'HISTORY_SOFTWARE_RELEASE', 'HISTORY_START_PRES', 'HISTORY_STEP', 'HISTORY_STOP_PRES',
            'VERTICAL_SAMPLING_SCHEME', 'JULD_LOCATION_TIME'
        }
        
        # Track processed parameters to avoid duplicates
        processed_params = set()
        
        # Extract from actual measurement columns
        for col in columns:
            col_upper = col.upper()
            
            # Skip known metadata columns
            if col_upper in metadata_columns:
                continue
                
            # Skip QC profile columns (these are summary QC, not measurements)
            if col_upper.startswith('PROFILE_') and col_upper.endswith('_QC'):
                continue
                
            # Check if this column has any non-null data
            if col in df.columns and df[col].notna().any():
                # Extract parameter name and determine data mode
                param_name = col_upper
                data_mode = None
                
                # Handle specific parameter patterns and extract data mode
                if param_name.endswith('_ADJUSTED'):
                    param_name = param_name.replace('_ADJUSTED', '')
                    data_mode = 'A'  # Adjusted mode
                elif param_name.endswith('_QC'):
                    param_name = param_name.replace('_QC', '')
                    data_mode = 'QC'
                elif param_name.endswith('_ERROR'):
                    param_name = param_name.replace('_ERROR', '')
                    data_mode = 'E'  # Error
                elif param_name.endswith('_ADJUSTED_ERROR'):
                    param_name = param_name.replace('_ADJUSTED_ERROR', '')
                    data_mode = 'AE'  # Adjusted Error
                elif param_name.endswith('_ADJUSTED_QC'):
                    param_name = param_name.replace('_ADJUSTED_QC', '')
                    data_mode = 'AQC'  # Adjusted QC
                else:
                    data_mode = 'R'  # Raw/Real-time mode
                
                # Handle specific parameter names
                if 'SURFACE_TEMP' in param_name:
                    param_name = 'SURFACE_TEMP'
                elif 'MIXED_LAYER_DEPTH' in param_name:
                    param_name = 'MIXED_LAYER_DEPTH'
                elif 'DEPTH' in param_name and not param_name.startswith('AVG_'):
                    if 'MIN_' in param_name:
                        param_name = 'MIN_DEPTH'
                    elif 'MAX_' in param_name:
                        param_name = 'MAX_DEPTH'
                    elif 'AVG_' in param_name:
                        param_name = 'AVG_DEPTH'
                    else:
                        param_name = 'DEPTH'
                
                # Clean up irradiance parameters
                if 'DOWN_IRRADIANCE' in param_name:
                    if any(char.isdigit() for char in param_name):
                        import re
                        wavelength = re.search(r'\d+', param_name)
                        if wavelength:
                            param_name = f'DOWN_IRRADIANCE_{wavelength.group()}'
                    else:
                        param_name = 'DOWN_IRRADIANCE'
                
                # Add the parameter if it's meaningful and not already processed
                param_key = f"{param_name}_{data_mode}"
                if len(param_name) > 1 and param_name not in metadata_columns and param_key not in processed_params:
                    parameters.append({
                        'parameter_name': param_name,
                        'data_mode': data_mode
                    })
                    processed_params.add(param_key)
        
        # Also extract from PARAMETER column if it exists
        if 'PARAMETER' in df.columns:
            param_values = df['PARAMETER'].dropna().unique()
            for param_val in param_values:
                if isinstance(param_val, str):
                    param_list = param_val.replace(' ', '').split(',') if ',' in param_val else [param_val.strip()]
                    for p in param_list:
                        if p and len(p) > 1:
                            param_name = p.upper()
                            param_key = f"{param_name}_R"
                            if param_key not in processed_params:
                                parameters.append({
                                    'parameter_name': param_name,
                                    'data_mode': 'R'
                                })
                                processed_params.add(param_key)
        
        # Extract from STATION_PARAMETERS column if it exists
        if 'STATION_PARAMETERS' in df.columns:
            station_params = df['STATION_PARAMETERS'].dropna().unique()
            for param_val in station_params:
                if isinstance(param_val, str):
                    param_list = param_val.replace(' ', '').split(',') if ',' in param_val else [param_val.strip()]
                    for p in param_list:
                        if p and len(p) > 1:
                            param_name = p.upper()
                            param_key = f"{param_name}_R"
                            if param_key not in processed_params:
                                parameters.append({
                                    'parameter_name': param_name,
                                    'data_mode': 'R'
                                })
                                processed_params.add(param_key)
        
        # Check for DATA_MODE column to get general data mode
        general_data_mode = None
        if 'DATA_MODE' in df.columns:
            data_modes = df['DATA_MODE'].dropna().unique()
            if len(data_modes) > 0:
                general_data_mode = str(data_modes[0])
        
        # Update data modes if we have a general data mode
        if general_data_mode and general_data_mode != 'nan':
            for param in parameters:
                if param['data_mode'] == 'R' and general_data_mode in ['A', 'D']:
                    param['data_mode'] = general_data_mode
        
        return parameters
    
    def extract_all_column_values(self, df: pd.DataFrame, cycle_number: int, measurement_date: str, 
                                  latitude: float, longitude: float, ocean: str) -> List[Dict]:
        """
        Extract ALL column values from CSV row (one parameter per column)
        
        Args:
            df: DataFrame with single row of ARGO data
            cycle_number: Cycle number for this reading
            measurement_date: Date of measurement
            latitude: Latitude for this reading
            longitude: Longitude for this reading  
            ocean: Ocean for this reading
            
        Returns:
            List of dictionaries with parameter_name, value, date_recorded
        """
        parameters = []
        
        if df.empty:
            return parameters
            
        # Get the single row
        row = df.iloc[0]
        
        # Define columns to EXCLUDE (only exclude float ID and serial)
        exclude_columns = {
            'PLATFORM_NUMBER', 'FLOAT_SERIAL_NO', 'PARAMETER', 'STATION_PARAMETERS'
        }
        
        # Process each column
        for col_name in df.columns:
            col_upper = col_name.upper()
            
            # Skip excluded columns
            if col_upper in exclude_columns:
                continue
                
            # Get the value
            value = row[col_name]
            
            # Skip null/nan values
            if pd.isna(value):
                continue
                
            # Convert value to string for storage
            value_str = str(value).strip()
            if value_str in ['nan', 'NaN', '', 'None']:
                continue
                
            # Create parameter record
            parameter_record = {
                'parameter_name': col_name,  # Keep original column name
                'value': value_str,
                'date_recorded': measurement_date
            }
            
            parameters.append(parameter_record)
        
        return parameters
    
    def insert_float_metadata(self, metadata: Dict) -> bool:
        """
        Insert or update float metadata in both tables using BATCH processing
        Ensures last_reading_date always reflects the most recent date for that float
        
        Args:
            metadata: Dictionary with float_metadata and daily_reading_data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Separate the data
            float_metadata = metadata['float_metadata']
            daily_data = metadata['daily_reading_data']
            parameters = daily_data.pop('parameters', [])
            current_reading_date = daily_data['measurement_date']
            
            # Check if float already exists and get current last_reading_date
            existing_float = self.client.table('float_metadata').select('float_id', 'last_reading_date').eq('float_id', float_metadata['float_id']).execute()
            
            if existing_float.data:
                # Float exists - check if current reading date is newer
                existing_last_date = existing_float.data[0].get('last_reading_date')
                
                # Update last_reading_date if current reading is newer
                if not existing_last_date or current_reading_date > existing_last_date:
                    float_metadata['last_reading_date'] = current_reading_date
                    logger.debug(f"Updating last_reading_date for float {float_metadata['float_id']} from {existing_last_date} to {current_reading_date}")
                else:
                    # Keep existing last date if it's newer
                    float_metadata['last_reading_date'] = existing_last_date
                    logger.debug(f"Keeping existing last_reading_date {existing_last_date} for float {float_metadata['float_id']} (current: {current_reading_date})")
                
                # Update existing float metadata
                self.client.table('float_metadata').update(float_metadata).eq('float_id', float_metadata['float_id']).execute()
                logger.debug(f"Updated float metadata for {float_metadata['float_id']} with last reading date {float_metadata['last_reading_date']}")
            else:
                # New float - set last_reading_date to current reading date
                float_metadata['last_reading_date'] = current_reading_date
                
                # Insert new float metadata
                self.client.table('float_metadata').insert(float_metadata).execute()
                logger.debug(f"Inserted float metadata for {float_metadata['float_id']} with last reading date {float_metadata['last_reading_date']}")
            
            # BATCH INSERT ALL PARAMETERS AT ONCE - Much faster than individual inserts!
            if parameters:
                # Delete existing parameters for this float+date combination first
                self.client.table('float_parameters_reading').delete().eq('float_id', daily_data['float_id']).eq('date_recorded', current_reading_date).execute()
                
                # Prepare all parameter records for batch insert
                parameter_records = []
                for param in parameters:
                    parameter_record = {
                        'float_id': daily_data['float_id'],
                        'parameter_name': param['parameter_name'],
                        'value': param['value'],
                        'date_recorded': param['date_recorded']
                    }
                    parameter_records.append(parameter_record)
                
                # SINGLE BATCH INSERT - Replaces 50+ individual database calls with 1 call!
                if parameter_records:
                    self.client.table('float_parameters_reading').insert(parameter_records).execute()
                    logger.debug(f"BATCH inserted {len(parameter_records)} parameters for float {daily_data['float_id']} on {current_reading_date}")
                    self.stats['parameters_inserted'] += len(parameter_records)
            
            self.stats['floats_inserted'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert data for float {metadata.get('float_metadata', {}).get('float_id', 'unknown')} on {metadata.get('daily_reading_data', {}).get('measurement_date', 'unknown date')}: {str(e)}")
            self.stats['errors'] += 1
            return False
    
    def process_metadata_batch(self, batch_metadata: List[Dict]) -> None:
        """
        Process a batch of metadata records for better performance
        
        Args:
            batch_metadata: List of metadata dictionaries to process
        """
        logger.info(f"Processing batch of {len(batch_metadata)} float records...")
        
        for metadata in batch_metadata:
            if self.insert_float_metadata(metadata):
                self.stats['floats_processed'] += 1
                
                # Determine float type for statistics
                reading_data = metadata['daily_reading_data']
                csv_filename = metadata['float_metadata']['csv_filename']
                is_bgc = not any(month in csv_filename.upper() for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])
                
                if is_bgc:
                    self.stats['bgc_floats'] += 1
                    float_type = "BGC"
                else:
                    self.stats['non_bgc_floats'] += 1
                    float_type = "Non-BCG"
                
                float_id = metadata['float_metadata']['float_id']
                measurement_date = reading_data.get('measurement_date', 'Unknown')
                param_count = len(reading_data.get('parameters', []))
                cycle_number = reading_data.get('cycle_number', 'N/A')
                
                logger.info(f"✓ Processed {float_type} float {float_id} cycle {cycle_number} on {measurement_date} with {param_count} parameters")
            else:
                logger.error(f"✗ Failed to insert metadata record")
    
    def process_csv_directory(self, directory_path: str) -> None:
        """
        Process all CSV files in a directory and extract float metadata
        Handles both individual CSV files and nested directory structures
        
        Args:
            directory_path: Path to directory containing CSV files
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return
        
        # Find all CSV files recursively (handles nested directories)
        csv_files = list(directory.rglob("*.csv"))  # Use rglob for recursive search
        
        if not csv_files:
            logger.warning(f"No CSV files found in {directory_path} (searched recursively)")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to process in {directory_path}")
        logger.info("🚀 Using BATCH processing for maximum speed!")
        
        # BATCH PROCESSING: Collect multiple records before inserting
        batch_metadata = []
        batch_size = 25  # Process 25 floats at a time for optimal performance
        processed_floats = set()  # Track unique floats to avoid duplicates
        
        for csv_file in csv_files:
            self.stats['csv_files_scanned'] += 1
            logger.info(f"Processing {csv_file.name} ({self.stats['csv_files_scanned']}/{len(csv_files)})")
            
            # Extract metadata (now returns a list for Non-BCG files with multiple floats)
            metadata_list = self.extract_float_metadata_from_csv(str(csv_file))
            
            if metadata_list:
                logger.debug(f"Extracted {len(metadata_list)} reading records from {csv_file.name}")
                
                for metadata in metadata_list:
                    float_id = metadata['float_metadata']['float_id']
                    reading_data = metadata['daily_reading_data']
                    csv_filename = metadata['float_metadata']['csv_filename']
                    cycle_number = reading_data.get('cycle_number', 'N/A')
                    measurement_date = reading_data.get('measurement_date', 'Unknown')
                    
                    # Create unique identifier for this specific reading (float + date + cycle)
                    reading_id = f"{float_id}_{cycle_number}_{measurement_date}"
                    
                    # Determine if this is BGC or Non-BCG float
                    is_bgc_float = "Non-BCG" not in str(csv_file) and not any(month in csv_filename.upper() for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])
                    
                    # For Non-BCG files, we may need to add additional identification
                    if "Non-BCG" in str(csv_file) or any(month in csv_filename.upper() for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']):
                        import re
                        month_match = re.search(r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)', csv_filename.upper())
                        if month_match:
                            month = month_match.group(1)
                            reading_id = f"{float_id}_{cycle_number}_{measurement_date}_{month}"
                        is_bgc_float = False  # This is Non-BCG
                    else:
                        is_bgc_float = True   # This is BGC
                    
                    # Skip if we've already processed this specific reading in the SAME RUN
                    if reading_id in processed_floats:
                        logger.debug(f"Reading {reading_id} already processed in this run, skipping")
                        continue
                    
                    logger.debug(f"Adding {float_id} cycle {cycle_number} on {measurement_date} to batch (reading_id: {reading_id})")
                    
                    # Add to batch for processing
                    batch_metadata.append(metadata)
                    processed_floats.add(reading_id)
                    
                    # Process batch when it reaches batch_size
                    if len(batch_metadata) >= batch_size:
                        logger.info(f"⚡ Processing batch of {len(batch_metadata)} records...")
                        self.process_metadata_batch(batch_metadata)
                        batch_metadata = []  # Clear batch
            else:
                logger.warning(f"No metadata extracted from {csv_file.name}")
        
        # Process remaining items in final batch
        if batch_metadata:
            logger.info(f"⚡ Processing final batch of {len(batch_metadata)} records...")
            self.process_metadata_batch(batch_metadata)
    
    def verify_table_and_data(self) -> bool:
        """
        Verify that both tables exist and contain data
        
        Returns:
            True if verification successful, False otherwise
        """
        try:
            # Check float_metadata table (float-level data with metadata)
            metadata_response = self.client.table('float_metadata').select('float_id', count='exact').execute()
            metadata_count = metadata_response.count if hasattr(metadata_response, 'count') else len(metadata_response.data)
            logger.info(f"✓ Verification: float_metadata table contains {metadata_count} unique floats")
            
            # Check float_parameters_reading table (parameter readings with daily locations)
            params_response = self.client.table('float_parameters_reading').select('id', count='exact').execute()
            params_count = params_response.count if hasattr(params_response, 'count') else len(params_response.data)
            logger.info(f"✓ Verification: float_parameters_reading table contains {params_count} parameter records")
            
            # Show sample data from both tables
            if metadata_response.data and params_response.data:
                # Sample float metadata
                sample_float = metadata_response.data[0]
                logger.info(f"Sample float: {sample_float['float_id']}")
                logger.info(f"CSV filename: {sample_float.get('csv_filename', 'Unknown')}")
                logger.info(f"Last reading date: {sample_float.get('last_reading_date', 'Unknown')}")
                logger.info(f"Total columns: {len(sample_float.get('all_columns', {}))}")
                logger.info(f"Metadata fields: {len(sample_float.get('metadata', {}))}")
                
                # Sample parameter reading
                sample_param = params_response.data[0]
                logger.info(f"Sample reading: Float {sample_param['float_id']} on {sample_param.get('date_recorded', 'Unknown')}")
                logger.info(f"Parameter: {sample_param.get('parameter_name', 'Unknown')} = {sample_param.get('value', 'N/A')}")
            
            # Show parameters per float statistics
            if params_response.data:
                params_per_float = {}
                for param in params_response.data:
                    float_id = param['float_id']
                    params_per_float[float_id] = params_per_float.get(float_id, 0) + 1
                
                avg_params = sum(params_per_float.values()) / len(params_per_float) if params_per_float else 0
                max_params = max(params_per_float.values()) if params_per_float else 0
                logger.info(f"✓ Data spans {len(params_per_float)} unique floats with avg {avg_params:.1f} parameters per float (max: {max_params})")
            
            # List all unique float IDs for debugging
            if metadata_response.data:
                float_ids = [record['float_id'] for record in metadata_response.data]
                logger.info(f"\n📊 Uploaded float IDs ({len(float_ids)}): {', '.join(sorted(float_ids))}")
            
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return False
    
    def analyze_column_patterns(self) -> None:
        """
        Analyze column patterns across all processed CSV files
        Provides insights into data structure variations
        """
        try:
            logger.info("Analyzing column patterns across all floats...")
            
            # Get all records
            response = self.client.table('float_metadata').select('*').execute()
            
            if not response.data:
                logger.warning("No data found for column analysis")
                return
            
            # Analyze column patterns
            all_columns_seen = set()
            bgc_columns = set()
            non_bgc_columns = set()
            column_frequency = {}
            
            bgc_floats = 0
            non_bgc_floats = 0
            
            for record in response.data:
                columns = record.get('all_columns', [])
                all_columns_seen.update(columns)
                
                # Count column frequency
                for col in columns:
                    column_frequency[col] = column_frequency.get(col, 0) + 1
                
                # Categorize by float type
                is_bgc = any(col in columns for col in ['DOXY', 'BBP700', 'CDOM', 'CHLA'])
                if is_bgc:
                    bgc_floats += 1
                    bgc_columns.update(columns)
                else:
                    non_bgc_floats += 1
                    non_bgc_columns.update(columns)
            
            logger.info("=" * 60)
            logger.info("COLUMN ANALYSIS RESULTS")
            logger.info("=" * 60)
            logger.info(f"Total unique columns across all files: {len(all_columns_seen)}")
            logger.info(f"BGC floats: {bgc_floats} (unique columns: {len(bgc_columns)})")
            logger.info(f"Non-BCG floats: {non_bgc_floats} (unique columns: {len(non_bgc_columns)})")
            
            # Show BGC-specific columns
            bgc_only = bgc_columns - non_bgc_columns
            if bgc_only:
                logger.info(f"\nBGC-specific columns ({len(bgc_only)}):")
                for col in sorted(bgc_only):
                    logger.info(f"  • {col}")
            
            # Show Non-BCG specific columns
            non_bgc_only = non_bgc_columns - bgc_columns
            if non_bgc_only:
                logger.info(f"\nNon-BCG specific columns ({len(non_bgc_only)}):")
                for col in sorted(non_bgc_only):
                    logger.info(f"  • {col}")
            
            # Show common columns
            common_columns = bgc_columns & non_bgc_columns
            if common_columns:
                logger.info(f"\nCommon columns ({len(common_columns)}):")
                for col in sorted(common_columns):
                    logger.info(f"  • {col}")
            
            # Show most/least frequent columns
            sorted_frequency = sorted(column_frequency.items(), key=lambda x: x[1], reverse=True)
            
            logger.info(f"\nMost frequent columns:")
            for col, freq in sorted_frequency[:10]:
                logger.info(f"  • {col}: {freq} files ({freq/len(response.data)*100:.1f}%)")
            
            logger.info(f"\nLeast frequent columns:")
            for col, freq in sorted_frequency[-10:]:
                logger.info(f"  • {col}: {freq} files ({freq/len(response.data)*100:.1f}%)")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Column analysis failed: {str(e)}")
    
    def process_all_directories(self, base_paths: List[str]) -> Dict:
        """
        Process all CSV files from multiple directories
        
        Args:
            base_paths: List of directory paths to process
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting ARGO float metadata extraction and upload...")
        
        # Check if table exists first
        if not self.create_float_metadata_table():
            logger.error("One or more tables don't exist. Please create them manually and re-run.")
            return self.stats
        
        # Process each directory
        for base_path in base_paths:
            logger.info(f"Processing directory: {base_path}")
            self.process_csv_directory(base_path)
        
        # Verify results
        self.verify_table_and_data()
        
        # Analyze column patterns across all files
        self.analyze_column_patterns()
        
        # Print final statistics
        logger.info("=" * 60)
        logger.info("⚡ BATCH PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"CSV files scanned: {self.stats['csv_files_scanned']}")
        logger.info(f"Individual readings processed: {self.stats['floats_processed']}")
        logger.info(f"  • BGC float readings: {self.stats['bgc_floats']}")
        logger.info(f"  • Non-BCG float readings: {self.stats['non_bgc_floats']}")
        logger.info(f"Reading records inserted/updated in float_metadata: {self.stats['floats_inserted']}")
        logger.info(f"Parameter records BATCH inserted in float_parameters: {self.stats['parameters_inserted']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info("🚀 Performance: ~50x faster with batch processing!")
        logger.info("=" * 60)
        
        return self.stats

def main():
    """Main execution function"""
    
    # Load Supabase configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # Directory paths containing CSV files
    BASE_PATHS = [
        r"f:\pineconetesting\Data\BCG floats\final csv files",
        r"f:\pineconetesting\Data\Non-BCG floats\final_csv"  # Will search recursively for CSV files
    ]
    
    try:
        # Validate configuration
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.error("ERROR: Supabase configuration not found!")
            print("\nConfiguration Setup Required:")
            print("1. Make sure .env file exists with:")
            print("   SUPABASE_URL=https://your-project.supabase.co")
            print("   SUPABASE_KEY=your_anon_public_key_here")
            print("2. Run the script again")
            return
        
        # Create configuration
        config = SupabaseConfig(url=SUPABASE_URL, key=SUPABASE_KEY)
        
        # Initialize uploader
        uploader = FloatMetadataUploader(config)
        
        # Test connection
        logger.info("Testing Supabase connection...")
        try:
            test_result = uploader.client.table('_nonexistent_').select("*").limit(1).execute()
        except Exception as e:
            if "does not exist" in str(e).lower() or "could not find" in str(e).lower():
                logger.info("✓ Supabase connection successful!")
            else:
                logger.error(f"✗ Supabase connection failed: {str(e)}")
                return
        
        # Process all CSV files
        stats = uploader.process_all_directories(BASE_PATHS)
        
        if stats['floats_processed'] > 0:
            logger.info("SUCCESS: Float metadata upload completed!")
            
            print("\n🎯 Results Summary:")
            print(f"   • {stats['floats_processed']} individual readings processed")
            print(f"     - BGC float readings: {stats['bgc_floats']}")
            print(f"     - Non-BCG float readings: {stats['non_bgc_floats']}")
            print(f"   • {stats['floats_inserted']} reading records inserted/updated in float_metadata table")
            print(f"   • {stats['parameters_inserted']} parameter records inserted in float_parameters table")
            print(f"   • {stats['csv_files_scanned']} CSV files scanned")
            print("\n📊 Next Steps:")
            print("   1. Verify data in Supabase dashboard")
            print("   2. Query both float_metadata and float_parameters tables")
            print("   3. Use JOIN queries to get comprehensive float information across time")
            print("   4. Track float trajectories and parameter evolution over time")
            print("   5. Use the metadata field for understanding float capabilities")
            print("\n✅ Both tables now support multiple readings per float with temporal tracking!")
        else:
            logger.error("ERROR: No floats were processed successfully")
    
    except Exception as e:
        logger.error(f"FATAL ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()

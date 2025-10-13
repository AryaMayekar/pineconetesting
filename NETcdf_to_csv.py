"""
Integrated NetCDF to Excel Processing Pipeline
============================================
This script combines the functionality of:
1. gettingdata.py - Downloads NetCDF files from FTP
2. converttoexcel.py - Converts NetCDF files to Excel format
3. tp.py - Creates final aggregated Excel summaries

Workflow: Download → Convert → Aggregate
"""

import numpy as np
import netCDF4
import xarray as xr
import cartopy.crs as ccrs
import cartopy
from ftplib import FTP
import os
import itertools
import pandas as pd
import gsw  # Gibbs SeaWater library for accurate depth
import shutil
from datetime import datetime
from shapely.geometry import Point, Polygon

# ============================================================================
# CONFIGURATION - USER INPUT SECTION
# ============================================================================

# User Configuration
YEAR = int(input("Enter the year (e.g., 2025): "))

print("\nAvailable months:")
print("1-January, 2-February, 3-March, 4-April, 5-May, 6-June")
print("7-July, 8-August, 9-September, 10-October, 11-November, 12-December")
print("\nEnter months to download (examples):")
print("- Single month: 8")
print("- Multiple months: 8,9,10")
print("- Range: 1-12")

month_input = input("Enter month(s): ")

# Parse month input
MONTHS_TO_PROCESS = []
if '-' in month_input:
    # Range input (e.g., "1-12")
    start, end = map(int, month_input.split('-'))
    MONTHS_TO_PROCESS = list(range(start, end + 1))
elif ',' in month_input:
    # Multiple months (e.g., "8,9,10")
    MONTHS_TO_PROCESS = [int(m.strip()) for m in month_input.split(',')]
else:
    # Single month (e.g., "8")
    MONTHS_TO_PROCESS = [int(month_input)]

print(f"\nProcessing year {YEAR}, months: {MONTHS_TO_PROCESS}")
print("=" * 60)

# ============================================================================
# UTILITY FUNCTIONS (SHARED ACROSS ALL MODULES)
# ============================================================================

def get_month_name(month_num):
    """Convert month number to month name"""
    months = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August", 
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    return months.get(month_num, "Unknown")

def get_month_abbrev(month_num):
    """Convert month number to 3-letter abbreviation"""
    abbrevs = {
        1: "JAN", 2: "FEB", 3: "MAR", 4: "APR",
        5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG", 
        9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
    }
    return abbrevs.get(month_num, "UNK")

def parse_filename_date(filename):
    """Extract year, month, day from filename like '20250801_prof.nc'"""
    try:
        # Extract the date part (first 8 characters)
        date_str = filename[:8]
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        return year, month, day
    except (ValueError, IndexError):
        return None, None, None

def convert_juld_to_datetime(juld_values):
    """Convert JULD (Julian day since 1950-01-01) to datetime"""
    try:
        # Check if values are already in datetime format
        if len(juld_values) > 0:
            # Check first non-null value to see if it's already datetime-like
            first_val = None
            for val in juld_values:
                if pd.notna(val) and val != 999999.0:
                    first_val = val
                    break
            
            if first_val is not None:
                # Try to detect if it's already a datetime string
                if isinstance(first_val, str):
                    try:
                        # If it can be parsed as datetime, it's already converted
                        pd.to_datetime(first_val)
                        print("  JULD appears to already be in datetime format, using as-is")
                        return pd.to_datetime(juld_values, errors='coerce')
                    except:
                        pass  # Not a datetime string, continue with Julian conversion
                
                # Check if it's already a datetime object
                if hasattr(first_val, 'year'):
                    print("  JULD is already datetime object, using as-is")
                    return pd.Series(juld_values)
        
        # Argo reference date: January 1, 1950 00:00:00 UTC
        reference_date = pd.Timestamp('1950-01-01 00:00:00', tz='UTC')
        
        # Convert JULD (days since reference) to datetime
        datetime_values = []
        for juld in juld_values:
            if pd.isna(juld) or juld == 999999.0:  # Missing data value in Argo
                datetime_values.append(pd.NaT)
            else:
                try:
                    # Convert to float first
                    juld_float = float(juld)
                    # Add days to reference date
                    datetime_val = reference_date + pd.Timedelta(days=juld_float)
                    datetime_values.append(datetime_val)
                except (ValueError, TypeError):
                    # If conversion fails, it might already be datetime
                    datetime_values.append(pd.NaT)
        
        return pd.Series(datetime_values)
    except Exception as e:
        print(f"Error converting JULD: {e}")
        # If all else fails, try to parse as datetime
        try:
            return pd.to_datetime(juld_values, errors='coerce')
        except:
            return pd.Series(juld_values)  # Return original if conversion fails

# ============================================================================
# MODULE 1: DATA DOWNLOADING (from gettingdata.py) (NETcdf files)
# ============================================================================

def create_folder_structure(base_dir, year, month):
    """Create the organized folder structure"""
    month_name = get_month_name(month)
    folder_path = os.path.join("Data", base_dir, f"argo_{year}", month_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def organize_existing_netcdf_files():
    """Move existing files from NETcdf_Data to organized structure"""
    old_dir = os.path.join("Data", "NETcdf_Data")
    if os.path.exists(old_dir):
        print("Organizing existing NetCDF files...")
        for filename in os.listdir(old_dir):
            if filename.endswith(".nc") and len(filename) >= 8:
                # Extract year and month from filename (YYYYMMDD_prof.nc)
                try:
                    year = int(filename[:4])
                    month = int(filename[4:6])
                    
                    # Create new folder structure
                    new_folder = create_folder_structure("NETcdf_Data", year, month)
                    
                    # Move file
                    old_path = os.path.join(old_dir, filename)
                    new_path = os.path.join(new_folder, filename)
                    
                    if old_path != new_path:  # Only move if different location
                        os.rename(old_path, new_path)
                        print(f"Moved {filename} to {new_folder}")
                except ValueError:
                    print(f"Could not parse date from filename: {filename}")

def download_netcdf_data(year, months_to_download):
    """Download NetCDF data from FTP server"""
    print("\n" + "="*50)
    print("STEP 1: DOWNLOADING NETCDF DATA")
    print("="*50)
    
    # Organize existing files first
    organize_existing_netcdf_files()

    for month in months_to_download:
        print(f"\n=== Downloading {get_month_name(month)} {year} ===")
        
        # FTP connection
        try:
            ftp = FTP("ftp.ifremer.fr", timeout=120)
            ftp.login()

            # Create organized folder structure
            local_folder = create_folder_structure("NETcdf_Data", year, month)
            print(f"Created/Using folder: {local_folder}")

            # Remote directory based on year and month
            remote_dir = f"/ifremer/argo/geo/indian_ocean/{year}/{month:02d}"
            print(f"Downloading from: {remote_dir}")

            # Change to the remote directory
            ftp.cwd(remote_dir)

            # List files in the remote folder
            files = ftp.nlst()
            print(f"Found {len(files)} files to process")

            # Download only missing files
            downloaded_count = 0
            for filename in files:
                local_path = os.path.join(local_folder, filename)

                if os.path.exists(local_path):
                    print(f"Skipping {filename} (already downloaded)")
                    continue

                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {filename}", f.write)
                    print(f"Downloaded {filename}")
                    downloaded_count += 1
            
            print(f"Downloaded {downloaded_count} new files for {get_month_name(month)} {year}")
            
        except Exception as e:
            print(f"Error downloading {get_month_name(month)} {year}: {e}")
        
        finally:
            try:
                ftp.quit()
            except:
                pass

    print("\n=== NetCDF Download Phase Completed ===")

# ============================================================================
# MODULE 2: NETCDF TO CSV CONVERSION (from converttoCSV.py)
# ============================================================================

def organize_existing_excel_files():
    """Move existing Excel files from rough_csv_Data root to appropriate year/month folders"""
    base_excel_folder = os.path.join("Data", "rough_csv_Data")
    
    if not os.path.exists(base_excel_folder):
        return
    
    print("=== Organizing existing Excel files ===")
    
    # Get all Excel files in the root directory
    for file in os.listdir(base_excel_folder):
        if file.endswith(".xlsx"):
            file_path = os.path.join(base_excel_folder, file)
            
            # Skip if it's already in a subdirectory
            if os.path.isfile(file_path):
                year, month, day = parse_filename_date(file)
                
                if year and month:
                    # Create target directory structure
                    month_name = get_month_name(month)
                    target_dir = os.path.join(base_excel_folder, str(year), month_name)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # Move file to appropriate folder
                    target_path = os.path.join(target_dir, file)
                    
                    if not os.path.exists(target_path):
                        shutil.move(file_path, target_path)
                        print(f"Moved: {file} -> {year}/{month_name}/")
                    else:
                        print(f"File already exists in target location: {target_path}")
                        os.remove(file_path)  # Remove duplicate

def decode_bytes(arr):
    """Function to decode byte strings"""
    if arr.dtype.kind in {'S', 'O'}:  # bytes or object
        return np.array([x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in arr])
    return arr

def calculate_surface_temp(temp_data, pres_data):
    """Calculate surface temperature (average of measurements between 0-10 dbar)"""
    try:
        surface_mask = (pres_data >= 0) & (pres_data <= 10) & np.isfinite(temp_data) & np.isfinite(pres_data)
        if np.any(surface_mask):
            surface_temps = temp_data[surface_mask]
            return {
                'min': float(np.min(surface_temps)),
                'max': float(np.max(surface_temps)),
                'avg': float(np.mean(surface_temps))
            }
        else:
            return {'min': None, 'max': None, 'avg': None}
    except:
        return {'min': None, 'max': None, 'avg': None}

def calculate_mixed_layer_depth(temp_data, pres_data, threshold=0.2):
    """Estimate mixed layer depth using temperature criterion"""
    try:
        valid_mask = np.isfinite(temp_data) & np.isfinite(pres_data)
        if np.sum(valid_mask) < 3:
            return None
        
        temp_clean = temp_data[valid_mask]
        pres_clean = pres_data[valid_mask]
        
        sort_idx = np.argsort(pres_clean)
        temp_sorted = temp_clean[sort_idx]
        pres_sorted = pres_clean[sort_idx]
        
        surface_mask = pres_sorted <= 10
        if np.sum(surface_mask) == 0:
            surface_temp = temp_sorted[0]
        else:
            surface_temp = np.mean(temp_sorted[surface_mask])
        
        for i, (temp, pres) in enumerate(zip(temp_sorted, pres_sorted)):
            if abs(temp - surface_temp) >= threshold:
                return float(pres)
        
        return None
    except:
        return None

def analyze_netcdf_variables(nc_file_path):
    """Analyze variables in NetCDF file for debugging"""
    try:
        ds = xr.open_dataset(nc_file_path)
        print(f"\nAnalyzing: {nc_file_path}")
        print(f"Dimensions: {dict(ds.sizes)}")
        print(f"Total variables: {len(ds.variables)}")
        
        for var_name, var_data in ds.variables.items():
            print(f"  {var_name}: dims={var_data.dims}, shape={var_data.shape}, dtype={var_data.dtype}")
        
        ds.close()
    except Exception as e:
        print(f"Error analyzing {nc_file_path}: {e}")

def extract_all_variables_from_netcdf(nc_file_path):
    """Extract ALL variables from NetCDF file and return as flattened DataFrame"""
    try:
        ds = xr.open_dataset(nc_file_path)
        
        n_prof = ds.sizes['N_PROF']
        n_levels = ds.sizes['N_LEVELS']
        n_param = ds.sizes.get('N_PARAM', 0)
        n_calib = ds.sizes.get('N_CALIB', 0)
        n_history = ds.sizes.get('N_HISTORY', 0)
        
        # Extract variables by type
        profile_vars = {}
        measurement_vars = {}
        scalar_vars = {}
        
        print(f"Processing {len(ds.variables)} variables from {nc_file_path}")
        print(f"Dimensions: N_PROF={n_prof}, N_LEVELS={n_levels}, N_PARAM={n_param}, N_CALIB={n_calib}, N_HISTORY={n_history}")
        
        for var_name, var_data in ds.variables.items():
            try:
                values = var_data.values
                dims = var_data.dims
                
                # Handle different dimension patterns
                if dims == ('N_PROF',):
                    # Profile-level variable
                    if var_data.dtype.kind in {'S', 'O'}:
                        profile_vars[var_name] = decode_bytes(values)
                    else:
                        profile_vars[var_name] = values
                    print(f"  Added profile var: {var_name}")
                        
                elif dims == ('N_PROF', 'N_LEVELS'):
                    # Measurement-level variable
                    measurement_vars[var_name] = values
                    print(f"  Added measurement var: {var_name}")
                    
                elif dims == ():
                    # Scalar variable - same for all profiles
                    if var_data.dtype.kind in {'S', 'O'}:
                        scalar_vars[var_name] = decode_bytes(np.array([values]))[0]
                    else:
                        scalar_vars[var_name] = values
                    print(f"  Added scalar var: {var_name}")
                    
                elif dims == ('N_PROF', 'N_PARAM'):
                    # Parameter-related variables - take first parameter for each profile
                    if n_param > 0:
                        if var_data.dtype.kind in {'S', 'O'}:
                            # Handle string arrays
                            decoded_values = []
                            for i in range(n_prof):
                                if len(values.shape) == 2:
                                    # Take first parameter
                                    param_value = values[i, 0] if values.shape[1] > 0 else ''
                                    if isinstance(param_value, bytes):
                                        decoded_values.append(param_value.decode('utf-8', errors='ignore').strip())
                                    else:
                                        decoded_values.append(str(param_value).strip())
                                else:
                                    decoded_values.append('')
                            profile_vars[var_name] = np.array(decoded_values)
                        else:
                            # Take first parameter for each profile
                            profile_vars[var_name] = values[:, 0] if values.shape[1] > 0 else np.full(n_prof, np.nan)
                        print(f"  Added N_PROF,N_PARAM var: {var_name}")
                    
                elif dims == ('N_PROF', 'N_CALIB', 'N_PARAM'):
                    # Calibration-related variables - take first calibration and first parameter
                    if n_calib > 0 and n_param > 0:
                        if var_data.dtype.kind in {'S', 'O'}:
                            # Handle string arrays
                            decoded_values = []
                            for i in range(n_prof):
                                if len(values.shape) == 3:
                                    # Take first calibration, first parameter
                                    calib_value = values[i, 0, 0] if values.shape[1] > 0 and values.shape[2] > 0 else ''
                                    if isinstance(calib_value, bytes):
                                        decoded_values.append(calib_value.decode('utf-8', errors='ignore').strip())
                                    else:
                                        decoded_values.append(str(calib_value).strip())
                                else:
                                    decoded_values.append('')
                            profile_vars[var_name] = np.array(decoded_values)
                        else:
                            # Take first calibration, first parameter for each profile
                            if values.shape[1] > 0 and values.shape[2] > 0:
                                profile_vars[var_name] = values[:, 0, 0]
                            else:
                                profile_vars[var_name] = np.full(n_prof, np.nan)
                        print(f"  Added N_PROF,N_CALIB,N_PARAM var: {var_name}")
                    
                elif dims == ('N_HISTORY', 'N_PROF'):
                    # History variables - if N_HISTORY > 0, take last history entry for each profile
                    if n_history > 0:
                        if var_data.dtype.kind in {'S', 'O'}:
                            # Handle string arrays
                            decoded_values = []
                            for i in range(n_prof):
                                if len(values.shape) == 2:
                                    # Take last history entry
                                    hist_value = values[-1, i] if values.shape[0] > 0 else ''
                                    if isinstance(hist_value, bytes):
                                        decoded_values.append(hist_value.decode('utf-8', errors='ignore').strip())
                                    else:
                                        decoded_values.append(str(hist_value).strip())
                                else:
                                    decoded_values.append('')
                            profile_vars[var_name] = np.array(decoded_values)
                        else:
                            # Take last history entry for each profile
                            if values.shape[0] > 0:
                                profile_vars[var_name] = values[-1, :]
                            else:
                                profile_vars[var_name] = np.full(n_prof, np.nan)
                        print(f"  Added N_HISTORY,N_PROF var: {var_name}")
                    else:
                        # No history data available
                        if var_data.dtype.kind in {'S', 'O'}:
                            profile_vars[var_name] = np.full(n_prof, '', dtype=object)
                        else:
                            profile_vars[var_name] = np.full(n_prof, np.nan)
                        print(f"  Added empty N_HISTORY,N_PROF var: {var_name}")
                    
                elif dims == ('N_PROF', 'STRING_length') or dims == ('N_PROF', 'STRING64', 'STRING_length'):
                    # String variables with character arrays
                    if var_data.dtype.kind in {'S', 'O'}:
                        # Decode and join character arrays
                        decoded_values = []
                        for i in range(n_prof):
                            if len(values.shape) == 2:
                                # 2D character array
                                char_array = values[i, :]
                            else:
                                # 3D character array
                                char_array = values[i, 0, :]
                            
                            if isinstance(char_array, bytes):
                                decoded_values.append(char_array.decode('utf-8', errors='ignore').strip())
                            elif hasattr(char_array, 'tobytes'):
                                decoded_values.append(char_array.tobytes().decode('utf-8', errors='ignore').strip())
                            else:
                                decoded_values.append(str(char_array).strip())
                        
                        profile_vars[var_name] = np.array(decoded_values)
                    else:
                        profile_vars[var_name] = values
                    print(f"  Added string var: {var_name}")
                        
                elif 'N_PROF' in dims:
                    # Other variables containing N_PROF - try to handle generically
                    print(f"  Attempting to handle variable {var_name} with dims {dims}")
                    try:
                        # Find the position of N_PROF in dimensions
                        prof_idx = dims.index('N_PROF')
                        
                        if prof_idx == 0:  # N_PROF is first dimension
                            if len(dims) == 1:
                                # Already handled above
                                profile_vars[var_name] = values
                            elif len(dims) == 2:
                                # Take first element of second dimension
                                profile_vars[var_name] = values[:, 0] if values.shape[1] > 0 else np.full(n_prof, np.nan)
                            else:
                                # Flatten other dimensions, take first elements
                                reshaped = values.reshape(n_prof, -1)
                                profile_vars[var_name] = reshaped[:, 0] if reshaped.shape[1] > 0 else np.full(n_prof, np.nan)
                        else:
                            # N_PROF is not first dimension - transpose to make it first
                            axes = list(range(len(dims)))
                            axes[0], axes[prof_idx] = axes[prof_idx], axes[0]
                            transposed = np.transpose(values, axes)
                            profile_vars[var_name] = transposed[:, 0] if transposed.shape[1] > 0 else np.full(n_prof, np.nan)
                        
                        print(f"    Successfully handled {var_name}")
                    except Exception as e:
                        print(f"    Could not handle complex variable {var_name}: {e}")
                            
                else:
                    print(f"  Skipping variable {var_name} with unsupported dims: {dims}")
                    
            except Exception as e:
                print(f"Warning: Could not extract variable {var_name}: {e}")
        
        # Convert JULD to datetime if present
        if 'JULD' in profile_vars:
            try:
                profile_vars['JULD'] = convert_juld_to_datetime(profile_vars['JULD'])
            except Exception as e:
                print(f"Warning: Could not convert JULD: {e}")
        
        # Convert JULD_LOCATION to datetime if present
        if 'JULD_LOCATION' in profile_vars:
            try:
                profile_vars['JULD_LOCATION'] = convert_juld_to_datetime(profile_vars['JULD_LOCATION'])
            except Exception as e:
                print(f"Warning: Could not convert JULD_LOCATION: {e}")
        
        # Create flattened DataFrame
        rows = []
        
        # Combine all column names
        profile_columns = list(sorted(profile_vars.keys()))
        measurement_columns = list(sorted(measurement_vars.keys()))
        scalar_columns = list(sorted(scalar_vars.keys()))
        
        columns = profile_columns + measurement_columns + scalar_columns
        
        print(f"Creating DataFrame with {len(columns)} columns:")
        print(f"  Profile vars: {len(profile_columns)}")
        print(f"  Measurement vars: {len(measurement_columns)}")
        print(f"  Scalar vars: {len(scalar_columns)}")
        print(f"  Total: {len(profile_columns) + len(measurement_columns) + len(scalar_columns)} columns")
        
        for i in range(n_prof):
            for j in range(n_levels):
                row = []
                
                # Add profile-level data
                for var_name in profile_columns:
                    row.append(profile_vars[var_name][i])
                    
                # Add measurement-level data
                for var_name in measurement_columns:
                    row.append(measurement_vars[var_name][i, j])
                    
                # Add scalar data (same for all rows)
                for var_name in scalar_columns:
                    row.append(scalar_vars[var_name])
                    
                rows.append(row)
        
        df = pd.DataFrame(rows, columns=columns)
        ds.close()
        
        print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"Error extracting variables from {nc_file_path}: {e}")
        return None

def convert_netcdf_to_csv():
    """Convert NetCDF files to CSV format with ALL variables"""
    print("\n" + "="*50)
    print("STEP 2: CONVERTING NETCDF TO CSV")
    print("="*50)
    
    # Base folders
    base_nc_folder = os.path.join("Data", "NETcdf_Data", "argo_" + str(YEAR))
    base_excel_folder = os.path.join("Data", "rough_csv_Data")

    # Make sure the output folder exists
    os.makedirs(base_excel_folder, exist_ok=True)

    # Organize existing Excel files first
    organize_existing_excel_files()

    # Process each month folder
    if not os.path.exists(base_nc_folder):
        print(f"NetCDF folder not found: {base_nc_folder}")
        return

    for month_folder in os.listdir(base_nc_folder):
        month_path = os.path.join(base_nc_folder, month_folder)
        if os.path.isdir(month_path):
            print(f"\n=== Converting {month_folder} ===")
            
            # Loop through all NetCDF files in the month folder
            for file in os.listdir(month_path):
                if file.endswith(".nc"):
                    # Parse filename to get year, month, day
                    year, month, day = parse_filename_date(file)
                    
                    if year and month:
                        # Create target Excel directory structure
                        month_name = get_month_name(month)
                        excel_dir = os.path.join(base_excel_folder, str(year), month_name)
                        os.makedirs(excel_dir, exist_ok=True)
                        
                        # Generate CSV filename and path (changed from Excel)
                        csv_filename = file.replace(".nc", ".csv")
                        csv_file = os.path.join(excel_dir, csv_filename)
                        
                        # Skip if CSV file already exists
                        if os.path.exists(csv_file):
                            print(f"Skipping {file} - CSV file already exists: {csv_file}")
                            continue
                        
                        print(f"Converting: {file} -> {year}/{month_name}/{csv_filename}")
                        
                        try:
                            nc_file = os.path.join(month_path, file)
                            
                            # Extract ALL variables using new method
                            df = extract_all_variables_from_netcdf(nc_file)
                            
                            if df is not None:
                                # Save as CSV instead of Excel
                                df.to_csv(csv_file, index=False)
                                print(f"Saved: {csv_file} ({len(df)} rows, {len(df.columns)} columns)")
                            else:
                                print(f"Failed to extract data from {file}")
                            
                        except Exception as e:
                            print(f"Error converting {file}: {e}")
                    else:
                        print(f"Warning: Could not parse date from filename: {file}")

    print("\n=== NetCDF to CSV Conversion Completed ===")

# ============================================================================
# MODULE 3: EXCEL AGGREGATION 
# ============================================================================

def organize_existing_final_excel_files():
    """Move existing CSV files from final_csv root to appropriate year/month folders"""
    base_final_folder = os.path.join("Data", "final_csv")
    
    if not os.path.exists(base_final_folder):
        return
    
    print("=== Organizing existing final Excel files ===")
    
    # Get all Excel files in the root directory
    for file in os.listdir(base_final_folder):
        if file.endswith(".xlsx"):
            file_path = os.path.join(base_final_folder, file)
            
            # Skip if it's already in a subdirectory
            if os.path.isfile(file_path):
                # Try to parse the filename to determine year/month
                # Look for patterns like final_JAN_days.xlsx or final_AUG_days.xlsx
                if "final_" in file and "_days" in file:
                    try:
                        # Extract month abbreviation
                        parts = file.split("_")
                        if len(parts) >= 2:
                            month_abbrev = parts[1]
                            # Convert abbreviation to month number
                            abbrev_to_num = {
                                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
                                "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
                                "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
                            }
                            
                            if month_abbrev in abbrev_to_num:
                                month = abbrev_to_num[month_abbrev]
                                year = YEAR  # Use the configured year
                                
                                # Create target directory structure
                                month_name = get_month_name(month)
                                target_dir = os.path.join(base_final_folder, str(year), month_name)
                                os.makedirs(target_dir, exist_ok=True)
                                
                                # Move file to appropriate folder
                                target_path = os.path.join(target_dir, file)
                                
                                if not os.path.exists(target_path):
                                    shutil.move(file_path, target_path)
                                    print(f"Moved: {file} -> {year}/{month_name}/")
                                else:
                                    print(f"File already exists in target location: {target_path}")
                                    os.remove(file_path)  # Remove duplicate
                    except Exception as e:
                        print(f"Could not organize file {file}: {e}")

def find_csv_files_in_structure(base_folder):
    """Find all CSV files in the year/month folder structure"""
    csv_files = []
    
    # Update base folder path to include Data directory
    full_base_folder = os.path.join("Data", base_folder)
    
    if not os.path.exists(full_base_folder):
        return csv_files
    
    # Walk through year folders
    for year_folder in os.listdir(full_base_folder):
        year_path = os.path.join(full_base_folder, year_folder)
        if os.path.isdir(year_path) and year_folder.isdigit():
            # Walk through month folders
            for month_folder in os.listdir(year_path):
                month_path = os.path.join(year_path, month_folder)
                if os.path.isdir(month_path):
                    # Find CSV files in month folder
                    for file in os.listdir(month_path):
                        if file.endswith(".csv"):
                            file_path = os.path.join(month_path, file)
                            csv_files.append({
                                'path': file_path,
                                'year': int(year_folder),
                                'month_name': month_folder,
                                'filename': file
                            })
    
    return csv_files

def create_final_csv_summaries():
    """Create final CSV summaries by aggregating data"""
    print("\n" + "="*50)
    print("STEP 3: CREATING FINAL CSV SUMMARIES")
    print("="*50)
    
    # Base folders
    rough_excel_folder = "rough_csv_Data"
    final_excel_folder = os.path.join("Data", "final_csv")

    # Make sure the output folder exists
    print(f"Creating base directory: {final_excel_folder}")
    try:
        os.makedirs(final_excel_folder, exist_ok=True)
        print(f"Base directory created/exists: {final_excel_folder}")
    except Exception as e:
        print(f"ERROR creating base directory {final_excel_folder}: {e}")
        return

    # Organize existing final Excel files first
    organize_existing_final_excel_files()

    # Find all CSV files in rough_csv_Data structure
    csv_files = find_csv_files_in_structure(rough_excel_folder)

    if not csv_files:
        print("No CSV files found in rough_csv_Data folder structure.")
        return

    print(f"Found {len(csv_files)} CSV files to process")

    # Group files by year and month
    files_by_month = {}
    for file_info in csv_files:
        year = file_info['year']
        month_name = file_info['month_name']
        key = (year, month_name)
        
        if key not in files_by_month:
            files_by_month[key] = []
        files_by_month[key].append(file_info)

    # Process each month
    for (year, month_name), month_files in files_by_month.items():
        print(f"\n=== Processing {month_name} {year} ===")
        
        # Convert month name to number for abbreviation
        month_num = None
        for num, name in {1: "January", 2: "February", 3: "March", 4: "April",
                          5: "May", 6: "June", 7: "July", 8: "August", 
                          9: "September", 10: "October", 11: "November", 12: "December"}.items():
            if name == month_name:
                month_num = num
                break
        
        if month_num is None:
            print(f"Could not determine month number for {month_name}")
            continue
        
        month_abbrev = get_month_abbrev(month_num)
        
        # Create target directory for final Excel
        final_dir = os.path.join(final_excel_folder, str(year), month_name)
        print(f"  Creating directory: {final_dir}")
        try:
            os.makedirs(final_dir, exist_ok=True)
            print(f"  Directory created/exists: {final_dir}")
        except Exception as e:
            print(f"  ERROR creating directory {final_dir}: {e}")
            continue
        
        # Generate output filename
        output_filename = f"final_{month_abbrev}_days.xlsx"
        output_file = os.path.join(final_dir, output_filename)
        
        # Skip if output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {month_name} {year} - Output file already exists: {output_file}")
            continue
        
        print(f"Processing {len(month_files)} files for {month_name} {year}")
        
        # Read all CSV files for this month and combine them
        all_dataframes = []
        for file_info in month_files:
            try:
                print(f"  Loading: {file_info['filename']}...")
                df_temp = pd.read_csv(file_info['path'], low_memory=False)
                all_dataframes.append(df_temp)
                print(f"    Loaded {len(df_temp)} rows")
            except Exception as e:
                print(f"  Error loading {file_info['filename']}: {e}")
        
        if not all_dataframes:
            print(f"No valid data found for {month_name} {year}")
            continue
        
        print(f"  Combining {len(all_dataframes)} files...")
        # Combine all dataframes into one
        df = pd.concat(all_dataframes, ignore_index=True)
        print(f"  Total records combined: {len(df)}")
        
        # Free memory
        del all_dataframes
        
        # Handle time data
        if 'JULD' in df.columns:
            # Check if JULD is already datetime or needs conversion
            if not pd.api.types.is_datetime64_any_dtype(df['JULD']):
                # Also check if it's already datetime strings
                if df['JULD'].dtype == 'object':
                    # Try to detect if it's datetime strings
                    sample_val = df['JULD'].dropna().iloc[0] if len(df['JULD'].dropna()) > 0 else None
                    if sample_val and isinstance(sample_val, str):
                        try:
                            pd.to_datetime(sample_val)
                            # It's datetime strings, convert directly
                            df['JULD'] = pd.to_datetime(df['JULD'], errors='coerce')
                        except:
                            # Not datetime strings, use JULD conversion
                            df['JULD'] = convert_juld_to_datetime(df['JULD'])
                    else:
                        df['JULD'] = convert_juld_to_datetime(df['JULD'])
                else:
                    df['JULD'] = convert_juld_to_datetime(df['JULD'])
            
            # Now extract date safely
            try:
                df['DATE'] = df['JULD'].dt.date
            except Exception as e:
                print(f"    Warning: Could not extract date from JULD: {e}")
                df['DATE'] = pd.to_datetime('2025-01-01').date()
        else:
            df['DATE'] = pd.to_datetime('2025-01-01').date()
        
        # Add derived columns for better analysis (following tp.py logic)
        print("  Adding derived columns...")
        
        # Add n_levels column (count of valid pressure measurements per profile)
        if 'PRES' in df.columns and 'PLATFORM_NUMBER' in df.columns and 'CYCLE_NUMBER' in df.columns:
            print("    Computing n_levels...")
            
            groupby_cols = ['PLATFORM_NUMBER', 'CYCLE_NUMBER']
            if 'JULD' in df.columns:
                groupby_cols.append('JULD')
            
            n_levels_data = []
            for name, group in df.groupby(groupby_cols):
                # Count valid pressure measurements
                valid_pres = group['PRES'].dropna()
                valid_pres = valid_pres[(valid_pres != 99999.0) & (valid_pres > -10) & np.isfinite(valid_pres)]
                n_levels = len(valid_pres)
                
                # Add n_levels to each row in this group
                for idx in group.index:
                    n_levels_data.append((idx, n_levels))
            
            # Create n_levels series and add to dataframe
            n_levels_series = pd.Series(index=df.index, dtype='int64')
            for idx, n_levels in n_levels_data:
                n_levels_series.loc[idx] = n_levels
            
            df['n_levels'] = n_levels_series
            print(f"      Added n_levels column (range: {df['n_levels'].min()}-{df['n_levels'].max()})")
        
        # Process JULD_LOCATION to extract only time portion
        if 'JULD_LOCATION' in df.columns:
            print("    Processing JULD_LOCATION to extract time only...")
            try:
                # Check if already datetime or needs conversion
                if not pd.api.types.is_datetime64_any_dtype(df['JULD_LOCATION']):
                    # Also check if it's already datetime strings
                    if df['JULD_LOCATION'].dtype == 'object':
                        # Try to detect if it's datetime strings
                        sample_val = df['JULD_LOCATION'].dropna().iloc[0] if len(df['JULD_LOCATION'].dropna()) > 0 else None
                        if sample_val and isinstance(sample_val, str):
                            try:
                                pd.to_datetime(sample_val)
                                # It's datetime strings, convert directly
                                df['JULD_LOCATION'] = pd.to_datetime(df['JULD_LOCATION'], errors='coerce')
                            except:
                                # Not datetime strings, use JULD conversion
                                df['JULD_LOCATION'] = convert_juld_to_datetime(df['JULD_LOCATION'])
                        else:
                            df['JULD_LOCATION'] = convert_juld_to_datetime(df['JULD_LOCATION'])
                    else:
                        df['JULD_LOCATION'] = convert_juld_to_datetime(df['JULD_LOCATION'])
                
                # Create only time column (no date column)
                df['JULD_LOCATION_time'] = df['JULD_LOCATION'].dt.time
                
                print(f"      Added JULD_LOCATION_time column (time only)")
            except Exception as e:
                print(f"      Warning: Could not process JULD_LOCATION: {e}")
        
        # Create comprehensive profile summaries using tp.py logic (preserving all 64+ variables)
        print("  Creating comprehensive profile summaries...")
        
        # Define grouping columns
        groupby_cols = ['PLATFORM_NUMBER', 'n_levels', 'DATE']
        
        # Check if grouping columns exist
        missing_cols = [col for col in groupby_cols if col not in df.columns]
        if missing_cols:
            print(f"    Warning: Missing grouping columns: {missing_cols}")
            # Use available columns
            groupby_cols = [col for col in groupby_cols if col in df.columns]
            if 'PLATFORM_NUMBER' not in groupby_cols:
                groupby_cols = ['PLATFORM_NUMBER', 'CYCLE_NUMBER', 'DATE']
        
        print(f"    Grouping by: {', '.join(groupby_cols)}")
        
        # Group the data
        grouped = df.groupby(groupby_cols)
        print(f"    Created {len(grouped)} groups")
        
        # Create summary for each group
        summary_rows = []
        
        for name, group in grouped:
            # Handle different grouping scenarios
            if len(groupby_cols) == 3:
                platform_num, n_levels_val, date = name
            elif len(groupby_cols) == 2:
                platform_num, date = name
                n_levels_val = group['n_levels'].iloc[0] if 'n_levels' in group.columns else len(group)
            else:
                platform_num = name
                date = group['DATE'].iloc[0] if 'DATE' in group.columns else pd.to_datetime('2025-01-01').date()
                n_levels_val = group['n_levels'].iloc[0] if 'n_levels' in group.columns else len(group)
            
            # Basic info
            summary = {
                'PLATFORM_NUMBER': platform_num,
                'n_levels': n_levels_val,
                'DATE': date,
                'row_count': len(group),
            }
            
            # Calculate oceanographic parameters
            if 'TEMP' in group.columns and 'PRES' in group.columns:
                try:
                    temp_data = group['TEMP'].dropna().values
                    pres_data = group['PRES'].dropna().values
                    
                    # Ensure we have matching data
                    if len(temp_data) > 0 and len(pres_data) > 0:
                        # Use minimum length to avoid index issues
                        min_len = min(len(temp_data), len(pres_data))
                        temp_data = temp_data[:min_len]
                        pres_data = pres_data[:min_len]
                        
                        # Calculate surface temperature statistics
                        surface_temp_stats = calculate_surface_temp(temp_data, pres_data)
                        summary['surface_temp_min_C'] = surface_temp_stats['min']
                        summary['surface_temp_max_C'] = surface_temp_stats['max']
                        summary['surface_temp_avg_C'] = surface_temp_stats['avg']
                        
                        # Calculate mixed layer depth
                        mixed_layer_depth = calculate_mixed_layer_depth(temp_data, pres_data)
                        summary['mixed_layer_depth_m'] = mixed_layer_depth
                    else:
                        summary['surface_temp_min_C'] = None
                        summary['surface_temp_max_C'] = None
                        summary['surface_temp_avg_C'] = None
                        summary['mixed_layer_depth_m'] = None
                except Exception as e:
                    print(f"      Warning: Could not calculate oceanographic parameters: {e}")
                    summary['surface_temp_min_C'] = None
                    summary['surface_temp_max_C'] = None
                    summary['surface_temp_avg_C'] = None
                    summary['mixed_layer_depth_m'] = None
            else:
                # No temperature or pressure data available
                summary['surface_temp_min_C'] = None
                summary['surface_temp_max_C'] = None
                summary['surface_temp_avg_C'] = None
                summary['mixed_layer_depth_m'] = None
            
            # Add all original columns with appropriate aggregation (tp.py logic)
            for col in df.columns:
                if col in ['PLATFORM_NUMBER', 'n_levels', 'DATE', 'row_count', 'JULD', 'JULD_LOCATION', 'PRES', 'PRES_ADJUSTED']:
                    continue  # Skip JULD, JULD_LOCATION, PRES, and PRES_ADJUSTED since we process PRES separately and PRES_ADJUSTED is empty
                    
                try:
                    # Special handling for JULD_LOCATION_time column
                    if col == 'JULD_LOCATION_time':
                        # For time columns, take just one time value (since they're usually the same)
                        time_data = group[col].dropna()
                        if len(time_data) > 0:
                            summary[f'{col}'] = time_data.iloc[0]
                        else:
                            summary[f'{col}'] = None
                        continue
                        
                    if df[col].dtype in ['object', 'string']:
                        # For string columns, take the first non-null value and clean b'' prefixes
                        non_null_values = group[col].dropna()
                        if len(non_null_values) > 0:
                            value = non_null_values.iloc[0]
                            # Clean up byte string representations like b'1223455677'
                            if isinstance(value, str) and value.startswith("b'") and value.endswith("'"):
                                value = value[2:-1]  # Remove b' and trailing '
                            elif isinstance(value, bytes):
                                value = value.decode('utf-8', errors='ignore')
                            
                            # Handle date columns with format like 20250101052756
                            if col in ['DATE_CREATION', 'DATE_UPDATE'] and isinstance(value, str) and len(value) == 14:
                                try:
                                    # Parse YYYYMMDDHHMMSS and convert to YYYY-MM-DD
                                    parsed_date = pd.to_datetime(value, format='%Y%m%d%H%M%S')
                                    value = parsed_date.strftime('%Y-%m-%d')
                                except:
                                    # If parsing fails, keep original value
                                    pass
                            
                            summary[f'{col}'] = value
                        else:
                            summary[f'{col}'] = None
                            
                    elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        # For numeric columns, provide statistics
                        numeric_data = group[col].dropna()
                        if len(numeric_data) > 0:
                            if col in ['PRES']:
                                # Calculate min/max/mean for pressure columns
                                summary[f'{col}_min'] = float(numeric_data.min())
                                summary[f'{col}_max'] = float(numeric_data.max())
                                summary[f'{col}_mean'] = float(numeric_data.mean())
                                
                                # Calculate depth statistics from pressure data (only for PRES, not PRES_ADJUSTED to avoid duplication)
                                if col == 'PRES':
                                    try:
                                        # Use a simple conversion: depth ≈ pressure (dbar to meters is approximately 1:1)
                                        # For more accurate conversion, could use gsw.z_from_p() but requires latitude
                                        # Using simple approximation: depth = pressure * 1.0 (good for most ocean depths)
                                        depth_data = numeric_data * 1.0  # Convert pressure to depth
                                        
                                        # Filter out invalid depth values
                                        valid_depth = depth_data[(depth_data >= 0) & (depth_data <= 12000) & np.isfinite(depth_data)]
                                        
                                        if len(valid_depth) > 0:
                                            summary['min_DEPTH_m'] = float(valid_depth.min())
                                            summary['max_DEPTH_m'] = float(valid_depth.max()) 
                                            summary['avg_DEPTH_m'] = float(valid_depth.mean())
                                        else:
                                            summary['min_DEPTH_m'] = None
                                            summary['max_DEPTH_m'] = None
                                            summary['avg_DEPTH_m'] = None
                                    except Exception as e:
                                        print(f"        Warning: Could not calculate depth statistics: {e}")
                                        summary['min_DEPTH_m'] = None
                                        summary['max_DEPTH_m'] = None
                                        summary['avg_DEPTH_m'] = None
                            elif col.endswith('_ERROR'):
                                # For ERROR columns, provide statistics since they contain uncertainty information
                                if len(numeric_data.unique()) == 1:
                                    # If all error values are the same (constant uncertainty), just take one
                                    summary[f'{col}'] = float(numeric_data.iloc[0])
                                else:
                                    # If error values vary, provide min/max/mean
                                    summary[f'{col}_min'] = float(numeric_data.min())
                                    summary[f'{col}_max'] = float(numeric_data.max())
                                    summary[f'{col}_mean'] = float(numeric_data.mean())
                            elif col in ['CYCLE_NUMBER', 'PLATFORM_NUMBER', 'CONFIG_MISSION_NUMBER', 'WMO_INST_TYPE']:
                                # For ID/categorical numbers, take first value (should be same for group)
                                summary[f'{col}'] = numeric_data.iloc[0]
                            elif col in ['LATITUDE', 'LONGITUDE']:
                                # For location data, keep it simple - just take first value
                                summary[f'{col}'] = numeric_data.iloc[0]
                            elif col in ['JULD_LOCATION_MIN', 'JULD_LOCATION_MAX']:
                                # For JULD location variables, provide range
                                summary[f'{col}_min'] = numeric_data.min()
                                summary[f'{col}_max'] = numeric_data.max()
                                if len(numeric_data.unique()) > 1:
                                    summary[f'{col}_mean'] = float(numeric_data.mean())
                                else:
                                    summary[f'{col}'] = numeric_data.iloc[0]
                            else:
                                # For other numeric variables, smart aggregation
                                if len(numeric_data.unique()) == 1:
                                    # If all values are the same, just take one
                                    summary[f'{col}'] = numeric_data.iloc[0]
                                elif col.startswith('HISTORY_') and 'PRES' in col:
                                    # For history pressure variables, take range
                                    summary[f'{col}_min'] = float(numeric_data.min())
                                    summary[f'{col}_max'] = float(numeric_data.max())
                                else:
                                    # For other varying numeric variables, take first value to avoid misleading averages
                                    summary[f'{col}'] = numeric_data.iloc[0]
                        else:
                            summary[f'{col}'] = None
                            
                    else:
                        # For other data types (including dates), take first value and clean b'' prefixes
                        non_null_values = group[col].dropna()
                        if len(non_null_values) > 0:
                            value = non_null_values.iloc[0]
                            # Clean up byte string representations like b'1223455677'
                            if isinstance(value, str) and value.startswith("b'") and value.endswith("'"):
                                value = value[2:-1]  # Remove b' and trailing '
                            elif isinstance(value, bytes):
                                value = value.decode('utf-8', errors='ignore')
                            
                            # Handle date columns with format like 20250101052756
                            if col in ['DATE_CREATION', 'DATE_UPDATE'] and isinstance(value, str) and len(value) == 14:
                                try:
                                    # Parse YYYYMMDDHHMMSS and convert to YYYY-MM-DD
                                    parsed_date = pd.to_datetime(value, format='%Y%m%d%H%M%S')
                                    value = parsed_date.strftime('%Y-%m-%d')
                                except:
                                    # If parsing fails, keep original value
                                    pass
                            
                            summary[f'{col}'] = value
                        else:
                            summary[f'{col}'] = None
                            
                except Exception as e:
                    print(f"      Warning: Could not process column {col}: {e}")
                    summary[f'{col}'] = None
            
            summary_rows.append(summary)
        
        # Create profile DataFrame
        profile_df = pd.DataFrame(summary_rows)
        print(f"  Created {len(profile_df)} profile summaries with {len(profile_df.columns)} columns")
        
        # Save to CSV in final_csv folder (but as CSV instead of Excel)
        output_file = output_file.replace('.xlsx', '.csv')
        profile_df.to_csv(output_file, index=False)
        print(f"  Profile-specific summary saved to: {output_file}")

    print("\n=== Final CSV Summary Creation Completed ===")

# ============================================================================
# MODULE 4: REGION ASSIGNMENT (from addingregion.py)
# ============================================================================

# Define Indian Ocean Regions with bounding boxes [min_lat, max_lat, min_lon, max_lon]
INDIAN_OCEAN_REGIONS = {
    # Red Sea and Northern Areas
    "Red Sea": [12, 30, 32, 44],
    "Gulf of Aqaba": [28, 30, 34, 36],  # Northern tip of Red Sea
    "Gulf of Suez": [27, 30, 32, 34],   # Northwestern Red Sea
    
    # Persian Gulf Region
    "Persian Gulf": [24, 30, 48, 57],
    "Gulf of Oman": [23, 27, 56, 60],   # Between Persian Gulf and Arabian Sea
    "Strait of Hormuz": [25, 27, 56, 58], # Connecting Persian Gulf to Gulf of Oman
    
    # Northern Indian Ocean
    "Arabian Sea": [5, 30, 50, 75],
    "Bay of Bengal": [5, 23, 80, 100],
    "Andaman Sea": [5, 20, 92, 100],
    "Gulf of Aden": [10, 18, 42, 52],   # Between Red Sea and Arabian Sea
    
    # Southern Indian Ocean - African Side
    "Mozambique Channel": [-25, -10, 35, 45],
    
    # Australian Waters
    "Great Australian Bight": [-38, -28, 125, 140], # Southern Australian coast
    "Gulf of Carpentaria": [-17, -10, 135, 142],    # Northern Australia
    "Arafura Sea": [-12, -8, 130, 140],             # Between Australia and New Guinea
    "Timor Sea": [-15, -8, 125, 135],               # Between Australia and Timor
    "Joseph Bonaparte Gulf": [-15, -13, 128, 130],  # Northwestern Australia
    
    # Sri Lankan Waters
    "Gulf of Mannar": [8, 10, 78, 80],              # Between India and Sri Lanka
    "Palk Strait and Palk Bay": [9, 10, 79, 80],   # Between India and Sri Lanka
    
    # Indian Waters
    "Lakshadweep Sea": [8, 15, 70, 77],             # Arabian Sea near Lakshadweep Islands
    
    # Additional Important Regions
    "Somali Basin": [-5, 15, 45, 60],               # Western Indian Ocean
    "Mascarene Basin": [-25, -10, 50, 70],          # Around Mauritius and Reunion
    "Central Indian Basin": [-20, 5, 75, 90],       # Central Indian Ocean
    "Wharton Basin": [-20, 5, 90, 110],             # Eastern Indian Ocean Basin
    "Perth Basin": [-35, -25, 110, 120],            # Western Australia offshore
    "Ninety East Ridge": [-30, 10, 88, 92],         # Major oceanic ridge
    "Chagos Archipelago": [-8, -4, 70, 75],         # British Indian Ocean Territory
    "Maldives Ridge": [-2, 8, 72, 75],              # Maldives region
    "Seychelles Plateau": [-10, 0, 50, 60],         # Seychelles region
    "Madagascar Ridge": [-25, -12, 43, 52],         # East of Madagascar
    "Crozet Basin": [-50, -40, 45, 65],             # Southern Indian Ocean
    "Kerguelen Plateau": [-55, -45, 65, 85],        # Subantarctic region
    "Java Trench": [-15, -5, 105, 115],             # Indonesian region
    "Banda Sea": [-8, -2, 125, 135],                # Indonesian seas
    "Celebes Sea": [-2, 8, 115, 125],               # Philippines/Indonesia
    "Coral Sea": [-25, -10, 145, 160],              # Northeast Australia
    "Tasman Sea": [-45, -25, 150, 170],             # Between Australia and New Zealand
    "Bass Strait": [-42, -38, 140, 150],            # Between Australia mainland and Tasmania
    "Spencer Gulf": [-35, -32, 136, 138],           # South Australia
    "Gulf of Thailand": [5, 15, 100, 106],          # Southeast Asia
    "South China Sea (Southern)": [5, 15, 106, 120], # Southern part adjacent to Indian Ocean
    
    # Additional Indian Ocean Exclusive Regions
    "Arabian Gulf": [24, 30, 48, 57],               # Alternative name for Persian Gulf
    "Oman Sea": [23, 27, 56, 63],                   # Extended Gulf of Oman
    "Hormuz Strait": [25, 27, 56, 58],              # Critical shipping lane
    "Makran Coast": [20, 28, 57, 67],               # Pakistan/Iran coastal waters
    "Indus Delta": [23, 25, 66, 68],                # Pakistan river delta
    "Kutch Gulf": [22, 24, 68, 70],                 # Western India
    "Cambay Gulf": [21, 23, 71, 73],                # Gujarat, India
    "Konkan Coast": [15, 21, 72, 74],               # Western India coast
    "Malabar Coast": [8, 15, 74, 76],               # Southwest India
    "Coromandel Coast": [8, 20, 79, 82],            # Southeast India coast
    "Bengal Shelf": [16, 23, 87, 92],               # Northern Bay of Bengal
    "Myanmar Coast": [10, 21, 92, 95],              # Myanmar coastal waters
    "Irrawaddy Delta": [15, 18, 94, 96],            # Myanmar river delta
    "Ten Degree Channel": [9, 11, 92, 93],          # Between Andaman Islands
    "Preparis Channel": [14, 16, 93, 95],           # Northern Andaman Sea
    "Malacca Strait": [1, 6, 99, 104],              # Between Malaysia and Sumatra
    "Sunda Strait": [-7, -5, 105, 107],             # Between Java and Somatra
    "Lombok Strait": [-9, -8, 115, 116],            # Between Bali and Lombok
    "Makassar Strait": [-6, 2, 116, 119],           # Between Borneo and Sulawesi
    "Flores Sea": [-9, -7, 118, 125],               # Indonesian waters
    "Sawu Sea": [-11, -9, 119, 123],                # Lesser Sunda Islands
    "Christmas Island Waters": [-11, -10, 105, 106], # Australian territory
    "Cocos Islands Waters": [-12, -11, 96, 97],     # Australian territory
    "Exmouth Plateau": [-22, -19, 112, 116],        # Western Australia offshore
    "Carnarvon Basin": [-26, -21, 112, 116],        # Western Australia offshore
    "Naturaliste Plateau": [-35, -32, 108, 112],    # Southwest Australia offshore
    "Wallaby Plateau": [-28, -25, 112, 115],        # Western Australia offshore
    "Broken Ridge": [-32, -29, 95, 98],             # Oceanic plateau
    "Amsterdam-St Paul Plateau": [-39, -37, 76, 78], # Southern Indian Ocean islands
    "Rodrigues Ridge": [-20, -18, 62, 64],          # Mauritius region
    "Saya de Malha Bank": [-11, -9, 59, 62],        # Shallow bank northeast of Mauritius
    "Nazareth Bank": [-15, -13, 56, 58],            # Southwest of Mauritius
    "Agulhas Bank": [-37, -34, 18, 26],             # Southern tip of Africa
    "Agulhas Plateau": [-41, -39, 23, 27],          # Southwest Indian Ridge
    "Prince Edward Islands": [-47, -46, 37, 38],    # South African territory
    "Marion Plateau": [-47, -46, 37, 38],           # Around Marion Island
    "Del Cano Rise": [-42, -40, 51, 53],            # Oceanic rise
    "Madagascar Plateau": [-26, -11, 46, 52],       # Around Madagascar
    "Comoros Basin": [-13, -11, 43, 46],            # Between Madagascar and Africa
    "Aldabra Group": [-10, -9, 46, 47],             # Seychelles outer islands
    "Farquhar Group": [-11, -10, 50, 52],           # Seychelles outer islands
    "Amirante Islands": [-6, -5, 52, 54],           # Seychelles archipelago
    "Socotra Island": [12, 13, 53, 54],             # Yemen territory
    "Abd al Kuri": [12, 13, 52, 53],                # Socotra archipelago
    "Hanish Islands": [13, 14, 42, 43],             # Red Sea islands
}

# Define the 5 major world oceans with bounding boxes [min_lat, max_lat, min_lon, max_lon]
WORLD_OCEANS = {
    "Indian Ocean": [-60, 30, 20, 147],      # Indian Ocean boundaries
    "Pacific Ocean": [-60, 70, 120, -70],    # Pacific Ocean (includes Western Pacific)
    "Atlantic Ocean": [-70, 80, -70, 20],    # Atlantic Ocean
    "Arctic Ocean": [65, 90, -180, 180],     # Arctic Ocean
    "Southern Ocean": [-90, -60, -180, 180], # Southern/Antarctic Ocean
}

def initialize_regions():
    """Convert bounding boxes to Polygon objects"""
    regions = {}
    for region_name, bbox in INDIAN_OCEAN_REGIONS.items():
        min_lat, max_lat, min_lon, max_lon = bbox
        # Create polygon from bounding box
        polygon = Polygon([
            (min_lon, min_lat),
            (max_lon, min_lat), 
            (max_lon, max_lat),
            (min_lon, max_lat)
        ])
        regions[region_name] = polygon
    return regions

def initialize_oceans():
    """Convert ocean bounding boxes to Polygon objects"""
    oceans = {}
    for ocean_name, bbox in WORLD_OCEANS.items():
        min_lat, max_lat, min_lon, max_lon = bbox
        # Handle longitude wrapping for Pacific Ocean
        if ocean_name == "Pacific Ocean":
            # Pacific Ocean spans across the 180° meridian
            # Create one large polygon covering the Pacific
            polygon = Polygon([
                (120, min_lat), (120, max_lat), (180, max_lat), (180, min_lat),
                (180, min_lat), (-180, min_lat), (-180, max_lat), (-70, max_lat),
                (-70, min_lat), (120, min_lat)
            ])
            oceans[ocean_name] = polygon
        else:
            # Regular polygon for other oceans
            polygon = Polygon([
                (min_lon, min_lat),
                (max_lon, min_lat), 
                (max_lon, max_lat),
                (min_lon, max_lat)
            ])
            oceans[ocean_name] = polygon
    return oceans

def assign_ocean(lat: float, lon: float, oceans: dict) -> str:
    """Return ocean name based on latitude and longitude coordinates."""
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown Ocean"
    
    try:
        # Simplified ocean assignment based on geographic boundaries
        # More accurate than polygon-based for global coverage
        
        # Arctic Ocean - northern regions
        if lat >= 65:
            return "Arctic Ocean"
        
        # Southern/Antarctic Ocean - southern regions
        if lat <= -60:
            return "Southern Ocean"
        
        # Atlantic Ocean - western longitudes
        if -70 <= lon <= 20:
            return "Atlantic Ocean"
        
        # Indian Ocean - our primary focus area
        if 20 <= lon <= 147:
            return "Indian Ocean"
        
        # Pacific Ocean - eastern and western extremes
        if lon >= 120 or lon <= -70:
            return "Pacific Ocean"
        
        # Default fallback
        return "Unknown Ocean"
            
    except Exception as e:
        print(f"Error assigning ocean for lat={lat}, lon={lon}: {e}")
        return "Unknown Ocean"

def assign_region(lat: float, lon: float, regions: dict) -> str:
    """Return Indian Ocean region name if (lat, lon) falls inside a polygon, else 'Unknown Region'."""
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown Region"
    
    try:
        point = Point(lon, lat)  # shapely uses (x=lon, y=lat)
        
        # Check each region polygon
        for region_name, polygon in regions.items():
            if polygon.contains(point):
                return region_name
        
        # If no region contains the point
        return "Unknown Region"
            
    except Exception as e:
        print(f"Error assigning region for lat={lat}, lon={lon}: {e}")
        return "Unknown Region"

def has_region_columns(df):
    """Check if the DataFrame already has region and ocean columns"""
    region_columns = ['avg_REGION', 'ocean']
    return all(col in df.columns for col in region_columns)

def add_region_columns(df, regions, oceans):
    """Add region and ocean columns based on latitude/longitude data"""
    print("    Adding region and ocean columns...")
    
    # Check if required columns exist
    required_cols = ['LATITUDE', 'LONGITUDE']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"    Warning: Missing required columns: {missing_cols}")
        return df
    
    # Calculate avg_REGION using LATITUDE and LONGITUDE
    print("      Adding region column...")
    df['avg_REGION'] = df.apply(
        lambda row: assign_region(row['LATITUDE'], row['LONGITUDE'], regions), 
        axis=1
    )
    
    # Calculate single OCEAN column using LATITUDE and LONGITUDE
    print("      Adding ocean column...")
    df['ocean'] = df.apply(
        lambda row: assign_ocean(row['LATITUDE'], row['LONGITUDE'], oceans), 
        axis=1
    )
    
    print("    Region and ocean columns added successfully!")
    return df

def add_regions_to_final_csv():
    """Add region and ocean information to all final CSV files"""
    print("\n" + "="*50)
    print("STEP 4: ADDING REGION AND OCEAN INFORMATION")
    print("="*50)
    
    # Initialize regions and oceans
    regions = initialize_regions()
    oceans = initialize_oceans()
    print(f"Initialized {len(regions)} Indian Ocean regions")
    print(f"Initialized {len(oceans)} world oceans")
    
    # Find all CSV files in final_csv structure
    csv_files = find_csv_files_in_structure("final_csv")
    
    if not csv_files:
        print("No CSV files found in final_csv folder structure.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    processed_files = 0
    skipped_files = 0
    total_profiles = 0
    region_stats = {}
    ocean_stats = {}
    
    for file_info in csv_files:
        try:
            file_path = file_info['path']
            filename = file_info['filename']
            
            print(f"  Processing: {filename}")
            
            # Read the CSV file
            df = pd.read_csv(file_path, low_memory=False)
            
            # Check if region and ocean columns already exist
            if has_region_columns(df):
                print(f"    Skipping - Region and ocean columns already exist")
                skipped_files += 1
                
                # Still count regions and oceans for statistics
                if 'avg_REGION' in df.columns:
                    for region in df['avg_REGION'].values:
                        region_stats[region] = region_stats.get(region, 0) + 1
                if 'ocean' in df.columns:
                    for ocean in df['ocean'].values:
                        ocean_stats[ocean] = ocean_stats.get(ocean, 0) + 1
                total_profiles += len(df)
            else:
                # Add region and ocean columns
                df_with_geo = add_region_columns(df, regions, oceans)
                
                # Save the updated DataFrame back to CSV
                df_with_geo.to_csv(file_path, index=False)
                print(f"    Updated and saved: {file_path}")
                processed_files += 1
                
                # Count regions and oceans for statistics
                for region in df_with_geo['avg_REGION'].values:
                    region_stats[region] = region_stats.get(region, 0) + 1
                for ocean in df_with_geo['ocean'].values:
                    ocean_stats[ocean] = ocean_stats.get(ocean, 0) + 1
                total_profiles += len(df_with_geo)
                
        except Exception as e:
            print(f"    Error processing {file_info['filename']}: {e}")
    
    print(f"\nGeographic processing completed:")
    print(f"  Files processed: {processed_files}")
    print(f"  Files skipped: {skipped_files}")
    print(f"  Total profiles: {total_profiles}")
    
    print("\nRegion assignment statistics:")
    for region, count in sorted(region_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_profiles) * 100 if total_profiles > 0 else 0
        print(f"  {region}: {count} profiles ({percentage:.1f}%)")
    
    print("\nOcean assignment statistics:")
    for ocean, count in sorted(ocean_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_profiles) * 100 if total_profiles > 0 else 0
        print(f"  {ocean}: {count} profiles ({percentage:.1f}%)")
    
    print("\n=== Region and Ocean Assignment Completed ===")

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution function that runs the complete pipeline"""
    print("\n" + "="*60)
    print("NETCDF TO EXCEL PROCESSING PIPELINE")
    print("="*60)
    print(f"Year: {YEAR}")
    print(f"Months: {[get_month_name(m) for m in MONTHS_TO_PROCESS]}")
    print("="*60)
    
    try:
        # Step 1: Download NetCDF data
        download_netcdf_data(YEAR, MONTHS_TO_PROCESS)
        
        # Step 2: Convert NetCDF to CSV
        convert_netcdf_to_csv()
        
        # Step 3: Create final CSV summaries
        create_final_csv_summaries()
        
        # Step 4: Add region information to final CSV files
        add_regions_to_final_csv()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Results:")
        print(f"• NetCDF files: Data/NETcdf_Data/argo_{YEAR}/[month]/")
        print(f"• Individual CSV files: Data/rough_csv_Data/{YEAR}/[month]/")
        print(f"• Final summaries: Data/final_csv/{YEAR}/[month]/final_[MON]_days.csv")
        print(f"• Region column: avg_REGION (based on LATITUDE/LONGITUDE) added")
        print(f"• Ocean column: ocean (based on LATITUDE/LONGITUDE) added")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR in pipeline execution: {e}")
        print("Pipeline stopped due to error.")

if __name__ == "__main__":
    main()
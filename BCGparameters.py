"""
BCG Parameters Extractor for Argo Bio-Profile Data
==================================================
This script fetches data from the Argo bio-profile index and extracts 
platform numbers for Indian Ocean based on year and month filters.

Data Source: https://data-argo.ifremer.fr/argo_bio-profile_index.txt

Output: CSV file containing platform numbers and related data for Indian Ocean
Location: Data/BCG floats/
"""

import pandas as pd
import requests
from datetime import datetime
import os
from ftplib import FTP
import io
import time
import xarray as xr
import numpy as np

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

def fetch_argo_bio_index_ftp():
    """
    Fetch Argo bio-profile index using FTP
    Returns the content as text
    """
    try:
        print("Attempting to fetch data via FTP...")
        ftp = FTP("ftp.ifremer.fr", timeout=30)
        ftp.login()
        
        print("Connected to FTP server, listing directories...")
        
        # Try different possible paths for the bio-profile index
        possible_paths = [
            "/ifremer/argo/argo_bio-profile_index.txt",
            "/ifremer/argo/argo_bio-profile_index.txt",
            "argo_bio-profile_index.txt",
            "/argo_bio-profile_index.txt"
        ]
        
        content = None
        for path in possible_paths:
            try:
                print(f"  Trying path: {path}")
                
                # If path contains directory, navigate to it first
                if '/' in path and not path.startswith('/'):
                    dir_path = '/'.join(path.split('/')[:-1])
                    ftp.cwd(dir_path)
                    filename = path.split('/')[-1]
                elif path.startswith('/') and '/' in path[1:]:
                    dir_path = '/'.join(path.split('/')[:-1])
                    ftp.cwd(dir_path)
                    filename = path.split('/')[-1]
                else:
                    filename = path
                
                # Create a bytes buffer to store the file content
                bio_data = io.BytesIO()
                
                # Download the file
                ftp.retrbinary(f"RETR {filename}", bio_data.write)
                
                # Convert bytes to string
                bio_data.seek(0)
                content = bio_data.read().decode('utf-8')
                print(f"  Successfully fetched data via FTP from: {path}")
                break
                
            except Exception as path_error:
                print(f"  Failed to fetch from {path}: {path_error}")
                # Reset to root directory for next attempt
                try:
                    ftp.cwd("/")
                except:
                    pass
                continue
        
        ftp.quit()
        return content
        
    except Exception as e:
        print(f"FTP method failed: {e}")
        return None

def fetch_argo_bio_index_http():
    """
    Fetch Argo bio-profile index using HTTP as fallback
    Returns the content as text
    """
    try:
        print("Attempting to fetch data via HTTP...")
        url = "https://data-argo.ifremer.fr/argo_bio-profile_index.txt"
        
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"Fetching from URL: {url}")
        response = requests.get(url, timeout=120, headers=headers, stream=True)
        response.raise_for_status()
        
        print(f"Response status: {response.status_code}")
        print(f"Content length: {response.headers.get('content-length', 'Unknown')}")
        
        # Get content
        content = response.text
        
        # Basic validation
        if len(content) < 1000:  # Bio-profile index should be much larger
            print(f"Warning: Content seems too short ({len(content)} characters)")
            print("First 500 characters:")
            print(content[:500])
        else:
            print(f"Successfully fetched {len(content)} characters via HTTP")
        
        return content
        
    except requests.exceptions.RequestException as e:
        print(f"HTTP request failed: {e}")
        return None
    except Exception as e:
        print(f"HTTP method failed: {e}")
        return None

def test_ftp_connection():
    """
    Test FTP connection and explore directory structure
    """
    try:
        print("Testing FTP connection...")
        ftp = FTP("ftp.ifremer.fr", timeout=30)
        ftp.login()
        print("Successfully connected to FTP server")
        
        print("Current directory:", ftp.pwd())
        print("Listing root directory:")
        files = ftp.nlst()
        for file in files[:20]:  # Show first 20 items
            print(f"  {file}")
        
        # Try to find argo-related directories
        for item in files:
            if 'argo' in item.lower():
                print(f"Found argo-related item: {item}")
                try:
                    ftp.cwd(item)
                    print(f"Contents of {item}:")
                    sub_files = ftp.nlst()
                    for sub_file in sub_files[:10]:
                        print(f"    {sub_file}")
                    ftp.cwd("..")
                except:
                    print(f"    Could not access {item}")
        
        ftp.quit()
        
    except Exception as e:
        print(f"FTP connection test failed: {e}")

def fetch_argo_bio_index():
    """
    Fetch Argo bio-profile index, trying HTTP first, then FTP
    """
    print("Fetching Argo bio-profile index...")
    
    # Try HTTP first (more reliable)
    content = fetch_argo_bio_index_http()
    
    # If HTTP fails, try FTP
    if content is None:
        print("HTTP failed, trying FTP...")
        content = fetch_argo_bio_index_ftp()
    
    # If both fail, try testing FTP connection
    if content is None:
        print("Both HTTP and FTP failed. Testing FTP connection...")
        test_ftp_connection()
        raise Exception("Failed to fetch data using both HTTP and FTP methods")
    
    return content

def parse_argo_bio_data(content):
    """
    Parse the Argo bio-profile index content into a DataFrame
    Expected format: file,date,latitude,longitude,ocean,profiler_type,institution,parameters,parameter_data_mode,date_update
    """
    print("Parsing Argo bio-profile data...")
    
    lines = content.strip().split('\n')
    
    # Skip header line if present (usually starts with '#' or contains column names)
    data_lines = []
    for line in lines:
        if line.startswith('#') or 'file,date,latitude' in line.lower():
            continue
        if line.strip():  # Skip empty lines
            data_lines.append(line.strip())
    
    print(f"Found {len(data_lines)} data records")
    
    # Parse each line
    parsed_data = []
    for i, line in enumerate(data_lines):
        try:
            parts = line.split(',')
            if len(parts) >= 10:  # Ensure we have all required fields
                
                # Extract platform number from file path (e.g., aoml/1900722/profiles/BD1900722_001.nc)
                file_path = parts[0]
                platform_number = None
                
                # Extract platform number from the file path
                path_parts = file_path.split('/')
                for part in path_parts:
                    if part.isdigit() and len(part) >= 7:  # Platform numbers are typically 7+ digits
                        platform_number = part
                        break
                
                if platform_number is None:
                    # Try to extract from filename
                    filename = path_parts[-1] if path_parts else ''
                    for part in filename.split('_'):
                        if part.replace('BD', '').replace('SD', '').replace('AD', '').replace('RD', '').isdigit():
                            platform_number = part.replace('BD', '').replace('SD', '').replace('AD', '').replace('RD', '')
                            break
                
                record = {
                    'file': parts[0],
                    'date': parts[1],
                    'latitude': float(parts[2]),
                    'longitude': float(parts[3]),
                    'ocean': parts[4],
                    'profiler_type': parts[5],
                    'institution': parts[6],
                    'parameters': parts[7],
                    'parameter_data_mode': parts[8],
                    'date_update': parts[9],
                    'platform_number': platform_number
                }
                parsed_data.append(record)
            
        except Exception as e:
            print(f"Warning: Could not parse line {i+1}: {line[:100]}... Error: {e}")
            continue
    
    df = pd.DataFrame(parsed_data)
    print(f"Successfully parsed {len(df)} records")
    return df

def filter_indian_ocean_data(df, target_year, target_month):
    """
    Filter data for Indian Ocean and specific year/month
    """
    print(f"Filtering for Indian Ocean data in {get_month_name(target_month)} {target_year}...")
    
    # Filter for Indian Ocean (ocean = 'I')
    indian_ocean_df = df[df['ocean'] == 'I'].copy()
    print(f"Found {len(indian_ocean_df)} Indian Ocean records")
    
    if len(indian_ocean_df) == 0:
        print("No Indian Ocean records found!")
        return pd.DataFrame()
    
    # Parse date and filter by year/month
    print("Parsing dates and filtering by year/month...")
    
    def parse_date(date_str):
        """Parse date string in format YYYYMMDDHHMMSS"""
        try:
            if len(date_str) >= 8:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                return year, month, day
            return None, None, None
        except:
            return None, None, None
    
    # Apply date parsing
    date_info = indian_ocean_df['date'].apply(parse_date)
    indian_ocean_df['year'] = [info[0] for info in date_info]
    indian_ocean_df['month'] = [info[1] for info in date_info]
    indian_ocean_df['day'] = [info[2] for info in date_info]
    
    # Filter by target year and month
    filtered_df = indian_ocean_df[
        (indian_ocean_df['year'] == target_year) & 
        (indian_ocean_df['month'] == target_month)
    ].copy()
    
    print(f"Found {len(filtered_df)} records for {get_month_name(target_month)} {target_year}")
    
    # Remove duplicates based on platform number to get unique platforms
    if len(filtered_df) > 0:
        unique_platforms_df = filtered_df.drop_duplicates(subset=['platform_number'])
        print(f"Found {len(unique_platforms_df)} unique platform numbers")
        return unique_platforms_df
    
    return filtered_df

def save_to_csv(df, year, month):
    """
    Save platform numbers to a single CSV in the Data/BCG floats/unique floats folder
    Appends data if file already exists to maintain a single comprehensive list
    """
    if len(df) == 0:
        print("No data to save")
        return None
    
    # Create directory structure
    base_dir = os.path.join("Data", "BCG floats", "unique floats")
    os.makedirs(base_dir, exist_ok=True)
    
    # Fixed filename for single CSV
    filename = "BCG floats Platform number list.csv"
    filepath = os.path.join(base_dir, filename)
    
    # Extract only platform numbers and remove duplicates from current data
    new_platform_numbers = df['platform_number'].dropna().unique()
    new_platform_numbers = [str(pn) for pn in new_platform_numbers if pn is not None]
    
    print(f"Found {len(new_platform_numbers)} unique platform numbers for {get_month_name(month)} {year}")
    
    # Check if file already exists
    existing_platforms = set()
    if os.path.exists(filepath):
        print(f"Existing file found: {filepath}")
        try:
            existing_df = pd.read_csv(filepath)
            if 'platform_number' in existing_df.columns:
                existing_platforms = set(str(pn) for pn in existing_df['platform_number'].dropna().unique())
                print(f"Found {len(existing_platforms)} existing platform numbers")
            else:
                print("Warning: Existing file doesn't have 'platform_number' column")
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}")
            existing_platforms = set()
    else:
        print("No existing file found, creating new one")
    
    # Combine existing and new platform numbers, removing duplicates
    all_platforms = existing_platforms.union(set(new_platform_numbers))
    newly_added = set(new_platform_numbers) - existing_platforms
    
    print(f"New platform numbers to add: {len(newly_added)}")
    print(f"Total unique platform numbers after merge: {len(all_platforms)}")
    
    # Create DataFrame with only platform numbers
    platform_df = pd.DataFrame({
        'platform_number': sorted(list(all_platforms))
    })
    
    # Save to CSV (overwrites the file with complete list)
    platform_df.to_csv(filepath, index=False)
    
    print(f"Platform numbers saved to: {filepath}")
    print(f"Total platforms in file: {len(platform_df)}")
    
    # Print summary statistics
    print(f"\nSummary for {get_month_name(month)} {year}:")
    print(f"- New platforms added: {len(newly_added)}")
    print(f"- Total unique platforms in file: {len(all_platforms)}")
    
    # Show sample of newly added platform numbers
    if newly_added:
        sample_new = list(newly_added)[:10]
        print(f"- Sample new platforms: {sample_new}{'...' if len(newly_added) > 10 else ''}")
    else:
        print("- No new platforms added (all were already in the list)")
    
    return filepath

def download_sprof_files():
    """
    Download Sprof.nc NetCDF files for all unique float platform numbers in the CSV list
    """
    print("\n" + "="*60)
    print("DOWNLOADING NETCDF FILES FOR UNIQUE FLOATS")
    print("="*60)
    
    # Path to the platform numbers CSV
    csv_path = os.path.join("Data", "BCG floats", "unique floats", "BCG floats Platform number list.csv")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Unique floats platform numbers CSV not found at: {csv_path}")
        print("Please run the extraction first to generate the unique floats platform list.")
        return
    
    # Read platform numbers
    try:
        df = pd.read_csv(csv_path)
        if 'platform_number' not in df.columns:
            print("ERROR: 'platform_number' column not found in CSV")
            return
        
        platform_numbers = df['platform_number'].dropna().astype(str).unique()
        print(f"Found {len(platform_numbers)} unique float platform numbers to process")
        
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        return
    
    # Create output directory
    output_dir = os.path.join("Data", "BCG floats", "netcdf files")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Download counters
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Process each platform number
    for i, platform_num in enumerate(platform_numbers, 1):
        print(f"\n[{i}/{len(platform_numbers)}] Processing unique float: {platform_num}")
        
        # Generate filename and paths
        filename = f"{platform_num}_Sprof.nc"
        local_path = os.path.join(output_dir, filename)
        
        # Check if file already exists
        if os.path.exists(local_path):
            print(f"  Skipping - File already exists: {filename}")
            skipped_count += 1
            continue
        
        # Generate download URL (limited to aoml only)
        url = f"https://data-argo.ifremer.fr/dac/aoml/{platform_num}/{filename}"
        
        print(f"  Downloading from: {url}")
        
        try:
            # Download with timeout and headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=60, stream=True)
            
            if response.status_code == 200:
                # Save the file
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                file_size = os.path.getsize(local_path)
                print(f"  ✓ Downloaded successfully: {filename} ({file_size:,} bytes)")
                downloaded_count += 1
                
            elif response.status_code == 404:
                print(f"  ✗ File not found (404): {filename}")
                failed_count += 1
                
            else:
                print(f"  ✗ HTTP Error {response.status_code}: {filename}")
                failed_count += 1
                
        except requests.exceptions.Timeout:
            print(f"  ✗ Timeout downloading: {filename}")
            failed_count += 1
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Request failed for {filename}: {e}")
            failed_count += 1
            
        except Exception as e:
            print(f"  ✗ Unexpected error downloading {filename}: {e}")
            failed_count += 1
        
        # Small delay to be respectful to the server
        time.sleep(0.5)
    
    # Summary
    print("="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total unique floats processed: {len(platform_numbers)}")
    print(f"NetCDF files downloaded: {downloaded_count}")
    print(f"Files skipped (already exist): {skipped_count}")
    print(f"Failed downloads: {failed_count}")
    print(f"Output directory: {output_dir}")
    
    if downloaded_count > 0:
        print(f"\n✓ Successfully downloaded {downloaded_count} unique float NetCDF files")
    
    if failed_count > 0:
        print(f"\n⚠ {failed_count} downloads failed (files may not exist or be inaccessible)")

def decode_bytes(arr):
    """Function to decode byte strings"""
    if arr.dtype.kind in {'S', 'O'}:  # bytes or object
        return np.array([x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in arr])
    return arr

def ensure_directories_exist():
    """Create all necessary directories if they don't exist"""
    directories = [
        os.path.join("Data"),
        os.path.join("Data", "BCG floats"),
        os.path.join("Data", "BCG floats", "unique floats"),
        os.path.join("Data", "BCG floats", "netcdf files"),
        os.path.join("Data", "BCG floats", "raw csv"),
        os.path.join("Data", "BCG floats", "final csv files")
    ]
    
    print("Ensuring all necessary directories exist...")
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"  Created directory: {directory}")
        else:
            print(f"  Directory exists: {directory}")
    print("Directory structure ready.")

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
        
        print(f"  Processing {len(ds.variables)} variables from {nc_file_path}")
        print(f"  Dimensions: N_PROF={n_prof}, N_LEVELS={n_levels}, N_PARAM={n_param}, N_CALIB={n_calib}, N_HISTORY={n_history}")
        
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
                        
                elif dims == ('N_PROF', 'N_LEVELS'):
                    # Measurement-level variable
                    measurement_vars[var_name] = values
                    
                elif dims == ():
                    # Scalar variable - same for all profiles
                    if var_data.dtype.kind in {'S', 'O'}:
                        scalar_vars[var_name] = decode_bytes(np.array([values]))[0]
                    else:
                        scalar_vars[var_name] = values
                    
                elif dims == ('N_PROF', 'N_PARAM'):
                    # Parameter-related variables - take first parameter for each profile
                    if n_param > 0:
                        if var_data.dtype.kind in {'S', 'O'}:
                            # Handle string arrays
                            decoded_values = []
                            for i in range(n_prof):
                                try:
                                    char_array = values[i, 0] if values.shape[1] > 0 else ''
                                    if isinstance(char_array, bytes):
                                        decoded_values.append(char_array.decode('utf-8', errors='ignore'))
                                    else:
                                        decoded_values.append(str(char_array))
                                except:
                                    decoded_values.append('')
                            profile_vars[var_name] = np.array(decoded_values)
                        else:
                            # Take first parameter for each profile
                            profile_vars[var_name] = values[:, 0] if values.shape[1] > 0 else np.full(n_prof, np.nan)
                    
                elif dims == ('N_PROF', 'N_CALIB', 'N_PARAM'):
                    # Calibration-related variables - take first calibration and first parameter
                    if n_calib > 0 and n_param > 0:
                        if var_data.dtype.kind in {'S', 'O'}:
                            # Handle string arrays
                            decoded_values = []
                            for i in range(n_prof):
                                try:
                                    char_array = values[i, 0, 0] if values.shape[1] > 0 and values.shape[2] > 0 else ''
                                    if isinstance(char_array, bytes):
                                        decoded_values.append(char_array.decode('utf-8', errors='ignore'))
                                    else:
                                        decoded_values.append(str(char_array))
                                except:
                                    decoded_values.append('')
                            profile_vars[var_name] = np.array(decoded_values)
                        else:
                            # Take first calibration, first parameter for each profile
                            if values.shape[1] > 0 and values.shape[2] > 0:
                                profile_vars[var_name] = values[:, 0, 0]
                            else:
                                profile_vars[var_name] = np.full(n_prof, np.nan)
                    
                elif dims == ('N_HISTORY', 'N_PROF'):
                    # History variables - if N_HISTORY > 0, take last history entry for each profile
                    if n_history > 0:
                        if var_data.dtype.kind in {'S', 'O'}:
                            # Handle string arrays
                            decoded_values = []
                            for i in range(n_prof):
                                try:
                                    char_array = values[-1, i] if values.shape[0] > 0 else ''
                                    if isinstance(char_array, bytes):
                                        decoded_values.append(char_array.decode('utf-8', errors='ignore'))
                                    else:
                                        decoded_values.append(str(char_array))
                                except:
                                    decoded_values.append('')
                            profile_vars[var_name] = np.array(decoded_values)
                        else:
                            # Take last history entry for each profile
                            if values.shape[0] > 0:
                                profile_vars[var_name] = values[-1, :]
                            else:
                                profile_vars[var_name] = np.full(n_prof, np.nan)
                    else:
                        # No history data available
                        if var_data.dtype.kind in {'S', 'O'}:
                            profile_vars[var_name] = np.full(n_prof, '', dtype=object)
                        else:
                            profile_vars[var_name] = np.full(n_prof, np.nan)
                    
                elif dims == ('N_PROF', 'STRING_length') or 'STRING' in str(dims):
                    # String variables with character arrays
                    if var_data.dtype.kind in {'S', 'O'}:
                        # Decode and join character arrays
                        decoded_values = []
                        for i in range(n_prof):
                            if len(values.shape) == 2:
                                char_array = values[i, :]
                            else:
                                char_array = values[i]
                            
                            if isinstance(char_array, bytes):
                                decoded_values.append(char_array.decode('utf-8', errors='ignore'))
                            elif hasattr(char_array, 'tobytes'):
                                decoded_values.append(char_array.tobytes().decode('utf-8', errors='ignore'))
                            else:
                                decoded_values.append(str(char_array))
                        
                        profile_vars[var_name] = np.array(decoded_values)
                    else:
                        profile_vars[var_name] = values
                        
                elif 'N_PROF' in dims:
                    # Other variables containing N_PROF - try to handle generically
                    try:
                        # Find the position of N_PROF in dimensions
                        prof_idx = dims.index('N_PROF')
                        
                        if prof_idx == 0:  # N_PROF is first dimension
                            if len(dims) == 1:
                                profile_vars[var_name] = values
                            elif len(dims) == 2:
                                profile_vars[var_name] = values[:, 0] if values.shape[1] > 0 else np.full(n_prof, np.nan)
                            else:
                                # Take first element along other dimensions
                                profile_vars[var_name] = values.flat[:n_prof] if values.size >= n_prof else np.full(n_prof, np.nan)
                        else:
                            # N_PROF is not first dimension - transpose to make it first
                            axes = list(range(len(dims)))
                            axes[0], axes[prof_idx] = axes[prof_idx], axes[0]
                            transposed = np.transpose(values, axes)
                            profile_vars[var_name] = transposed[:, 0] if transposed.shape[1] > 0 else np.full(n_prof, np.nan)
                        
                    except Exception as e:
                        print(f"    Could not handle complex variable {var_name}: {e}")
                            
                else:
                    print(f"  Skipping variable {var_name} with unsupported dims: {dims}")
                    
            except Exception as e:
                print(f"Warning: Could not extract variable {var_name}: {e}")
        
        # Create flattened DataFrame (no date conversion)
        rows = []
        
        # Combine all column names
        profile_columns = list(sorted(profile_vars.keys()))
        measurement_columns = list(sorted(measurement_vars.keys()))
        scalar_columns = list(sorted(scalar_vars.keys()))
        
        columns = profile_columns + measurement_columns + scalar_columns
        
        print(f"  Creating DataFrame with {len(columns)} columns:")
        print(f"    Profile vars: {len(profile_columns)}")
        print(f"    Measurement vars: {len(measurement_columns)}")
        print(f"    Scalar vars: {len(scalar_columns)}")
        
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
        
        print(f"  Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"Error extracting variables from {nc_file_path}: {e}")
        return None

def convert_netcdf_to_csv():
    """
    Convert all unique float NetCDF files in 'netcdf files' to CSV format
    """
    print("\n" + "="*60)
    print("CONVERTING UNIQUE FLOAT NETCDF FILES TO CSV")
    print("="*60)
    
    # Source and destination directories
    source_dir = os.path.join("Data", "BCG floats", "netcdf files")
    dest_dir = os.path.join("Data", "BCG floats", "raw csv")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"ERROR: Source directory not found: {source_dir}")
        print("Please download unique float NetCDF files first using option 2 or 4.")
        return
    
    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}")
    
    # Find all NetCDF files
    nc_files = [f for f in os.listdir(source_dir) if f.endswith('.nc')]
    
    if not nc_files:
        print("No unique float NetCDF files found in source directory.")
        return
    
    print(f"Found {len(nc_files)} unique float NetCDF files to convert")
    
    # Conversion counters
    converted_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Process each NetCDF file
    for i, nc_filename in enumerate(nc_files, 1):
        print(f"\n[{i}/{len(nc_files)}] Processing: {nc_filename}")
        
        # Generate CSV filename
        csv_filename = nc_filename.replace('.nc', '.csv')
        nc_path = os.path.join(source_dir, nc_filename)
        csv_path = os.path.join(dest_dir, csv_filename)
        
        # Check if CSV already exists
        if os.path.exists(csv_path):
            print(f"  Skipping - CSV already exists: {csv_filename}")
            skipped_count += 1
            continue
        
        try:
            # Extract all variables from NetCDF
            df = extract_all_variables_from_netcdf(nc_path)
            
            if df is not None and len(df) > 0:
                # Save to CSV
                df.to_csv(csv_path, index=False)
                print(f"  ✓ Converted successfully: {csv_filename} ({len(df)} rows, {len(df.columns)} columns)")
                converted_count += 1
            else:
                print(f"  ✗ Failed to extract data: {nc_filename}")
                failed_count += 1
                
        except Exception as e:
            print(f"  ✗ Error converting {nc_filename}: {e}")
            failed_count += 1
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.1)
    
    # Summary
    print("="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Total unique float NetCDF files processed: {len(nc_files)}")
    print(f"Files converted: {converted_count}")
    print(f"Files skipped (already exist): {skipped_count}")
    print(f"Failed conversions: {failed_count}")
    print(f"Output directory: {dest_dir}")
    
    if converted_count > 0:
        print(f"\n✓ Successfully converted {converted_count} unique float NetCDF files to CSV")
    
    if failed_count > 0:
        print(f"\n⚠ {failed_count} conversions failed")

def process_raw_csv_to_final():
    """
    Process raw CSV files to create final aggregated CSV files grouped by platform, cycle, and date
    """
    print("\n" + "="*60)
    print("PROCESSING RAW CSV TO FINAL AGGREGATED CSV")
    print("="*60)
    
    # Define Indian Ocean Regions
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
        "Sunda Strait": [-7, -5, 105, 107],             # Between Java and Sumatra
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
    
    # Define World Oceans
    WORLD_OCEANS = {
        "Indian Ocean": [-60, 30, 20, 147],      # Indian Ocean boundaries
        "Pacific Ocean": [-60, 70, 120, -70],    # Pacific Ocean (includes Western Pacific)
        "Atlantic Ocean": [-70, 80, -70, 20],    # Atlantic Ocean
        "Arctic Ocean": [65, 90, -180, 180],     # Arctic Ocean
        "Southern Ocean": [-90, -60, -180, 180], # Southern/Antarctic Ocean
    }
    
    def assign_region(lat, lon):
        """Assign region based on latitude and longitude"""
        if pd.isna(lat) or pd.isna(lon):
            return "Unknown"
        
        for region_name, bbox in INDIAN_OCEAN_REGIONS.items():
            min_lat, max_lat, min_lon, max_lon = bbox
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                return region_name
        return "Unknown"
    
    def assign_ocean(lat, lon):
        """Assign ocean based on latitude and longitude"""
        if pd.isna(lat) or pd.isna(lon):
            return "Unknown"
        
        for ocean_name, bbox in WORLD_OCEANS.items():
            min_lat, max_lat, min_lon, max_lon = bbox
            if ocean_name == "Pacific Ocean":
                # Handle Pacific Ocean spanning 180° meridian
                if min_lat <= lat <= max_lat and (lon >= min_lon or lon <= max_lon):
                    return ocean_name
            else:
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    return ocean_name
        return "Unknown"
    
    def calculate_surface_temp(temp_data, pres_data):
        """Calculate mean surface temperature (0-20 dbar)"""
        try:
            if len(temp_data) == 0 or len(pres_data) == 0:
                return None
            
            # Convert to numpy arrays and ensure same length
            temp_arr = np.array(temp_data)
            pres_arr = np.array(pres_data)
            min_len = min(len(temp_arr), len(pres_arr))
            temp_arr = temp_arr[:min_len]
            pres_arr = pres_arr[:min_len]
            
            # Filter for surface measurements (0-20 dbar)
            surface_mask = (pres_arr >= 0) & (pres_arr <= 20) & np.isfinite(temp_arr) & np.isfinite(pres_arr)
            
            if np.any(surface_mask):
                surface_temps = temp_arr[surface_mask]
                return float(np.mean(surface_temps))
            else:
                return None
        except:
            return None
    
    def calculate_mixed_layer_depth(temp_data, pres_data, threshold=0.2):
        """Calculate mixed layer depth using temperature criterion"""
        try:
            if len(temp_data) < 3 or len(pres_data) < 3:
                return None
            
            # Convert to numpy arrays and ensure same length
            temp_arr = np.array(temp_data)
            pres_arr = np.array(pres_data)
            min_len = min(len(temp_arr), len(pres_arr))
            temp_arr = temp_arr[:min_len]
            pres_arr = pres_arr[:min_len]
            
            # Remove invalid data
            valid_mask = np.isfinite(temp_arr) & np.isfinite(pres_arr) & (pres_arr >= 0)
            if np.sum(valid_mask) < 3:
                return None
            
            temp_clean = temp_arr[valid_mask]
            pres_clean = pres_arr[valid_mask]
            
            # Sort by pressure
            sort_idx = np.argsort(pres_clean)
            temp_sorted = temp_clean[sort_idx]
            pres_sorted = pres_clean[sort_idx]
            
            # Calculate reference surface temperature (0-10 dbar)
            surface_mask = pres_sorted <= 10
            if np.sum(surface_mask) == 0:
                surface_temp = temp_sorted[0]
            else:
                surface_temp = np.mean(temp_sorted[surface_mask])
            
            # Find mixed layer depth
            for i, (temp, pres) in enumerate(zip(temp_sorted, pres_sorted)):
                if abs(temp - surface_temp) >= threshold:
                    return float(pres)
            
            return None
        except:
            return None
    
    def pressure_to_depth(pressure):
        """Convert pressure (dbar) to approximate depth (m)"""
        # Simple approximation: 1 dbar ≈ 1 meter depth
        return pressure
    
    def calculate_depth_stats(pres_data):
        """Calculate depth statistics from pressure data"""
        try:
            if len(pres_data) == 0:
                return None, None, None
            
            # Convert pressure to depth and filter valid values
            depths = [pressure_to_depth(p) for p in pres_data if pd.notna(p) and p >= 0]
            
            if len(depths) == 0:
                return None, None, None
            
            depths = np.array(depths)
            return float(np.min(depths)), float(np.max(depths)), float(np.mean(depths))
        except:
            return None, None, None

    # Source and destination directories
    source_dir = os.path.join("Data", "BCG floats", "raw csv")
    dest_dir = os.path.join("Data", "BCG floats", "final csv files")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"ERROR: Source directory not found: {source_dir}")
        print("Please convert NetCDF files to CSV first using option 3 or 5.")
        return
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Created destination directory: {dest_dir}")
    else:
        print(f"Destination directory exists: {dest_dir}")
    
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {dest_dir}")
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in source directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Processing counters
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Process each CSV file
    for i, csv_filename in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing: {csv_filename}")
        
        csv_path = os.path.join(source_dir, csv_filename)
        final_csv_path = os.path.join(dest_dir, csv_filename)
        
        # Check if final CSV already exists
        if os.path.exists(final_csv_path):
            print(f"  Skipping - Final CSV already exists: {csv_filename}")
            skipped_count += 1
            continue
        
        try:
            # Read the raw CSV
            df = pd.read_csv(csv_path)
            print(f"  Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            if len(df) == 0:
                print(f"  Warning: Empty CSV file: {csv_filename}")
                failed_count += 1
                continue
            
            # Process date columns
            if 'JULD' in df.columns:
                print("  Processing JULD column...")
                # Convert JULD to date only (remove time)
                df['JULD'] = pd.to_datetime(df['JULD'], errors='coerce').dt.date
            
            if 'JULD_LOCATION' in df.columns:
                print("  Processing JULD_LOCATION column...")
                # Extract time only from JULD_LOCATION
                df['JULD_LOCATION'] = pd.to_datetime(df['JULD_LOCATION'], errors='coerce').dt.time
            
            # Add n_levels column (count of valid pressure measurements per profile)
            if 'PRES' in df.columns and 'PLATFORM_NUMBER' in df.columns and 'CYCLE_NUMBER' in df.columns:
                print("    Computing n_levels...")
                
                groupby_cols = ['PLATFORM_NUMBER', 'CYCLE_NUMBER']
                if 'FLOAT_SERIAL_NO' in df.columns:
                    groupby_cols.append('FLOAT_SERIAL_NO')
                    print("      Including FLOAT_SERIAL_NO in n_levels grouping")
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
            
            # Identify grouping columns
            grouping_columns = []
            
            # Add platform number column (look for various possible names)
            platform_cols = [col for col in df.columns if 'platform' in col.lower()]
            if platform_cols:
                grouping_columns.append(platform_cols[0])
                print(f"  Using platform column: {platform_cols[0]}")
            
            # Add float serial number column if available
            if 'FLOAT_SERIAL_NO' in df.columns:
                grouping_columns.append('FLOAT_SERIAL_NO')
                print("  Using FLOAT_SERIAL_NO for grouping")
            
            # Add cycle column
            cycle_cols = [col for col in df.columns if 'cycle' in col.lower()]
            if cycle_cols:
                grouping_columns.append(cycle_cols[0])
                print(f"  Using cycle column: {cycle_cols[0]}")
            
            # Add n_levels column if available
            if 'n_levels' in df.columns:
                grouping_columns.append('n_levels')
                print("  Using n_levels for grouping")
            
            # Add date column (JULD)
            if 'JULD' in df.columns:
                grouping_columns.append('JULD')
                print("  Using JULD for date grouping")
            
            if not grouping_columns:
                print(f"  Warning: No suitable grouping columns found in {csv_filename}")
                failed_count += 1
                continue
            
            print(f"  Grouping by: {grouping_columns}")
            
            # Remove NaN values from grouping columns
            df_clean = df.dropna(subset=grouping_columns)
            print(f"  After removing NaN in grouping columns: {len(df_clean)} rows")
            
            if len(df_clean) == 0:
                print(f"  Warning: No valid data after cleaning: {csv_filename}")
                failed_count += 1
                continue
            
            # Group by platform, cycle, and date
            grouped = df_clean.groupby(grouping_columns)
            print(f"  Created {len(grouped)} groups")
            
            # Aggregate the data
            aggregated_rows = []
            
            for group_key, group_data in grouped:
                row = {}
                
                # Add grouping column values
                if len(grouping_columns) == 1:
                    row[grouping_columns[0]] = group_key
                else:
                    for i, col in enumerate(grouping_columns):
                        row[col] = group_key[i]
                
                # Calculate new oceanographic and geographical parameters
                print(f"    Processing group: {group_key}")
                
                # Get temperature and pressure data for calculations
                temp_data = group_data.get('TEMP', pd.Series()).dropna().values
                pres_data = group_data.get('PRES', pd.Series()).dropna().values
                
                # Calculate surface temperature (mean only)
                surface_temp = calculate_surface_temp(temp_data, pres_data)
                row['surface_temp_C'] = surface_temp
                
                # Calculate mixed layer depth
                mld = calculate_mixed_layer_depth(temp_data, pres_data)
                row['mixed_layer_depth_m'] = mld
                
                # Calculate depth statistics from pressure
                min_depth, max_depth, avg_depth = calculate_depth_stats(pres_data)
                row['min_DEPTH_m'] = min_depth
                row['max_DEPTH_m'] = max_depth
                row['avg_DEPTH_m'] = avg_depth
                
                # Get latitude and longitude for region/ocean assignment
                lat_data = group_data.get('LATITUDE', pd.Series()).dropna()
                lon_data = group_data.get('LONGITUDE', pd.Series()).dropna()
                
                if len(lat_data) > 0 and len(lon_data) > 0:
                    # Use first valid coordinates (assuming consistent location per profile)
                    lat = lat_data.iloc[0]
                    lon = lon_data.iloc[0]
                    
                    # Assign region and ocean
                    row['avg_REGION'] = assign_region(lat, lon)
                    row['ocean'] = assign_ocean(lat, lon)
                else:
                    row['avg_REGION'] = "Unknown"
                    row['ocean'] = "Unknown"
                
                # Process all other columns
                for col in df_clean.columns:
                    if col in grouping_columns:
                        continue  # Already handled above
                    
                    # Get non-null values for this column in this group
                    col_values = group_data[col].dropna()
                    
                    if len(col_values) == 0:
                        row[col] = None
                        continue
                    
                    # Clean byte strings from the data
                    cleaned_values = []
                    for val in col_values:
                        if isinstance(val, str) and val.startswith("b'") and val.endswith("'"):
                            # Remove b'...' format
                            cleaned_val = val[2:-1]
                            cleaned_values.append(cleaned_val)
                        elif isinstance(val, bytes):
                            # Decode actual byte objects
                            try:
                                cleaned_val = val.decode('utf-8')
                                cleaned_values.append(cleaned_val)
                            except:
                                cleaned_values.append(str(val))
                        else:
                            cleaned_values.append(val)
                    
                    col_values = pd.Series(cleaned_values)
                    
                    # Check if column contains numeric data
                    try:
                        # Try to convert to numeric
                        numeric_values = pd.to_numeric(col_values, errors='coerce').dropna()
                        
                        if len(numeric_values) > 0:
                            # Calculate mean for numeric columns
                            mean_value = numeric_values.mean()
                            
                            # Rename column if it contains measurement terms
                            final_col_name = col
                            if any(term in col.upper() for term in ['PRES', 'TEMP', 'PSAL', 'DOXY', 'CHLA', 'BBP']):
                                # Extract the measurement type
                                for term in ['PRES', 'TEMP', 'PSAL', 'DOXY', 'CHLA', 'BBP']:
                                    if term in col.upper():
                                        final_col_name = term
                                        break
                            
                            row[final_col_name] = mean_value
                        else:
                            # Non-numeric data - take the first non-null value
                            row[col] = col_values.iloc[0] if len(col_values) > 0 else None
                    
                    except:
                        # If conversion fails, treat as non-numeric
                        row[col] = col_values.iloc[0] if len(col_values) > 0 else None
                
                aggregated_rows.append(row)
            
            # Create final DataFrame
            final_df = pd.DataFrame(aggregated_rows)
            print(f"  Aggregated to {len(final_df)} rows with {len(final_df.columns)} columns")
            
            # Save to final CSV
            final_df.to_csv(final_csv_path, index=False)
            print(f"  ✓ Saved final CSV: {csv_filename}")
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {csv_filename}: {e}")
            failed_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total CSV files processed: {len(csv_files)}")
    print(f"Files successfully processed: {processed_count}")
    print(f"Files skipped (already exist): {skipped_count}")
    print(f"Failed processing: {failed_count}")
    print(f"Output directory: {dest_dir}")
    
    if processed_count > 0:
        print(f"\n✓ Successfully processed {processed_count} CSV files to final aggregated format")
    
    if failed_count > 0:
        print(f"\n⚠ {failed_count} files failed processing")

def main():
    """
    Main function to run the BCG parameters extraction
    """
    print("="*60)
    print("BCG PARAMETERS EXTRACTOR FOR ARGO BIO-PROFILE DATA")
    print("="*60)
    
    try:
        # Ensure all necessary directories exist
        ensure_directories_exist()
        
        # Get user input for functionality choice first
        print("\nChoose an option:")
        print("1. Extract unique float platform numbers from bio-profile index")
        print("2. Download NetCDF files for unique floats (aoml only)")
        print("3. Convert unique float NetCDF files to CSV format")
        print("4. Process raw CSV to final aggregated CSV files")
        print("5. Extract unique floats + Download NetCDF files")
        print("6. Extract + Download + Convert to CSV")
        print("7. Complete workflow: Extract → Download → Convert → Process to Final CSV")
        
        choice = input("Enter your choice (1-7): ").strip()
        
        # Only ask for year/month if needed (options 1, 5, 6, or 7)
        if choice in ['1', '5', '6', '7']:
            print("\n--- YEAR AND MONTH INPUT ---")
            year = int(input("Enter the year (e.g., 2025): "))
            
            print("\nAvailable months:")
            print("1-January, 2-February, 3-March, 4-April, 5-May, 6-June")
            print("7-July, 8-August, 9-September, 10-October, 11-November, 12-December")
            print("\nEnter months to process (examples):")
            print("- Single month: 8")
            print("- Multiple months: 8,9,10")
            print("- Range: 1-12")

            month_input = input("Enter month(s): ")

            # Parse month input
            months_to_process = []
            if '-' in month_input:
                # Range input (e.g., "1-12")
                start, end = map(int, month_input.split('-'))
                months_to_process = list(range(start, end + 1))
            elif ',' in month_input:
                # Multiple months (e.g., "8,9,10")
                months_to_process = [int(m.strip()) for m in month_input.split(',')]
            else:
                # Single month (e.g., "8")
                months_to_process = [int(month_input)]
            
            # Validate months
            for month in months_to_process:
                if month < 1 or month > 12:
                    raise ValueError(f"Month {month} must be between 1 and 12")
            
            print(f"\nProcessing year {year}, months: {[get_month_name(m) for m in months_to_process]}")
        
        if choice in ['1', '5', '6', '7']:
            # Extract platform numbers using the year/months provided above
            print("\n--- UNIQUE FLOAT EXTRACTION ---")
            print("="*60)
            
            # Step 1: Fetch data (once for all months)
            content = fetch_argo_bio_index()
            
            # Step 2: Parse data (once for all months)
            df = parse_argo_bio_data(content)
            
            if len(df) == 0:
                print("No data was successfully parsed")
                return
            
            # Step 3 & 4: Process each month
            total_months = len(months_to_process)
            for i, month in enumerate(months_to_process, 1):
                print(f"\n=== Processing {get_month_name(month)} {year} ({i}/{total_months}) ===")
                
                # Filter for Indian Ocean and target year/month
                filtered_df = filter_indian_ocean_data(df, year, month)
                
                # Save to CSV
                if len(filtered_df) > 0:
                    filepath = save_to_csv(filtered_df, year, month)
                    print(f"SUCCESS: {get_month_name(month)} data saved to {filepath}")
                else:
                    print(f"No unique floats found for Indian Ocean in {get_month_name(month)} {year}")
            
            print(f"\nCompleted processing {total_months} months for year {year}")
        
        if choice in ['2', '5', '6', '7']:
            # Download Sprof.nc files (aoml only)
            print("\n--- UNIQUE FLOAT NETCDF DOWNLOAD ---")
            download_sprof_files()
        
        if choice in ['3', '6', '7']:
            # Convert NetCDF files to CSV
            print("\n--- UNIQUE FLOAT NETCDF TO CSV CONVERSION ---")
            convert_netcdf_to_csv()
        
        if choice in ['4', '7']:
            # Process raw CSV to final aggregated CSV
            print("\n--- RAW CSV TO FINAL CSV PROCESSING ---")
            process_raw_csv_to_final()
            
        elif choice not in ['1', '2', '3', '4', '5', '6', '7']:
            print("Invalid choice. Please select 1-7.")
        
        print("="*60)
        print("UNIQUE FLOAT PROCESSING COMPLETED")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        print("BCG parameters processing failed")

if __name__ == "__main__":
    main()
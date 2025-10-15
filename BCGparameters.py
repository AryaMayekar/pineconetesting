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
            
            # Identify grouping columns
            grouping_columns = []
            
            # Add platform number column (look for various possible names)
            platform_cols = [col for col in df.columns if 'platform' in col.lower() or 'float' in col.lower()]
            if platform_cols:
                grouping_columns.append(platform_cols[0])
                print(f"  Using platform column: {platform_cols[0]}")
            
            # Add cycle column
            cycle_cols = [col for col in df.columns if 'cycle' in col.lower()]
            if cycle_cols:
                grouping_columns.append(cycle_cols[0])
                print(f"  Using cycle column: {cycle_cols[0]}")
            
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
            month = int(input("Enter the month (1-12): "))
            
            if month < 1 or month > 12:
                raise ValueError("Month must be between 1 and 12")
            
            print(f"\nProcessing {get_month_name(month)} {year}")
        
        if choice in ['1', '5', '6', '7']:
            # Extract platform numbers using the year/month provided above
            print("\n--- UNIQUE FLOAT EXTRACTION ---")
            print("="*60)
            
            # Step 1: Fetch data
            content = fetch_argo_bio_index()
            
            # Step 2: Parse data
            df = parse_argo_bio_data(content)
            
            if len(df) == 0:
                print("No data was successfully parsed")
                return
            
            # Step 3: Filter for Indian Ocean and target year/month
            filtered_df = filter_indian_ocean_data(df, year, month)
            
            # Step 4: Save to CSV
            if len(filtered_df) > 0:
                filepath = save_to_csv(filtered_df, year, month)
                print(f"\nSUCCESS: Unique float platform data saved to {filepath}")
            else:
                print(f"No unique floats found for Indian Ocean in {get_month_name(month)} {year}")
        
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
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import re
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('csv_processing.log'),
        logging.StreamHandler()
    ]
)

def clean_byte_strings(value):
    """
    Remove b' and b\" prefixes from strings.
    Converts bytes-like string representation to normal string.
    """
    if pd.isna(value):
        return value
    
    value_str = str(value).strip()
    
    # Remove b' or b" prefix and trailing quote
    if value_str.startswith("b'") and value_str.endswith("'"):
        return value_str[2:-1]
    elif value_str.startswith('b"') and value_str.endswith('"'):
        return value_str[2:-1]
    
    return value_str

def split_datetime_column(df, column_name):
    """
    Split datetime columns into DATE and TIME columns.
    Handles various datetime formats.
    """
    if column_name not in df.columns:
        return df
    
    try:
        # Convert to string and clean
        df[column_name] = df[column_name].astype(str)
        
        # Parse datetime
        # Handle formats like: 2025-04-15 07:34:37.000800256 or 15-04-2025 07:34:37 AM
        date_parts = []
        time_parts = []
        
        for val in df[column_name]:
            val = str(val).strip()
            
            if pd.isna(val) or val == 'nan' or val == '':
                date_parts.append(np.nan)
                time_parts.append(np.nan)
            else:
                # Try to split by space
                parts = val.split()
                
                if len(parts) >= 2:
                    date_parts.append(parts[0])
                    # Combine remaining parts as time (handles cases with AM/PM)
                    time_parts.append(' '.join(parts[1:]))
                elif len(parts) == 1:
                    # Might be just date or just time
                    if ':' in val:
                        date_parts.append(np.nan)
                        time_parts.append(val)
                    else:
                        date_parts.append(val)
                        time_parts.append(np.nan)
                else:
                    date_parts.append(np.nan)
                    time_parts.append(np.nan)
        
        # Create new columns with JULD prefix
        date_col_name = f'{column_name}_DATE'
        time_col_name = f'{column_name}_TIME'
        df[date_col_name] = date_parts
        df[time_col_name] = time_parts
        
        # Drop original column
        df = df.drop(columns=[column_name])
        
        logging.info(f"Split column '{column_name}' into '{date_col_name}' and '{time_col_name}'")
        
    except Exception as e:
        logging.warning(f"Could not split datetime column '{column_name}': {str(e)}")
    
    return df

def remove_null_columns(df):
    """
    Remove columns that are completely null/empty.
    """
    original_cols = len(df.columns)
    
    # Drop columns where all values are NaN, empty, or consist only of whitespace
    for col in df.columns:
        non_null = df[col].apply(lambda x: pd.notna(x) and str(x).strip() != '')
        if not non_null.any():
            df = df.drop(columns=[col])
    
    removed_cols = original_cols - len(df.columns)
    if removed_cols > 0:
        logging.info(f"Removed {removed_cols} completely null columns")
    
    return df

def process_csv_file(input_path, output_path):
    """
    Process a single CSV file:
    1. Remove completely null columns
    2. Clean byte string format (b'...')
    3. Split datetime columns
    4. Save cleaned CSV
    """
    try:
        logging.info(f"Processing: {input_path}")
        
        # Read CSV
        df = pd.read_csv(input_path, dtype=str)
        
        logging.info(f"  Original shape: {df.shape}")
        
        # Step 1: Remove completely null columns
        df = remove_null_columns(df)
        
        # Step 2: Clean byte string format in all columns
        for col in df.columns:
            df[col] = df[col].apply(clean_byte_strings)
        
        # Step 3: Split datetime columns
        datetime_columns = ['JULD', 'JULD_LOCATION']
        for col in datetime_columns:
            if col in df.columns:
                df = split_datetime_column(df, col)
        
        # Step 4: Remove duplicate columns (in case something went wrong)
        df = df.loc[:, ~df.columns.duplicated()]
        
        logging.info(f"  Final shape: {df.shape}")
        
        # Save processed CSV
        df.to_csv(output_path, index=False)
        logging.info(f"  Saved to: {output_path}")
        
        return True, len(df.columns), len(df)
        
    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")
        return False, 0, 0

def process_folder(input_folder, output_folder, folder_name):
    """
    Process all CSV files in a folder.
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"Processing {folder_name}")
    logging.info(f"Input: {input_folder}")
    logging.info(f"Output: {output_folder}")
    logging.info(f"{'='*80}\n")
    
    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all CSV files
    csv_files = list(Path(input_folder).rglob('*.csv'))
    logging.info(f"Found {len(csv_files)} CSV files")
    
    if len(csv_files) == 0:
        logging.warning(f"No CSV files found in {input_folder}")
        return
    
    successful = 0
    failed = 0
    total_cols = 0
    total_rows = 0
    
    for idx, csv_file in enumerate(csv_files, 1):
        # Create output file path maintaining subdirectory structure
        rel_path = csv_file.relative_to(input_folder)
        output_file = Path(output_folder) / rel_path
        
        # Create subdirectories if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        success, cols, rows = process_csv_file(str(csv_file), str(output_file))
        
        if success:
            successful += 1
            total_cols += cols
            total_rows += rows
        else:
            failed += 1
        
        # Progress indicator
        if idx % 10 == 0:
            logging.info(f"Progress: {idx}/{len(csv_files)} files processed")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Summary for {folder_name}:")
    logging.info(f"  Total files: {len(csv_files)}")
    logging.info(f"  Successful: {successful}")
    logging.info(f"  Failed: {failed}")
    logging.info(f"  Total rows processed: {total_rows}")
    logging.info(f"  Average columns per file: {total_cols // max(successful, 1)}")
    logging.info(f"{'='*80}\n")

def main():
    """
    Main function to process both BCG and Non-BCG float CSV files.
    """
    logging.info("Starting CSV Processing Script")
    logging.info(f"Execution time: {datetime.now()}")
    
    # BCG Floats
    bcg_raw_csv_folder = r"f:\pineconetesting\Data\BCG floats\raw csv"
    bcg_output_folder = r"f:\pineconetesting\Data\BCG floats\raw_supabase_upload_csvs"
    
    # Non-BCG Floats
    non_bcg_rough_csv_folder = r"f:\pineconetesting\Data\Non-BCG floats\rough_csv_Data"
    non_bcg_output_folder = r"f:\pineconetesting\Data\Non-BCG floats\raw_supabase_upload_csvs"
    
    # Process BCG floats
    if Path(bcg_raw_csv_folder).exists():
        process_folder(bcg_raw_csv_folder, bcg_output_folder, "BCG Floats - Raw CSV")
    else:
        logging.error(f"BCG raw CSV folder not found: {bcg_raw_csv_folder}")
    
    # Process Non-BCG floats
    if Path(non_bcg_rough_csv_folder).exists():
        process_folder(non_bcg_rough_csv_folder, non_bcg_output_folder, "Non-BCG Floats - Rough CSV Data")
    else:
        logging.error(f"Non-BCG rough CSV folder not found: {non_bcg_rough_csv_folder}")
    
    logging.info("CSV Processing Complete!")

if __name__ == "__main__":
    main()

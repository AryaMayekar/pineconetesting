#!/usr/bin/env python3
"""
ARGO Float Raw Data Upload System - PostgreSQL Local Storage
Uploads cleaned raw CSV data to local PostgreSQL database
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PostgreSQL imports
try:
    import psycopg2
    from psycopg2 import pool, extras
except ImportError:
    print("ERROR: psycopg2 package not found. Install with: pip install psycopg2-binary")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL connection"""
    host: str
    port: int
    database: str
    user: str
    password: str

class RawFloatDataUploader:
    """Handles batch upload of raw ARGO float data to PostgreSQL"""
    
    def __init__(self, config: PostgresConfig):
        """Initialize uploader with PostgreSQL configuration"""
        self.config = config
        
        # Create connection pool
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password,
            connect_timeout=5
        )
        
        # Statistics tracking
        self.stats = {
            'csv_files_processed': 0,
            'total_rows_uploaded': 0,
            'total_records_inserted': 0,
            'bgc_records': 0,
            'non_bgc_records': 0,
            'errors': 0,
            'batch_uploads': 0
        }
        
        # Batch configuration
        self.batch_size = 30000  # Process 10000 records per batch (local storage, no network latency)
    
    def get_connection(self):
        """Get a connection from the pool"""
        return self.connection_pool.getconn()
    
    def put_connection(self, conn):
        """Return connection to the pool"""
        self.connection_pool.putconn(conn)
    
    def create_raw_tables(self) -> bool:
        """
        Create raw float tables if they don't exist
        
        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            logger.info("Creating raw data tables...")
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Create raw_float_metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_float_metadata (
                    float_id TEXT PRIMARY KEY,
                    float_type TEXT NOT NULL CHECK (float_type IN ('BCG', 'NON-BCG')),
                    profiler_type TEXT,
                    institution TEXT,
                    project_name TEXT,
                    wmo_inst_type TEXT,
                    platform_type TEXT,
                    all_columns TEXT[],
                    csv_filename TEXT,
                    last_reading_date DATE
                )
            """)
            
            # Create raw_float_parameters_reading table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_float_parameters_reading (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    float_id TEXT NOT NULL,
                    parameter_name TEXT NOT NULL,
                    value TEXT,
                    JULD_DATE DATE,
                    JULD_TIME TIME,
                    FOREIGN KEY (float_id) REFERENCES raw_float_metadata(float_id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for raw_float_metadata
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_float_metadata_float_id 
                ON raw_float_metadata (float_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_float_metadata_float_type 
                ON raw_float_metadata (float_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_float_metadata_institution 
                ON raw_float_metadata (institution)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_float_metadata_profiler_type 
                ON raw_float_metadata (profiler_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_float_metadata_last_reading_date 
                ON raw_float_metadata (last_reading_date)
            """)
            
            # Create indexes for raw_float_parameters_reading
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_float_parameters_reading_float_id 
                ON raw_float_parameters_reading (float_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_float_parameters_reading_juld_date 
                ON raw_float_parameters_reading (JULD_DATE)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_float_parameters_reading_parameter 
                ON raw_float_parameters_reading (parameter_name)
            """)
            
            conn.commit()
            logger.info("✓ Raw data tables and indexes created successfully")
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.put_connection(conn)
    
    def verify_raw_tables_exist(self) -> bool:
        """
        Verify that both raw tables exist: raw_float_metadata and raw_float_parameters_reading
        
        Returns:
            True if both tables exist, False otherwise
        """
        conn = None
        try:
            logger.info("Verifying raw data tables...")
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("""
                SELECT EXISTS(
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'raw_float_metadata'
                )
            """)
            metadata_exists = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT EXISTS(
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'raw_float_parameters_reading'
                )
            """)
            parameters_exists = cursor.fetchone()[0]
            
            cursor.close()
            
            if metadata_exists:
                logger.info("✓ raw_float_metadata table exists and is accessible")
            else:
                logger.warning("⚠ raw_float_metadata table does not exist, creating...")
                return self.create_raw_tables()
            
            if parameters_exists:
                logger.info("✓ raw_float_parameters_reading table exists and is accessible")
            else:
                logger.warning("⚠ raw_float_parameters_reading table does not exist, creating...")
                return self.create_raw_tables()
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying tables: {str(e)}")
            logger.info("Attempting to create tables...")
            return self.create_raw_tables()
        finally:
            if conn:
                self.put_connection(conn)
    
    def determine_float_type(self, csv_path: str) -> str:
        """
        Determine if float is BCG or NON-BCG based on file path
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            'BCG' or 'NON-BCG'
        """
        if "BCG" in csv_path and "Non-BCG" not in csv_path:
            return "BCG"
        else:
            return "NON-BCG"
    
    def process_csv_batch(self, df: pd.DataFrame, float_type: str, csv_filename: str, 
                          float_id: str) -> tuple:
        """
        Process rows from CSV and convert to parameter-based records for upload
        Each row becomes multiple parameter records (one per column value)
        
        Args:
            df: DataFrame with rows to process
            float_type: 'BCG' or 'NON-BCG'
            csv_filename: Name of the CSV file
            float_id: Platform number/Float ID
            
        Returns:
            Tuple of (records, additional_records)
        """
        records = []
        
        # Extract date and time columns if available
        date_column = None
        time_column = None
        
        if 'JULD_DATE' in df.columns:
            date_column = 'JULD_DATE'
            time_column = 'JULD_TIME'
        elif 'JULD_LOCATION_DATE' in df.columns:
            date_column = 'JULD_LOCATION_DATE'
            time_column = 'JULD_LOCATION_TIME'
        
        # Fast vectorized processing - convert to dict records
        dict_records = df.to_dict('records')
        
        for row in dict_records:
            # Get date for this row if available
            date_val = None
            if date_column and date_column in row:
                date_str = str(row[date_column]).strip()
                if date_str and date_str not in ['nan', 'NaN', '', 'None']:
                    date_val = date_str[:10]  # Get YYYY-MM-DD part
            
            # Get time for this row if available (already in 24-hour format in CSV)
            time_val = None
            if time_column and time_column in row:
                time_str = str(row[time_column]).strip()
                if time_str and time_str not in ['nan', 'NaN', '', 'None']:
                    # Time is already in 24-hour format: HH:MM:SS.fractional
                    # Extract just the HH:MM:SS part (first 8 characters)
                    time_val = time_str[:8]
            
            # Create parameter records for each non-null column value
            for col_name, value in row.items():
                # Skip null values
                if pd.isna(value):
                    continue
                
                # Convert to string and clean
                value_str = str(value).strip()
                if value_str in ['nan', 'NaN', '', 'None']:
                    continue
                
                # Create a parameter record for this value (raw string only)
                param_record = (
                    float_id,
                    col_name,
                    value_str,
                    date_val,
                    time_val
                )
                records.append(param_record)
        
        return records, []
    
    def batch_insert_records(self, records: List[tuple], additional_records: List[Dict], 
                            table_name: str) -> bool:
        """
        Insert records in batches using PostgreSQL
        
        Args:
            records: List of tuples to insert
            additional_records: List of additional records (not used for raw tables)
            table_name: Name of the main table to insert into
            
        Returns:
            True if successful, False otherwise
        """
        if not records:
            return True
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Process records in batches
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i + self.batch_size]
                
                try:
                    # Insert batch using executemany
                    if table_name == 'raw_float_parameters_reading':
                        query = """
                            INSERT INTO raw_float_parameters_reading 
                            (float_id, parameter_name, value, JULD_DATE, JULD_TIME)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """
                    
                    extras.execute_batch(cursor, query, batch, page_size=5000)
                    conn.commit()
                    
                    self.stats['batch_uploads'] += 1
                    self.stats['total_records_inserted'] += len(batch)
                    
                    logger.info(f"  ✓ Inserted batch {self.stats['batch_uploads']} ({len(batch)} records)")
                    
                except Exception as batch_error:
                    logger.error(f"  ⚠ Batch insert failed: {str(batch_error)}")
                    conn.rollback()
                    self.stats['errors'] += 1
                    continue
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Batch insert failed for {table_name}: {str(e)}")
            self.stats['errors'] += 1
            return False
        finally:
            if conn:
                self.put_connection(conn)
    
    def process_csv_file(self, csv_path: str) -> bool:
        """
        Process a single CSV file and upload to PostgreSQL
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            logger.info(f"Processing: {Path(csv_path).name}")
            
            # Read CSV
            df = pd.read_csv(csv_path, dtype=str)
            
            if df.empty:
                logger.warning(f"  ⚠ Empty CSV file: {csv_path}")
                return True
            
            # Get float type and ID
            float_type = self.determine_float_type(csv_path)
            csv_filename = Path(csv_path).name
            
            # Get float ID from PLATFORM_NUMBER if available
            if 'PLATFORM_NUMBER' in df.columns:
                float_id = df['PLATFORM_NUMBER'].iloc[0]
                if pd.isna(float_id):
                    float_id = csv_filename.replace('_Sprof.csv', '')
            else:
                float_id = csv_filename.replace('_Sprof.csv', '')
            
            float_id = str(float_id).strip()
            
            logger.info(f"  Float ID: {float_id} | Type: {float_type} | Rows: {len(df)}")
            
            # Extract metadata from first row of CSV
            first_row = df.iloc[0] if len(df) > 0 else {}
            
            # Helper function to safely get value from DataFrame
            def get_value(col_name):
                if col_name in df.columns:
                    val = first_row[col_name]
                    return str(val).strip() if pd.notna(val) and val not in ['nan', 'NaN', '', 'None'] else None
                return None
            
            # Extract last_reading_date from CSV (use JULD_DATE if available, otherwise JULD_LOCATION_DATE)
            last_reading_date = None
            if 'JULD_DATE' in df.columns:
                # Get the maximum date from the JULD_DATE column
                valid_dates = df['JULD_DATE'].dropna()
                if len(valid_dates) > 0:
                    date_strs = [str(d).strip()[:10] for d in valid_dates if str(d).strip() not in ['nan', 'NaN', '', 'None']]
                    if date_strs:
                        last_reading_date = max(date_strs)
            
            if not last_reading_date and 'JULD_LOCATION_DATE' in df.columns:
                valid_dates = df['JULD_LOCATION_DATE'].dropna()
                if len(valid_dates) > 0:
                    date_strs = [str(d).strip()[:10] for d in valid_dates if str(d).strip() not in ['nan', 'NaN', '', 'None']]
                    if date_strs:
                        last_reading_date = max(date_strs)
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Insert or update metadata record
            try:
                # Prepare metadata columns as JSONB
                all_columns = list(df.columns)
                
                cursor.execute("""
                    INSERT INTO raw_float_metadata 
                    (float_id, float_type, profiler_type, institution, project_name, 
                     wmo_inst_type, platform_type, csv_filename, last_reading_date, all_columns)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (float_id) DO UPDATE SET
                        last_reading_date = EXCLUDED.last_reading_date,
                        all_columns = EXCLUDED.all_columns
                """, (
                    float_id,
                    float_type,
                    get_value('WMO_INST_TYPE'),
                    get_value('DATA_CENTRE'),
                    get_value('PROJECT_NAME'),
                    get_value('WMO_INST_TYPE'),
                    get_value('PLATFORM_TYPE'),
                    csv_filename,
                    last_reading_date,
                    all_columns
                ))
                conn.commit()
                logger.info(f"  ✓ Metadata inserted/updated for {float_id}")
            except Exception as e:
                logger.error(f"  ❌ Failed to insert metadata: {str(e)}")
                conn.rollback()
                self.stats['errors'] += 1
            finally:
                cursor.close()
            
            # Process entire CSV into parameter records
            all_records, _ = self.process_csv_batch(df, float_type, csv_filename, float_id)
            
            # Upload all records in optimized batches
            if all_records:
                logger.info(f"  📤 Uploading {len(all_records)} parameter records in batches...")
                
                # Insert into raw_float_parameters_reading table
                success = self.batch_insert_records(all_records, [], 'raw_float_parameters_reading')
                
                if success:
                    self.stats['total_rows_uploaded'] += len(all_records)
                    if float_type == "BCG":
                        self.stats['bgc_records'] += len(all_records)
                    else:
                        self.stats['non_bgc_records'] += len(all_records)
                    
                    logger.info(f"  ✅ Successfully uploaded {len(all_records)} parameter records")
                    return True
                else:
                    return False
            else:
                logger.warning(f"  ⚠ No valid records found in {csv_filename}")
                return True
            
        except Exception as e:
            logger.error(f"Error processing {csv_path}: {str(e)}")
            self.stats['errors'] += 1
            return False
        finally:
            if conn:
                self.put_connection(conn)
    
    def process_directory(self, directory_path: str) -> None:
        """
        Process all CSV files in a directory recursively
        
        Args:
            directory_path: Path to directory containing CSV files
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return
        
        # Find all CSV files recursively
        csv_files = list(directory.rglob("*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {directory_path}")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing directory: {directory_path}")
        logger.info(f"Found {len(csv_files)} CSV files")
        logger.info(f"{'='*80}\n")
        
        successful = 0
        failed = 0
        
        for idx, csv_file in enumerate(csv_files, 1):
            if self.process_csv_file(str(csv_file)):
                successful += 1
            else:
                failed += 1
            
            self.stats['csv_files_processed'] += 1
            
            # Progress indicator
            if idx % 10 == 0:
                logger.info(f"Progress: {idx}/{len(csv_files)} files processed\n")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Directory Processing Summary:")
        logger.info(f"  Successful: {successful}/{len(csv_files)}")
        logger.info(f"  Failed: {failed}/{len(csv_files)}")
        logger.info(f"{'='*80}\n")
    
    def upload_all_raw_csvs(self, base_paths: List[str]) -> Dict:
        """
        Upload all raw CSV files from multiple directories
        
        Args:
            base_paths: List of directory paths containing raw CSV files
            
        Returns:
            Dictionary with upload statistics
        """
        logger.info("🚀 Starting RAW CSV Upload to PostgreSQL (Batch Processing)")
        logger.info(f"Batch size: {self.batch_size} records per batch")
        
        # Create tables if needed and verify they exist
        if not self.verify_raw_tables_exist():
            logger.error("Cannot proceed without tables. Failed to create tables.")
            return self.stats
        
        # Process each directory
        for base_path in base_paths:
            self.process_directory(base_path)
        
        # Print final statistics
        logger.info(f"\n{'='*80}")
        logger.info("⚡ RAW CSV BATCH UPLOAD TO POSTGRESQL COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"CSV files processed: {self.stats['csv_files_processed']}")
        logger.info(f"Total rows uploaded: {self.stats['total_rows_uploaded']}")
        logger.info(f"  • BCG float records: {self.stats['bgc_records']}")
        logger.info(f"  • Non-BCG float records: {self.stats['non_bgc_records']}")
        logger.info(f"Total records inserted in raw_float_parameters_reading: {self.stats['total_records_inserted']}")
        logger.info(f"Batch uploads performed: {self.stats['batch_uploads']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"{'='*80}\n")
        
        return self.stats
    
    def close(self):
        """Close all connections in the pool"""
        self.connection_pool.closeall()

def main():
    """Main execution function"""
    
    # Load PostgreSQL configuration from environment variables
    PG_HOST = os.getenv("PG_HOST", "localhost")
    PG_PORT = int(os.getenv("PG_PORT", "5432"))
    PG_DATABASE = os.getenv("PG_DATABASE", "float_data")
    PG_USER = os.getenv("PG_USER", "postgres")
    PG_PASSWORD = os.getenv("PG_PASSWORD", "")
    
    # Directory paths containing raw CSV files
    BASE_PATHS = [
        r"f:\pineconetesting\Data\BCG floats\raw_supabase_upload_csvs",
        r"f:\pineconetesting\Data\Non-BCG floats\raw_supabase_upload_csvs"
    ]
    
    try:
        # Validate configuration
        if not PG_PASSWORD:
            logger.error("ERROR: PostgreSQL password not found!")
            print("\nConfiguration Setup Required:")
            print("1. Make sure .env file exists with:")
            print("   PG_HOST=localhost")
            print("   PG_PORT=5432")
            print("   PG_DATABASE=float_data")
            print("   PG_USER=postgres")
            print("   PG_PASSWORD=your_password_here")
            print("2. Run the script again")
            return
        
        # Create configuration
        config = PostgresConfig(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        
        # Initialize uploader
        uploader = RawFloatDataUploader(config)
        
        # Test connection
        logger.info("Testing PostgreSQL connection...")
        try:
            conn = uploader.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            uploader.put_connection(conn)
            logger.info("✓ PostgreSQL connection successful!")
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            raise
        
        # Upload all raw CSV files (tables will be created automatically if needed)
        stats = uploader.upload_all_raw_csvs(BASE_PATHS)
        
        if stats['total_rows_uploaded'] > 0:
            logger.info("✅ SUCCESS: Raw CSV upload to PostgreSQL completed!")
            print("\n🎯 Upload Results Summary:")
            print(f"   • {stats['csv_files_processed']} CSV files processed")
            print(f"   • {stats['total_rows_uploaded']} total rows uploaded")
            print(f"     - BCG float records: {stats['bgc_records']}")
            print(f"     - Non-BCG float records: {stats['non_bgc_records']}")
            print(f"   • {stats['total_records_inserted']} records inserted in raw_float_parameters table")
            print(f"   • {stats['batch_uploads']} batch uploads performed")
            print(f"\n📊 Next Steps:")
            print(f"   1. Verify data in PostgreSQL: SELECT COUNT(*) FROM raw_float_parameters_reading;")
            print(f"   2. Query by float_id: SELECT * FROM raw_float_parameters_reading WHERE float_id = '...'")
            print(f"   3. Monitor query performance with indexed columns")
        else:
            logger.warning("⚠ WARNING: No rows were uploaded. Check if CSV files exist and contain data.")
        
        # Close all connections
        uploader.close()
    
    except Exception as e:
        logger.error(f"FATAL ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()

-- Create raw_float_metadata table
-- Stores metadata about each float (raw data version)
CREATE TABLE IF NOT EXISTS raw_float_metadata (
    float_id TEXT PRIMARY KEY,
    float_type TEXT NOT NULL CHECK (float_type IN ('BCG', 'NON-BCG')),
    profiler_type TEXT,
    institution TEXT,
    project_name TEXT,
    wmo_inst_type TEXT,
    platform_type TEXT,
    all_columns JSONB,
    csv_filename TEXT,
    last_reading_date DATE
);

-- Create raw_float_parameters_reading table
-- Stores individual parameter readings (raw data version)
-- NO UNIQUE constraint - allows multiple readings per parameter per date/time
-- Stores raw string values only
CREATE TABLE IF NOT EXISTS raw_float_parameters_reading (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    float_id TEXT NOT NULL,
    parameter_name TEXT NOT NULL,
    value TEXT,
    JULD_DATE DATE,
    JULD_TIME TIME,
    FOREIGN KEY (float_id) REFERENCES raw_float_metadata(float_id) ON DELETE CASCADE
);
    
-- Core metadata columns

-- Create indexes for raw_float_metadata
CREATE INDEX IF NOT EXISTS idx_raw_float_metadata_float_id ON raw_float_metadata (float_id);
CREATE INDEX IF NOT EXISTS idx_raw_float_metadata_float_type ON raw_float_metadata (float_type);
CREATE INDEX IF NOT EXISTS idx_raw_float_metadata_institution ON raw_float_metadata (institution);
CREATE INDEX IF NOT EXISTS idx_raw_float_metadata_profiler_type ON raw_float_metadata (profiler_type);
CREATE INDEX IF NOT EXISTS idx_raw_float_metadata_last_reading_date ON raw_float_metadata (last_reading_date);
CREATE INDEX IF NOT EXISTS idx_raw_float_metadata_columns ON raw_float_metadata USING GIN (all_columns);

-- Create indexes for raw_float_parameters_reading
CREATE INDEX IF NOT EXISTS idx_raw_float_parameters_reading_float_id ON raw_float_parameters_reading (float_id);
CREATE INDEX IF NOT EXISTS idx_raw_float_parameters_reading_juld_date ON raw_float_parameters_reading (JULD_DATE);
CREATE INDEX IF NOT EXISTS idx_raw_float_parameters_reading_parameter ON raw_float_parameters_reading (parameter_name);

-- ============================================================================
-- DELETE/DROP EVERYTHING - Uncomment below to delete all tables and indexes
-- ============================================================================
DROP TABLE IF EXISTS raw_float_parameters_reading CASCADE;
DROP TABLE IF EXISTS raw_float_metadata CASCADE;



-- Create database
CREATE DATABASE argo_float_data;
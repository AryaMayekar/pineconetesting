# Pinecone Index Structure & Data Storage Summary

## Overview
This document describes the complete structure of data and metadata stored in the **`argo-ocean-data`** Pinecone index for Argo float oceanographic data.

---

## Index Configuration

| Property | Value |
|----------|-------|
| **Index Name** | `argo-ocean-data` |
| **Embedding Model** | `all-MiniLM-L6-v2` |
| **Embedding Dimensions** | 384 |
| **Similarity Metric** | Cosine |
| **Cloud** | AWS |
| **Region** | us-east-1 |
| **Batch Size** | 100 vectors per upsert |

---

## Record Structure

Each record in the Pinecone index contains three components:

### 1. **Vector ID**
```
Format: argo-{12-character-hex-uuid}
Example: argo-a1b2c3d4e5f6
```
- Uniquely identifies each record
- Generated using UUID v4 (first 12 hex characters)

### 2. **Embedding Vector (384 dimensions)**
- Generated using `all-MiniLM-L6-v2` model
- Created from combined text of three summaries
- Used for semantic similarity search

### 3. **Metadata** (50+ fields)
- Structured key-value pairs
- All queryable through Pinecone metadata filters
- Includes summaries and raw measurements

---

## Complete Metadata Field Reference

### **A. Source & Float Identification**

| Field | Type | Source | Example |
|-------|------|--------|---------|
| `float_type` | String | CSV | `BCG` or `Non-BCG` |
| `source_file` | String | Filename | `1902367_Sprof.csv` |
| `platform_number` | String | PLATFORM_NUMBER | `1902367` |
| `float_serial_no` | String | FLOAT_SERIAL_NO | `KR456` |

---

### **B. Geographic & Spatial Data**

| Field | Type | Source | Unit | Notes |
|-------|------|--------|------|-------|
| `latitude` | Float | LATITUDE | Decimal degrees | ±90 |
| `longitude` | Float | LONGITUDE | Decimal degrees | ±180 |
| `geohash` | String | Calculated | - | Precision 6 (~1.2km) |
| `avg_region` | String | avg_REGION | - | Geographic region name |
| `ocean` | String | ocean | - | Ocean basin (Indian, Pacific, Atlantic, etc.) |

**Geohashing Details:**
- Precision level: 6
- Spatial resolution: ~1.2 km
- Enables geographic proximity queries

---

### **C. Depth Information**

| Field | Type | Source | Unit | Purpose |
|-------|------|--------|------|---------|
| `min_depth_m` | Float | min_DEPTH_m | meters | Minimum sampling depth |
| `max_depth_m` | Float | max_DEPTH_m | meters | Maximum sampling depth |
| `avg_depth_m` | Float | avg_DEPTH_m | meters | Average sampling depth |
| `n_levels` | Integer | n_levels | count | Number of vertical levels sampled |

---

### **D. Pressure (PRES) - Raw & Adjusted**

#### Raw Pressure
| Field | Type | Unit |
|-------|------|------|
| `pres_raw_min` | Float | dbar (decibar) |
| `pres_raw_max` | Float | dbar |
| `pres_min` | Float | dbar (alt naming) |
| `pres_max` | Float | dbar (alt naming) |

#### Adjusted Pressure
| Field | Type | Unit | Purpose |
|-------|------|------|---------|
| `pres_adjusted_dbar` | Float | dbar | Corrected pressure value |
| `pres_adjusted_error` | Float | dbar | Overall error estimate |
| `pres_adjusted_error_min` | Float | dbar | Minimum error |
| `pres_adjusted_error_max` | Float | dbar | Maximum error |
| `pres_adjusted_error_mean` | Float | dbar | Mean error |
| `pres_adjusted_qc` | String | flag | Quality control status |

---

### **E. Temperature (TEMP) - Raw & Adjusted**

#### Raw Temperature
| Field | Type | Unit |
|-------|------|------|
| `temp_raw_min` | Float | °C |
| `temp_raw_max` | Float | °C |
| `temp_min` | Float | °C (alt naming) |
| `temp_max` | Float | °C (alt naming) |

#### Adjusted Temperature
| Field | Type | Unit | Purpose |
|-------|------|------|---------|
| `temp_adjusted_c` | Float | °C | Corrected temperature |
| `temp_adjusted_error` | Float | °C | Overall error estimate |
| `temp_adjusted_error_min` | Float | °C | Minimum error |
| `temp_adjusted_error_max` | Float | °C | Maximum error |
| `temp_adjusted_error_mean` | Float | °C | Mean error |
| `temp_adjusted_qc` | String | flag | Quality control status |

---

### **F. Salinity (PSAL) - Raw & Adjusted**

#### Raw Salinity
| Field | Type | Unit |
|-------|------|------|
| `psal_raw_min` | Float | PSU (Practical Salinity Units) |
| `psal_raw_max` | Float | PSU |
| `psal_min` | Float | PSU (alt naming) |
| `psal_max` | Float | PSU (alt naming) |

#### Adjusted Salinity
| Field | Type | Unit | Purpose |
|-------|------|------|---------|
| `psal_adjusted_psu` | Float | PSU | Corrected salinity |
| `psal_adjusted_error` | Float | PSU | Overall error estimate |
| `psal_adjusted_error_min` | Float | PSU | Minimum error |
| `psal_adjusted_error_max` | Float | PSU | Maximum error |
| `psal_adjusted_error_mean` | Float | PSU | Mean error |
| `psal_adjusted_qc` | String | flag | Quality control status |

---

### **G. Biogeochemistry (BGC) Parameters**

#### Colored Dissolved Organic Matter (CDOM)
| Field | Type | Unit |
|-------|------|------|
| `cdom_adjusted` | Float | ppb or m⁻¹ |
| `cdom_adjusted_error` | Float | ppb or m⁻¹ |
| `cdom_adjusted_qc` | String | flag |

#### Down-Welling Irradiance (by wavelength)
| Field | Type | Unit | Wavelength |
|-------|------|------|------------|
| `irr380_adjusted` | Float | µE/m²/s | 380 nm (UV-A) |
| `irr443_adjusted` | Float | µE/m²/s | 443 nm (blue) |
| `irr490_adjusted` | Float | µE/m²/s | 490 nm (blue-green) |
| `irr555_adjusted` | Float | µE/m²/s | 555 nm (green) |

---

### **H. Temporal Information**

| Field | Type | Format | Source |
|-------|------|--------|--------|
| `measurement_date` | String | YYYY-MM-DD | DATE or JULD |
| `juld_location_time` | String | YYYY-MM-DD HH:MM:SS | JULD_LOCATION_time |
| `juld_location_timestamp` | String | YYYY-MM-DD HH:MM:SS | JULD_LOCATION |
| `reference_date_time` | String | YYYY-MM-DD | REFERENCE_DATE_TIME |
| `date_creation` | String | YYYY-MM-DD | DATE_CREATION |
| `date_update` | String | YYYY-MM-DD | DATE_UPDATE |
| `cycle_number` | Integer | N/A | CYCLE_NUMBER |

**Date Conversion Rules:**
- JULD (Julian Day) format: Converted from 1950-01-01 epoch
- All dates standardized to ISO 8601 format
- Times included where available

---

### **I. Data Quality & Administrative**

| Field | Type | Source | Values |
|-------|------|--------|--------|
| `data_type` | String | DATA_TYPE | Argo format string |
| `data_mode` | String | DATA_MODE | `R` (Real-time) or `D` (Delayed) |
| `data_centre` | String | DATA_CENTRE | Center code |
| `format_version` | String | FORMAT_VERSION | Version number |
| `handbook_version` | String | HANDBOOK_VERSION | Handbook reference |

---

### **J. Profile-Level Quality Control Flags**

All stored as Strings. QC codes typically follow Argo standards:
- `0` = No QC performed
- `1` = Good data
- `2` = Probably good data
- `3` = Probably bad data
- `4` = Bad data
- `9` = Missing data

| Flag | Field Name |
|------|------------|
| Pressure | `PROFILE_PRES_QC` |
| Temperature | `PROFILE_TEMP_QC` |
| Salinity | `PROFILE_PSAL_QC` |
| Density | `PROFILE_DENSITY_QC` |
| Salinity (alt) | `PROFILE_SALINITY_QC` |
| Dissolved Oxygen | `PROFILE_DOXY_QC` |
| Nitrate | `PROFILE_NITRATE_QC` |
| Chlorophyll-a | `PROFILE_CHLA_QC` |
| CDOM | `PROFILE_CDOM_QC` |
| Backscatter 700nm | `PROFILE_BBP700_QC` |
| pH (in-situ total) | `PROFILE_PH_IN_SITU_TOTAL_QC` |
| Chlorophyll Fluorescence | `PROFILE_CHLA_FLUORESCENCE_QC` |
| Irradiance 380nm | `PROFILE_DOWN_IRRADIANCE380_QC` |
| Irradiance 443nm | `PROFILE_DOWN_IRRADIANCE443_QC` |
| Irradiance 490nm | `PROFILE_DOWN_IRRADIANCE490_QC` |
| Irradiance 555nm | `PROFILE_DOWN_IRRADIANCE555_QC` |

---

### **K. Text Summaries (Searchable)**

These fields contain human-readable summaries and are used for semantic search:

#### `temporal_summary` (String)
Contains:
- Measurement date
- Surface temperature (average)
- Mixed layer depth

**Example:**
```
Measurement recorded on 2023-06-15 | Surface temperature (avg): 18.5°C | Mixed layer depth: 120.0m
```

#### `location_summary` (String)
Contains:
- Geographic region
- Ocean basin
- Latitude and longitude
- Depth range

**Example:**
```
Region: Southern Ocean | Ocean: Indian Ocean | Latitude: -45.2340° | Longitude: 135.6780° | Depth range: 5.20m to 2000.00m
```

#### `contextual_summary` (String)
Contains:
- Core parameters (Platform, Type, Cycle, Pressure, Temperature, Salinity)
- Biogeochemistry (Dissolved Oxygen, Nitrate, Chlorophyll, CDOM, Irradiance)
- QC flags
- Raw parameter ranges
- Mission & technical metadata

**Example (truncated):**
```
CORE PARAMETERS: Platform: 1902367 | Type: APEX | Cycle: 42 | PRES=2000.00 | PRES_ADJUSTED=2000.15 | TEMP=12.45 | TEMP_ADJUSTED=12.47 | PSAL=34.56 | PSAL_ADJUSTED=34.57 | BIOGEOCHEMISTRY: Dissolved oxygen: 250.30 µmol/kg | Nitrate: 45.20 µmol/kg | Chlorophyll a: 0.250 mg/m³ | QC FLAGS: PROFILE_PRES_QC=1 | PROFILE_TEMP_QC=1 | PROFILE_PSAL_QC=1 | ...
```

---

## Data Processing Pipeline

```
┌─────────────────────────────────┐
│  BCG & Non-BCG CSV Files        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  For Each Row:                  │
│  - Extract all columns          │
│  - Convert dates (JULD format)  │
│  - Generate 3 summaries         │
│  - Compute geohash              │
│  - Extract metadata fields      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Combine Summaries:             │
│  temporal | location | contextual
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Generate 384-dim Embeddings    │
│  (all-MiniLM-L6-v2 model)       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Create Record:                 │
│  (ID, Embedding, Metadata)      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Batch 100 Records              │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Upsert to Pinecone Index       │
└─────────────────────────────────┘
```

---

## Complete Record Example

```json
{
  "id": "argo-a1b2c3d4e5f6",
  "embedding": [
    0.1234, 0.5678, 0.9012, ..., 0.3456
  ],
  "metadata": {
    "float_type": "BCG",
    "source_file": "1902367_Sprof.csv",
    "platform_number": "1902367",
    "float_serial_no": "KR456",
    "latitude": -45.234,
    "longitude": 135.678,
    "geohash": "re3wp1",
    "ocean": "Indian Ocean",
    "avg_region": "Southern Ocean",
    "min_depth_m": 5.2,
    "max_depth_m": 2000.0,
    "avg_depth_m": 1200.5,
    "n_levels": 256,
    "pres_raw_min": 5.2,
    "pres_raw_max": 2000.5,
    "pres_adjusted_dbar": 2000.15,
    "pres_adjusted_error": 2.5,
    "pres_adjusted_qc": "1",
    "temp_raw_min": 2.34,
    "temp_raw_max": 28.56,
    "temp_adjusted_c": 12.45,
    "temp_adjusted_error": 0.12,
    "temp_adjusted_qc": "1",
    "psal_raw_min": 33.45,
    "psal_raw_max": 35.67,
    "psal_adjusted_psu": 34.56,
    "psal_adjusted_error": 0.08,
    "psal_adjusted_qc": "1",
    "cdom_adjusted": 1.23,
    "cdom_adjusted_error": 0.05,
    "irr380_adjusted": 0.123,
    "irr443_adjusted": 0.456,
    "irr490_adjusted": 0.789,
    "irr555_adjusted": 0.321,
    "data_type": "Argo float vertical profile",
    "data_mode": "R",
    "data_centre": "GDAC",
    "format_version": "3.1",
    "measurement_date": "2023-06-15",
    "juld_location_time": "2023-06-15 14:30:45",
    "reference_date_time": "2023-06-01",
    "cycle_number": 42,
    "PROFILE_PRES_QC": "1",
    "PROFILE_TEMP_QC": "1",
    "PROFILE_PSAL_QC": "1",
    "PROFILE_DOXY_QC": "1",
    "PROFILE_NITRATE_QC": "1",
    "PROFILE_CHLA_QC": "1",
    "PROFILE_CDOM_QC": "1",
    "temporal_summary": "Measurement recorded on 2023-06-15 | Surface temperature (avg): 18.5°C | Mixed layer depth: 120.0m",
    "location_summary": "Region: Southern Ocean | Ocean: Indian Ocean | Latitude: -45.2340° | Longitude: 135.6780° | Depth range: 5.20m to 2000.00m",
    "contextual_summary": "CORE PARAMETERS: Platform: 1902367 | Type: APEX | Cycle: 42 | PRES=2000.00 | PRES_ADJUSTED=2000.15 | TEMP=12.45 | TEMP_ADJUSTED=12.47 | PSAL=34.56 | PSAL_ADJUSTED=34.57 | BIOGEOCHEMISTRY: Dissolved oxygen: 250.30 µmol/kg | Nitrate: 45.20 µmol/kg | Chlorophyll a: 0.250 mg/m³ | CDOM=1.23 | DOWN_IRRADIANCE380=0.123 | DOWN_IRRADIANCE443=0.456 | ... [full details]"
  }
}
```

---

## Query Capabilities

### **Semantic Search**
- Search using natural language queries
- Embedded text summaries enable context-based retrieval
- Example: "Find floats with high nitrate levels in the Indian Ocean"

### **Metadata Filtering**
- Filter by: `float_type`, `ocean`, `data_mode`, `region`, `geohash`
- Range queries on: `latitude`, `longitude`, `depth`, `temperature`, `salinity`
- Example: `ocean = "Pacific Ocean" AND temp_adjusted_c > 15 AND data_mode = "D"`

### **Geographic Queries**
- Geohash-based spatial queries (precision 6 = ~1.2km cells)
- Latitude/Longitude range filters

### **Temporal Queries**
- Filter by `measurement_date`, `date_creation`, `date_update`
- Find recent data: `date_update > "2023-06-01"`

### **Quality-Based Queries**
- Filter by QC flags (e.g., `PROFILE_TEMP_QC = "1"` for good data)
- Find only delayed-mode data: `data_mode = "D"`

---

## Data Integrity & Consistency

✅ **Complete metadata preservation** - No fields lost during conversion  
✅ **Type consistency** - All fields consistently typed (Float, String, Integer)  
✅ **Date standardization** - All dates converted to ISO 8601 format  
✅ **Error handling** - Missing values stored as NULL or omitted fields  
✅ **Geohashing** - Automatically computed for geographic analysis  
✅ **Dual representations** - Both raw and adjusted measurements retained  
✅ **QC preservation** - All quality control flags included  

---

## Storage Statistics

- **Fields per record**: 50+ metadata fields
- **Summary text length**: 500-2000 characters per record
- **Embedding size**: 384 dimensions × 4 bytes ≈ 1.5 KB
- **Metadata size**: ~3-5 KB per record
- **Total per record**: ~5-7 KB

---

## Sources

- **Primary source**: Argo float netCDF/CSV data
- **CSV paths**:
  - BCG floats: `Data/BCG floats/final csv files/`
  - Non-BCG floats: `Data/Non-BCG floats/final_csv/`
- **Processing script**: `pinecone_upload.py`
- **Embedding model**: Hugging Face `sentence-transformers/all-MiniLM-L6-v2`


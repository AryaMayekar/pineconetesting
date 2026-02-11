import os
import json
import ollama
import geolib.geohash as gh
from mcp.server.fastmcp import FastMCP
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel
from datetime import datetime


# --- SETUP ---
load_dotenv()
mcp = FastMCP("Argo_MCP")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "argo-ocean-data"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

import psycopg2
import os

def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        dbname=os.getenv("PG_DATABASE"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD")
    )

PARAMETER_MAP = {
    "TEMP": ["temperature", "temp", "sea temperature"],
    "PSAL": ["salinity", "psal"],
    "PRES": ["pressure", "pres"],
    "DOXY": ["oxygen", "doxy", "dissolved oxygen"]
}

def resolve_parameter_name(user_input: str) -> Optional[str]:
    user_input = user_input.strip().lower()

    for canonical, synonyms in PARAMETER_MAP.items():
        if user_input == canonical.lower():
            return canonical
        if user_input in [s.lower() for s in synonyms]:
            return canonical

    return None

# =========================================================
# BASIC TOOL
# =========================================================

@mcp.tool()
def ping() -> str:
    """
    Health check tool.

    Returns:
        Confirmation string verifying MCP server is active.
    """
    return "MCP server is running."


# =========================================================
# LOCATION TOOL
# =========================================================

@mcp.tool()
def search_nearby_floats(location_name: str) -> str:
    """
    Search floats near a geographic location.

    Parameters:
        location_name (str): City, country, or ocean region.

    Returns:
        List of floats near the location including
        platform, region, temperature, depth and data mode.
    """

    response = ollama.chat(
        model="llama3.2",
        format="json",
        messages=[{
            "role": "user",
            "content": f"Extract latitude/longitude for: {location_name}. Return ONLY JSON with keys 'lat' and 'lon'."
        }]
    )

    try:
        coords = json.loads(response["message"]["content"])
        lat = float(str(coords.get("lat")).strip("°NnSs"))
        lon = float(str(coords.get("lon")).strip("°EeWw"))
    except Exception:
        return f"Could not determine coordinates from '{location_name}'."

    seen = set()

    for p in [3, 2, 1]:
        target_hash = gh.encode(lat, lon, p)

        results = index.query(
            vector=[0.0] * 384,
            top_k=10,
            include_metadata=True,
            filter={"geohash_list": {"$in": [target_hash]}}
        )

        matches = results.get("matches", [])
        if matches:

            output = [
                f"Floats near {location_name} (lat: {lat}, lon: {lon})"
            ]

            for m in matches:
                meta = m["metadata"]
                platform = meta.get("platform_number")

                if platform in seen:
                    continue
                seen.add(platform)

                output.append(
        f"""
- Platform: {meta.get('platform_number')}
  Float Type: {meta.get('float_type')}
  Ocean: {meta.get('ocean')}
  Region: {meta.get('avg_region')}
  Measurement Date: {meta.get('measurement_date')}
  Last Updated: {meta.get('date_update')}
"""
            )
            return "\n".join(output)

    return f"No floats found near {location_name}."


# =========================================================
# FLOAT DETAILS
# =========================================================

@mcp.tool()
def get_float_details(platform_number: str) -> str:
    """
    Retrieve only the important metadata for a specific float.

    Parameters:
        platform_number (str): Unique float ID.

    Returns:
        Complete sensor and metadata profile.
    """

    results = index.query(
        vector=[0.0] * 384,
        top_k=1,
        include_metadata=True,
        filter={"platform_number": platform_number}
    )

    matches = results.get("matches", [])
    if not matches:
        return f"No float found with platform {platform_number}."

    meta = matches[0]["metadata"]

    return f"""
Platform: {meta.get('platform_number')}
Float Type: {meta.get('float_type')}
Ocean: {meta.get('ocean')}
Region: {meta.get('avg_region')}
Measurement Date: {meta.get('measurement_date')}
Last Updated: {meta.get('date_update')}
Temperature (Adjusted): {meta.get('temp_adjusted_c')} °C
Salinity (Adjusted): {meta.get('psal_adjusted_psu')} PSU
Pressure Error: {meta.get('pres_adjusted_error')}
Max Depth: {meta.get('max_depth_m')} m
Cycle: {meta.get('cycle_number')}
"""




# =========================================================
# ADVANCED SEMANTIC TOOL
# =========================================================
from pydantic import BaseModel, ConfigDict
from typing import Optional

class SemanticFloatQueryInput(BaseModel):
    ocean: Optional[str] = None
    float_type: Optional[str] = None
    min_depth: Optional[float] = None
    min_temperature: Optional[float] = None
    max_temperature: Optional[float] = None
    qc_required: Optional[bool] = None
    data_mode: Optional[str] = None
    recent_after_year: Optional[int] = None
    model_config = ConfigDict(extra="ignore")  # 🔥 THIS LINE FIXES IT


from datetime import datetime
from typing import List
import ollama


@mcp.tool()
def semantic_float_query(input: SemanticFloatQueryInput) -> str:
    """
    Search ARGO float data using structured filtering criteria.

    Parameters:
        input (SemanticFloatQueryInput):
            ocean (Optional[str]): Name of the ocean (e.g., Indian Ocean, Pacific Ocean).
            float_type (Optional[str]): Type of float (Core or BGC).
            min_depth (Optional[float]): Minimum depth in meters.
            min_temperature (Optional[float]): Minimum temperature in °C.
            max_temperature (Optional[float]): Maximum temperature in °C.
            qc_required (Optional[bool]): If True, returns only quality-controlled data.
            data_mode (Optional[str]): Data mode (A = Adjusted, R = Real-time, D = Delayed).
            recent_after_year (Optional[int]): Filter floats measured after a specific year.

    Returns:
        A formatted list of matching floats including
        platform number, temperature, depth, region,
        ocean, float type, data mode, QC flag, and year.

    Notes:
        - Returns up to 50 matching results.
        - If no filters are provided, returns top available floats.
        - Numeric filters are applied using range conditions.
    """
    pinecone_filter = {}

    # 1. Force Numeric Conversion (Crucial for Pinecone)
    try:
        if input.min_temperature is not None:
            pinecone_filter["temp_adjusted_c"] = {"$gte": float(input.min_temperature)}
        
        if input.max_temperature is not None:
            if "temp_adjusted_c" not in pinecone_filter:
                pinecone_filter["temp_adjusted_c"] = {}
            pinecone_filter["temp_adjusted_c"]["$lte"] = float(input.max_temperature)

        if input.min_depth is not None:
            pinecone_filter["max_depth_m"] = {"$gte": float(input.min_depth)}
    except ValueError as e:
        return f"Error: Temperature and Depth must be numbers. Received: {e}"

    # 2. Quality Control (Integer match)
    if input.qc_required:
        pinecone_filter["temp_adjusted_qc"] = 1

    # 3. String Sanitization
    for field, value in [
        ("ocean", input.ocean),
        ("float_type", input.float_type),
        ("data_mode", input.data_mode)
    ]:
        # Ignore placeholders and empty strings
        if value and str(value).lower() not in ["none", "all", "null", ""]:
            pinecone_filter[field] = str(value).strip()

    # -----------------------------
    # 🔎 Execution
    # -----------------------------
    results = index.query(
        vector=[0.1] * 384, 
        top_k=50,
        include_metadata=True,
        filter=pinecone_filter if pinecone_filter else None
    )

    matches = results.get("matches", [])
    if not matches:
        return f"No results for filter: {pinecone_filter}. Check if ocean names or QC flags are too restrictive."

    # 📋 Format the matching data
    res_list = ["### 🌊 Results Found:"]
    for m in matches:
        meta = m.get("metadata", {})
        res_list.append(
    f"""
- **Platform Number**: {meta.get('platform_number')}
  • Temperature: {meta.get('temp_adjusted_c')}°C
  • Depth: {meta.get('max_depth_m')} m
  • Region: {meta.get('avg_region')}
  • Ocean: {meta.get('ocean')}
  • Float Type: {meta.get('float_type')}
  • Data Mode: {meta.get('data_mode')}
  • QC Flag: {meta.get('temp_adjusted_qc')}
  • Year: {meta.get('year')}
""".strip()
)
    
    return "\n".join(res_list)


# =========================================================
# Postgrese Tools
# =========================================================

@mcp.tool()
def get_raw_float_details(platform_number: str) -> dict:
    """
    Retrieve raw metadata and list available raw fields for a specific float.

    Parameters:
        platform_number (str):
            The ARGO platform number (unique float identifier).
            Example: "5906651"

    Returns:
        A structured JSON object containing:
        - Core metadata (float type, institution, profiler type, etc.)
        - A list of available raw fields stored in the database
    """

    float_id = str(platform_number).strip()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT float_id,
               float_type,
               profiler_type,
               institution,
               project_name,
               wmo_inst_type,
               platform_type,
               last_reading_date,
               all_columns
        FROM raw_float_metadata
        WHERE float_id = %s
    """, (float_id,))

    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:
        return {
            "status": "not_found",
            "float_id": float_id
        }

    all_columns = row[8] or []

    return {
        "status": "success",
        "float_id": row[0],
        "core_metadata": {
            "float_type": row[1],
            "profiler_type": row[2],
            "institution": row[3],
            "project_name": row[4],
            "wmo_inst_type": row[5],
            "platform_type": row[6],
            "last_reading_date": str(row[7]) if row[7] else None
        },
        "available_raw_fields": all_columns
    }




@mcp.tool()
def get_specific_parameter_readings(
    float_id: str,
    parameter_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50
) -> dict:
    """
    Retrieve parameter readings for a specific float within a date range.

    Parameters:
        float_id (str):
            The float/platform number.
            Example: "5906651"

        parameter_name (str):
            The parameter to retrieve.
            Can be canonical or natural language.
            Examples:
                "TEMP"
                "temperature"
                "salinity"
                "pressure"

        start_date (Optional[str]):
            Start date in ISO format (YYYY-MM-DD).
            Example: "2025-01-01"

        end_date (Optional[str]):
            End date in ISO format (YYYY-MM-DD).
            Example: "2025-12-31"

        limit (int):
            Maximum number of records to return.
            Default: 50

    Returns:
        A JSON object containing:
        - Canonical parameter name
        - Number of readings
        - List of readings (value, date, time)
    """

    float_id = str(float_id).strip()

    # 🔎 Resolve canonical parameter
    canonical_param = resolve_parameter_name(parameter_name)

    if not canonical_param:
        return {
            "status": "invalid_parameter",
            "requested_parameter": parameter_name,
            "available_parameters": list(PARAMETER_MAP.keys())
        }

    conn = get_connection()
    cur = conn.cursor()

    query = """
        SELECT value, JULD_DATE, JULD_TIME
        FROM raw_float_parameters_reading
        WHERE float_id = %s
        AND parameter_name = %s
    """

    params = [float_id, canonical_param]

    if start_date:
        query += " AND JULD_DATE >= %s"
        params.append(start_date)

    if end_date:
        query += " AND JULD_DATE <= %s"
        params.append(end_date)

    query += " ORDER BY JULD_DATE DESC, JULD_TIME DESC LIMIT %s"
    params.append(limit)

    cur.execute(query, tuple(params))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:
        return {
            "status": "not_found",
            "float_id": float_id,
            "parameter": canonical_param
        }

    return {
        "status": "success",
        "float_id": float_id,
        "parameter": canonical_param,
        "count": len(rows),
        "readings": [
            {
                "value": r[0],
                "date": str(r[1]) if r[1] else None,
                "time": str(r[2]) if r[2] else None
            }
            for r in rows
        ]
    }

@mcp.tool()
def get_latest_reading_of_a_parameter(float_id: str, parameter_name: str) -> dict:
    """
    Retrieve the most recent reading of a parameter for a float.

    Parameters:
        float_id (str):
            The float/platform number.
            Example: "5906651"

        parameter_name (str):
            The parameter name (canonical or natural language).
            Examples:
                "TEMP"
                "temperature"
                "oxygen"
                "salinity"

    Returns:
        A JSON object containing:
        - Canonical parameter name
        - Latest value
        - Date and time of measurement
    """

    float_id = str(float_id).strip()

    # 🔎 Resolve canonical parameter name
    canonical_param = resolve_parameter_name(parameter_name)

    if not canonical_param:
        return {
            "status": "invalid_parameter",
            "requested_parameter": parameter_name,
            "available_parameters": list(PARAMETER_MAP.keys())
        }

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT value, JULD_DATE, JULD_TIME
        FROM raw_float_parameters_reading
        WHERE float_id = %s
        AND parameter_name = %s
        ORDER BY JULD_DATE DESC, JULD_TIME DESC
        LIMIT 1
    """, (float_id, canonical_param))

    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:
        return {
            "status": "not_found",
            "float_id": float_id,
            "parameter": canonical_param
        }

    return {
        "status": "success",
        "float_id": float_id,
        "parameter": canonical_param,
        "latest_reading": {
            "value": row[0],
            "date": str(row[1]) if row[1] else None,
            "time": str(row[2]) if row[2] else None
        }
    }



@mcp.tool()
def search_floats_by_type(float_type: str, limit: int = 50) -> dict:
    """
    Retrieve floats filtered by float type.

    Parameters:
        float_type (str):
            Type of float.
            Allowed values:
                "BCG"
                "NON-BCG"

        limit (int):
            Maximum number of floats to return.
            Default: 50

    Returns:
        A JSON object containing:
        - Float type
        - Count
        - List of matching floats (float_id, institution, last_reading_date)
    """

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT float_id, institution, last_reading_date
        FROM raw_float_metadata
        WHERE float_type = %s
        ORDER BY last_reading_date DESC
        LIMIT %s
    """, (float_type.upper(), limit))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:
        return {
            "status": "not_found",
            "float_type": float_type
        }

    return {
        "status": "success",
        "float_type": float_type.upper(),
        "count": len(rows),
        "floats": [
            {
                "float_id": r[0],
                "institution": r[1],
                "last_reading_date": str(r[2]) if r[2] else None
            }
            for r in rows
        ]
    }




# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":
    mcp.run(transport="sse")

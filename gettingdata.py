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

## start downloading files in the Data folder

# FTP connection
ftp = FTP("ftp.ifremer.fr", timeout=120)
ftp.login()
#ftp.set_pasv(False)  # active mode to avoid timeout on Windows

# Remote + local directories
remote_dir = "/ifremer/argo/geo/indian_ocean/2025/08"
local_dir = "Data"
os.makedirs(local_dir, exist_ok=True)

# Change to the remote directory
ftp.cwd(remote_dir)

# List files in the remote folder
files = ftp.nlst()

# Download only missing files
for filename in files:
    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        print(f"Skipping {filename} (already downloaded)")
        continue

    with open(local_path, "wb") as f:
        ftp.retrbinary(f"RETR {filename}", f.write)
        print(f"Downloaded {filename}")

ftp.quit()


## for downloading all files

# files = ftp.nlst()  # list files
# for filename in files:
#     local_path = os.path.join(local_dir, filename)
#     with open(local_path, "wb") as f:
#         ftp.retrbinary(f"RETR {filename}", f.write)
#         print(f"Downloaded {filename}")

# ftp.quit()

# Define path to Data folder
# data_dir = "Data" 

# file_name = "20250801_prof.nc"   # change if needed (abhi ke liye using only one file)
# file_path = os.path.join(data_dir, file_name)

# # Debug: check if file exists
# print("Looking for:", os.path.abspath(file_path))
# print("Exists?", os.path.exists(file_path))

# # Open dataset if the file exists
# if os.path.exists(file_path):
#     ds = xr.open_dataset(file_path)
#     print("\nDataset summary:")
#     print(ds) # summary of variables & dimensions

#     print("\nAvailable variables:")
#     print(list(ds.variables.keys()))
# else:
#     print(f"ERROR: File not found at {file_path}")






from sqlalchemy import create_engine

# Step 2: Convert to DataFrame
# Select only key variables (to avoid memory overload)
# df = ds[["JULD", "LATITUDE", "LONGITUDE", "PRES", "TEMP", "PSAL"]].to_dataframe().reset_index()

# print("Data preview:")
# print(df.head())

# # Step 3: Connect to PostgreSQL
# engine = create_engine("postgresql://postgres:root@localhost:5432/postgres")

# # Step 4: Write DataFrame to SQL table
# df.to_sql("argo_profiles", engine, if_exists="replace", index=False)

# print("✅ Data successfully stored in PostgreSQL")






# Open dataset
# if not os.path.exists(file_path):
#     raise FileNotFoundError(f"File not found at {file_path}")

# # ---------- Open dataset ----------
# ds = xr.open_dataset(file_path)
# # print("\nDataset summary:")
# # print(ds)

# # ---------- Step 2: Select key variables ----------
# vars_to_keep = ["PLATFORM_NUMBER", "CYCLE_NUMBER", "JULD", "LATITUDE", "LONGITUDE", "PRES", "TEMP", "PSAL"]
# df = ds[vars_to_keep].to_dataframe().reset_index()

# # ---------- Step 3: Clean/convert variables ----------

# # PLATFORM_NUMBER: decode bytes if needed
# if df["PLATFORM_NUMBER"].dtype == object:
#     df["PLATFORM_NUMBER"] = df["PLATFORM_NUMBER"].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

# # CYCLE_NUMBER: ensure integer
# df["CYCLE_NUMBER"] = df["CYCLE_NUMBER"].astype(int)

# # JULD: convert to datetime
# if not pd.api.types.is_datetime64_any_dtype(df["JULD"]):
#     df["JULD"] = pd.to_timedelta(df["JULD"], unit='D') + pd.Timestamp("1950-01-01")
#     df["JULD"] = df["JULD"].dt.to_pydatetime()  # Python datetime

# # ---------- Step 4: Compute DEPTH from PRES ----------
# # gsw.z_from_p returns negative depth below sea surface, so multiply by -1
# df["DEPTH"] = df.apply(lambda row: -gsw.z_from_p(row["PRES"], row["LATITUDE"]), axis=1)

# # ---------- Step 5: Rename columns to match desired output ----------
# df = df.rename(columns={
#     "PLATFORM_NUMBER": "platform_id",
#     "CYCLE_NUMBER": "cycle_number",
#     "LATITUDE": "LATITUDE",
#     "LONGITUDE": "LONGITUDE",
#     "TEMP": "TEMP",
#     "PSAL": "PSAL"
# })

# # Optional: drop PRES column if you only want DEPTH
# df = df.drop(columns=["PRES"])

# # ---------- Step 6: Connect to PostgreSQL ----------
# engine = create_engine("postgresql://postgres:root@localhost:5432/postgres")

# # ---------- Step 7: Write DataFrame to SQL ----------
# df.to_sql("argo_profiles", engine, if_exists="replace", index=False)

# print("✅ Data successfully stored in PostgreSQL")
# print("\nData preview:")
# print(df.head())

# # ---------- Optionally save to CSV ----------
# output_csv = os.path.join(data_dir, "argo_filtered.csv")
# df.to_csv(output_csv, index=False)
# print(f"\nSaved filtered data to {output_csv}")
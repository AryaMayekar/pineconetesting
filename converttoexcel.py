import xarray as xr
import pandas as pd
import gsw   # pip install gsw
import numpy as np
import os

# Folders
nc_folder = r"F:\pineconetesting\NETcdf_Data"         # your NetCDF files
excel_folder = r"F:\pineconetesting\rough_excel_data"  # folder to save Excel files

# Make sure the output folder exists
os.makedirs(excel_folder, exist_ok=True)

# Function to decode byte strings
def decode_bytes(arr):
    if arr.dtype.kind in {'S', 'O'}:  # bytes or object
        return np.array([x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in arr])
    return arr

# Loop through all NetCDF files
for file in os.listdir(nc_folder):
    if file.endswith(".nc"):
        nc_file = os.path.join(nc_folder, file)
        ds = xr.open_dataset(nc_file)

        # Extract variables
        platform = decode_bytes(ds["PLATFORM_NUMBER"].values)
        project = decode_bytes(ds["PROJECT_NAME"].values)
        pi = decode_bytes(ds["PI_NAME"].values)
        cycle = ds["CYCLE_NUMBER"].values
        lat = ds["LATITUDE"].values
        lon = ds["LONGITUDE"].values
        time = pd.to_datetime(ds["JULD"].values)
        mode = decode_bytes(ds["DATA_MODE"].values)

        pres = ds["PRES"].values
        temp = ds["TEMP"].values
        psal = ds["PSAL"].values

        # Calculate depth from pressure
        depth = []
        for i in range(pres.shape[0]):
            depth.append(abs(gsw.z_from_p(pres[i, :], lat[i])))
        depth = np.array(depth)

        # Flatten all data into rows
        rows = []
        for i in range(pres.shape[0]):
            for j in range(pres.shape[1]):
                rows.append([
                    platform[i], project[i], pi[i], cycle[i],
                    lat[i], lon[i], time[i], mode[i],
                    pres[i, j], temp[i, j], psal[i, j], depth[i, j]
                ])

        # Create DataFrame
        df = pd.DataFrame(rows, columns=[
            "PLATFORM_NUMBER", "PROJECT_NAME", "PI_NAME", "CYCLE_NUMBER",
            "LATITUDE", "LONGITUDE", "TIME", "DATA_MODE",
            "PRES", "TEMP", "PSAL", "DEPTH"
        ])

        # Save Excel with same base name as NetCDF
        excel_file = os.path.join(excel_folder, file.replace(".nc", ".xlsx"))
        df.to_excel(excel_file, index=False)
        print(f"Saved: {excel_file}")

print("All files converted!")

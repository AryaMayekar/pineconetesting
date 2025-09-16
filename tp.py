import pandas as pd

import pandas as pd

# Load your Excel
df = pd.read_excel("argo_profiles.xlsx")

# Ensure TIME is a datetime object
df['TIME'] = pd.to_datetime(df['TIME'])

# Extract date only (ignore time of day)
df['DATE'] = df['TIME'].dt.date

# Ensure TIME is a datetime object
df['TIME'] = pd.to_datetime(df['TIME'])

# Extract date only (ignore time of day)
df['DATE'] = df['TIME'].dt.date

# Group by PLATFORM_NUMBER + CYCLE_NUMBER + DATE to get profile-specific summaries
profile_df = df.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER", "DATE"]).agg(
    PROJECT_NAME=("PROJECT_NAME", "first"),    # metadata
    PI_NAME=("PI_NAME", "first"),              # metadata
    LATITUDE_MEAN=("LATITUDE", "mean"),        # average latitude
    LATITUDE_MIN=("LATITUDE", "min"),          # min latitude
    LATITUDE_MAX=("LATITUDE", "max"),          # max latitude
    LONGITUDE_MEAN=("LONGITUDE", "mean"),      # average longitude
    LONGITUDE_MIN=("LONGITUDE", "min"),        # min longitude
    LONGITUDE_MAX=("LONGITUDE", "max"),        # max longitude
    avg_PRES=("PRES", "mean"),                 # average pressure
    avg_TEMP=("TEMP", "mean"),                 # average temperature
    avg_PSAL=("PSAL", "mean"),                 # average salinity
    avg_DEPTH=("DEPTH", "mean"),               # average depth
    min_DEPTH=("DEPTH", "min"),                # minimum depth
    max_DEPTH=("DEPTH", "max")                 # maximum depth
).reset_index()

# Save to Excel or CSV for PostgreSQL
profile_df.to_excel("argo_profile_summary.xlsx", index=False)
# profile_df.to_csv("argo_profile_summary.csv", index=False)

print("Profile-specific summary with DATE created successfully!")

import duckdb
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

train_path = (RAW / "train.csv").as_posix()
bmeta_path = (RAW / "building_metadata.csv").as_posix()
weather_path = (RAW / "weather_train.csv").as_posix()
out_path = (OUT / "ashrae_joined.parquet").as_posix()

con = duckdb.connect(database=":memory:")
con.execute("PRAGMA threads=4;")                 # uses available CPU
con.execute("PRAGMA enable_progress_bar=true;")  # shows progress
# Optional: if your codespace is small, keep this low (2GB–4GB)
# con.execute("PRAGMA memory_limit='3GB';")

query = f"""
COPY (
  SELECT
    t.building_id::INTEGER AS building_id,
    t.meter::INTEGER       AS meter,

    -- ✅ timestamp is already TIMESTAMP in duckdb
    t.timestamp AS ts,

    ln(1 + t.meter_reading) AS y,

    b.site_id::INTEGER     AS site_id,
    b.primary_use          AS primary_use,
    ln(1 + b.square_feet)  AS log_sqft,
    b.year_built::INTEGER  AS year_built,
    b.floor_count::INTEGER AS floor_count,

    w.air_temperature,
    w.dew_temperature,
    w.wind_speed,
    w.cloud_coverage,
    w.precip_depth_1_hr,
    w.sea_level_pressure,

    -- ✅ extract directly from TIMESTAMP
    extract('hour'  from t.timestamp)::INTEGER AS hour,
    extract('month' from t.timestamp)::INTEGER AS month,
    extract('isodow' from t.timestamp)::INTEGER AS isodow,
    CASE WHEN extract('isodow' from t.timestamp) IN (6,7)
         THEN 1 ELSE 0 END AS is_weekend

  FROM read_csv_auto('{train_path}', header=true) t
  JOIN read_csv_auto('{bmeta_path}', header=true) b
    ON t.building_id = b.building_id
  LEFT JOIN read_csv_auto('{weather_path}', header=true) w
    ON b.site_id = w.site_id AND t.timestamp = w.timestamp
) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
"""

print("Writing parquet to:", out_path)
con.execute(query)
print("Done ✅")
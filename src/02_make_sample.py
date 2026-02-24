import duckdb
from pathlib import Path

JOINED = "data/processed/ashrae_joined.parquet"
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# ✅ keep it small enough for Codespaces
OUT_SAMPLE = "data/processed/ashrae_sample.parquet"

con = duckdb.connect(database=":memory:")
con.execute("PRAGMA threads=4;")
con.execute("PRAGMA enable_progress_bar=true;")

# Simple sampling strategy (works reliably):
# take a subset of buildings + a time window
query = f"""
COPY (
  SELECT *
  FROM read_parquet('{JOINED}')
  WHERE building_id BETWEEN 0 AND 199
    AND ts >= TIMESTAMP '2016-01-01'
    AND ts <  TIMESTAMP '2016-03-01'
) TO '{OUT_SAMPLE}' (FORMAT PARQUET, COMPRESSION ZSTD);
"""

print("Writing sample to:", OUT_SAMPLE)
con.execute(query)
print("Done ✅")
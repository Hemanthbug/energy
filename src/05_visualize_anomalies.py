import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

IN_PATH = "data/processed/anomaly_scored.parquet"
OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(IN_PATH)

# Convert log back to normal scale for plots
df["actual"] = np.expm1(df["y"])
df["expected"] = np.expm1(df["y_pred"])
df["abs_resid"] = (df["y"] - df["y_pred"]).abs()

# ---- Plot 1: Top buildings by anomaly count ----
top_buildings = (
    df[df["is_anomaly"] == 1]
    .groupby("building_id")
    .size()
    .sort_values(ascending=False)
    .head(15)
    .reset_index(name="anomaly_count")
)

fig1 = px.bar(top_buildings, x="building_id", y="anomaly_count",
              title="Top 15 Buildings by Anomaly Count")
fig1.write_html(OUT_DIR / "01_top_buildings_anomalies.html")


# ---- Plot 2: Heatmap of anomalies (hour vs dayofweek) ----
heat = (
    df.groupby(["isodow", "hour"])["is_anomaly"]
    .mean()
    .reset_index(name="anomaly_rate")
)

fig2 = px.density_heatmap(
    heat,
    x="hour",
    y="isodow",
    z="anomaly_rate",
    title="Anomaly Rate Heatmap (DayOfWeek vs Hour)"
)

fig2.write_html(OUT_DIR / "02_anomaly_heatmap.html")

# ---- Plot 3: Actual vs Expected with anomaly markers for ONE building ----
b = int(top_buildings["building_id"].iloc[0]) if len(top_buildings) else int(df["building_id"].iloc[0])
df_b = df[df["building_id"] == b].sort_values("ts").head(2000)  # limit for display

fig3 = px.line(df_b, x="ts", y=["actual", "expected"],
               title=f"Building {b}: Actual vs Expected (first 2000 points)")
# add anomaly points
anom = df_b[df_b["is_anomaly"] == 1]
fig3.add_scatter(x=anom["ts"], y=anom["actual"], mode="markers", name="Anomalies")

fig3.write_html(OUT_DIR / "03_actual_vs_expected_with_anomalies.html")

print("Saved 3 visualizations to outputs/plots/")
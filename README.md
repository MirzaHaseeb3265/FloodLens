# FloodLens Streamlit dashboard

## 1. Folder structure

```text
FloodLens_Streamlit/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── Pakistan_GPM_Rainfall_2001_2024.csv
    ├── FloodLens_Pakistan_Pixel_Trends_2001_2024.csv
    ├── pakistan_boundary.geojson
    └── FloodLens_Pakistan_Terrain_1km.tif   (optional)
```

The dashboard can start without the optional terrain GeoTIFF, but the national
rainfall CSV, pixel trend CSV and boundary GeoJSON are needed for the main pages.

## 2. Export the analysis files from your existing Google Colab notebook

Run these cells after the variables `df`, `trend_df`, `original_df` and
`pakistan_geometry` exist.

### A. National monthly rainfall CSV

If you already have `rainfall_df`:

```python
rainfall_df.to_csv(
    "/content/Pakistan_GPM_Rainfall_2001_2024.csv",
    index=False
)
```

Or, if you have `df` with `date` and `monthly_rainfall_mm`:

```python
df.to_csv(
    "/content/Pakistan_GPM_Rainfall_2001_2024.csv",
    index=False
)
```

### B. Pixel trend CSV

This joins Sen's slope and Hamed–Rao results in `trend_df` with the original
Mann–Kendall p-values and BH/FDR-adjusted p-values from `original_df`.

```python
pixel_trends = trend_df.merge(
    original_df[["latitude", "longitude", "original_mk_p", "adjusted_p"]],
    on=["latitude", "longitude"],
    how="left"
)

pixel_trends.to_csv(
    "/content/FloodLens_Pakistan_Pixel_Trends_2001_2024.csv",
    index=False
)
```

**Important:** In your notebook, `original_df["adjusted_p"]` was calculated
using the original Mann–Kendall p-values with Benjamini–Hochberg FDR. The
dashboard uses this column for the significance map. It does not use the
Hamed–Rao p-values for this map.

### C. Pakistan boundary GeoJSON

Your `pakistan_geometry.getInfo()` returned a GeometryCollection. This cell
wraps it as a valid GeoJSON FeatureCollection:

```python
import json

boundary = pakistan_geometry.simplify(1000).getInfo()

geojson = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {"name": "Pakistan"},
        "geometry": boundary
    }]
}

with open("/content/pakistan_boundary.geojson", "w") as f:
    json.dump(geojson, f)
```

### D. Copy files to your computer

In Colab, run:

```python
from google.colab import files

files.download("/content/Pakistan_GPM_Rainfall_2001_2024.csv")
files.download("/content/FloodLens_Pakistan_Pixel_Trends_2001_2024.csv")
files.download("/content/pakistan_boundary.geojson")
```

If you want the terrain page too, download your existing
`FloodLens_Pakistan_Terrain_1km.tif` from Google Drive and put it in `data/`.

Then place all downloaded files inside the project's `data` folder.

## 3. Run locally

Open a terminal in this project folder and run:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install packages and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will show a local URL, usually `http://localhost:8501`.

## 4. Deploy

Push this folder to a GitHub repository. Add the same data files to the
repository only if their size and the dataset's licensing/hosting arrangement
are appropriate. For large GeoTIFFs, use an external data host or omit the
terrain file for the first deployment.

On Streamlit Community Cloud:
- Select the repository and branch.
- Set the main file path to `app.py`.
- Streamlit installs packages from `requirements.txt`.

## Scientific notes

- The national Hamed–Rao modified Mann–Kendall result in the current analysis
  is increasing, p = 0.035, with Sen's slope approximately +5.231 mm/year.
- The spatial significance map uses original Mann–Kendall p-values with
  Benjamini–Hochberg FDR correction. In the current analysis, zero of 7,335
  pixels passed adjusted p < 0.05.
- A coloured Sen's slope is an estimated rate of change, not proof of
  statistical significance.
- Rainfall trends do not constitute a flood forecast or official warning.
- SRTM elevation is static geographic context, not a time series.

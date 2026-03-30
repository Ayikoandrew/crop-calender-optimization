# Data Directory

This directory contains rainfall data for crop calendar optimization in western Uganda. The dataset comprises daily satellite-derived precipitation estimates from two complementary sources across five rainfall regions, spanning the period 1983-2019.

## Dataset Citation

Diem, Jeremy (2021), "Daily rainfall totals for western Uganda, 1983-2019", Mendeley Data, V2, doi: [10.17632/ppxktx233y.2](https://data.mendeley.com/datasets/ppxktx233y/2)

License: CC BY 4.0

The regional delineation and climatological analysis are described in the accompanying paper:

> Diem, J. E., Sung, H. S., Konecky, B. L., Palace, M. W., Salerno, J., and Hartter, J. (2019). Rainfall characteristics and trends—and the role of Congo westerlies—in the western Uganda transition zone of equatorial Africa from 1983 to 2017. *Journal of Geophysical Research: Atmospheres*, 124, 10712-10729. https://doi.org/10.1029/2019JD031243

## Data Sources

The rainfall estimates are derived from two satellite precipitation products:

- **CHIRPS** (Climate Hazards Group InfraRed Precipitation with Station data): A quasi-global gridded rainfall dataset combining satellite imagery with in-situ station observations.
- **TAMSAT** (Tropical Applications of Meteorology using SATellite data): An African-focused precipitation dataset developed by the University of Reading.

The `Mean` column represents the arithmetic mean of CHIRPS and TAMSAT values, providing a more robust precipitation estimate by averaging systematic biases inherent in each product.

## Directory Structure

```
data/
├── raw/
│   ├── RainfallData/
│   │   ├── Region1.csv
│   │   ├── Region2.csv
│   │   ├── Region3.csv
│   │   ├── Region4.csv
│   │   └── Region5.csv
│   └── Regions/
│       └── [ESRI Shapefile components]
└── processed/
    ├── combined_rainfall_data.csv
    ├── rainfall_features_full.csv
    ├── rainfall_rl_features.csv
    ├── monthly_rainfall_stats.csv
    ├── annual_rainfall_totals.csv
    ├── rainy_day_probability.csv
    ├── season_dates.csv
    └── [visualization outputs]
```

## Raw Data

### RainfallData (CSV)

Daily rainfall records for five regions with the following schema:

| Column  | Type    | Description                          |
|---------|---------|--------------------------------------|
| Year    | Integer | Calendar year (1983-2019)            |
| Month   | Integer | Calendar month (1-12)                |
| Day     | Integer | Day of month (1-31)                  |
| CHIRPS  | Float   | CHIRPS precipitation estimate (mm)   |
| TAMSAT  | Float   | TAMSAT precipitation estimate (mm)   |
| Mean    | Float   | Average of CHIRPS and TAMSAT (mm)    |

Total records: approximately 67,525 daily observations (5 regions x 37 years x 365 days).

### Regions (Shapefile)

ESRI Shapefile containing polygon geometries defining the five rainfall regions in western Uganda as delineated by Diem et al. (2019). Components include `.shp` (geometry), `.dbf` (attributes), `.prj` (coordinate reference system), and associated index files.

## Processed Data

The following datasets are generated through the data exploration and feature engineering pipelines.

### combined_rainfall_data.csv

Consolidated daily rainfall records from all five regions with computed date fields. Contains the original CHIRPS, TAMSAT, and Mean columns plus a Region identifier and parsed Date column.

### rainfall_features_full.csv

Comprehensive feature matrix for machine learning applications containing 50+ engineered features:

**Rolling Statistics** (7, 14, and 30-day windows):
- `rainfall_sum_Xd`: Cumulative rainfall over window
- `rainfall_avg_Xd`: Mean daily rainfall over window
- `rainfall_max_Xd`: Maximum daily rainfall intensity
- `rainfall_std_Xd`: Standard deviation (variability measure)
- `rainy_days_Xd`: Count of days exceeding 1mm threshold

**Dry Spell Indicators**:
- `dry_spell_length`: Consecutive days below 1mm threshold
- `max_dry_spell_30d`: Maximum dry spell in preceding 30 days
- `days_since_rain_5mm`: Days since last significant rainfall event

**Temporal Encodings**:
- `doy_sin`, `doy_cos`: Cyclical day-of-year encoding
- `month_sin`, `month_cos`: Cyclical month encoding
- `dekad`: 10-day period within month (1-3)
- `week_of_year`: ISO week number

**Lag Features**:
- `rainfall_lag_Xd`: Rainfall values at 1, 7, 14, 30-day lags
- `rainfall_lag_1y`: Same-day rainfall from previous year

**Cumulative Features**:
- `cumulative_annual`: Year-to-date rainfall accumulation
- `cumulative_season`: Rainfall since season onset

**Anomaly Features**:
- `clim_mean`: Long-term climatological mean for day-of-year
- `rainfall_anomaly`: Deviation from climatological mean
- `rainfall_zscore`: Standardized anomaly (z-score)
- `rainfall_percentile`: Percentile rank relative to historical distribution

### rainfall_rl_features.csv

A reduced feature subset optimized for reinforcement learning state representation. Contains the most informative predictors for crop calendar decision-making.

### monthly_rainfall_stats.csv

Monthly aggregated statistics by region including mean, standard deviation, and total rainfall for each combination of region and month.

### annual_rainfall_totals.csv

Annual total rainfall by region for trend analysis and inter-annual variability assessment.

### rainy_day_probability.csv

Monthly probability of rainfall occurrence by region, computed as the proportion of days exceeding the 1mm threshold.

### season_dates.csv

Detected rainy season onset and cessation dates for each region-year combination. Detection criteria:

- **Onset**: First 3-day period with cumulative rainfall exceeding 20mm, followed by no dry spell exceeding 7 consecutive days within the subsequent 30 days.
- **Cessation**: Last 3-day period with cumulative rainfall exceeding 10mm before the end of the rainy season window.

Derived metrics include:
- `onset_doy`: Day of year for season onset
- `cessation_doy`: Day of year for season cessation
- `season_length`: Duration in days

## Processing Pipeline

1. **Data Exploration** (`notebooks/01_data_exploration.ipynb`): Quality assessment, distribution analysis, correlation between products, seasonal pattern identification.

2. **Feature Engineering** (`notebooks/02_feature_engineering.ipynb`): Rolling statistics computation, temporal encoding, anomaly calculation, season detection.

## References

Diem, J. E., Sung, H. S., Konecky, B. L., Palace, M. W., Salerno, J., and Hartter, J. (2019). Rainfall characteristics and trends—and the role of Congo westerlies—in the western Uganda transition zone of equatorial Africa from 1983 to 2017. *Journal of Geophysical Research: Atmospheres*, 124, 10712-10729. https://doi.org/10.1029/2019JD031243

Funk, C., et al. (2015). The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. *Scientific Data*, 2, 150066. https://doi.org/10.1038/sdata.2015.66

Maidment, R. I., et al. (2014). A new, long-term daily satellite-based rainfall dataset for operational monitoring in Africa. *Scientific Data*, 1, 140017. https://doi.org/10.1038/sdata.2014.17
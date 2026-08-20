# Persistent Winter Electricity Demand in SEAS5

This repository contains the code and data used to support the analysis in an MSc dissertation on persistent winter electricity demand in Great Britain.

The study compares historical winter conditions with ECMWF SEAS5 seasonal forecast ensemble realisations, with particular emphasis on electricity-demand severity across different averaging durations and on the behaviour of the upper tail.

## Overview

The analysis considers winter electricity-demand severity over multiple averaging windows, ranging from short-duration peaks to sustained high-demand periods.

The main comparison is between:

- a historical Great Britain weather and demand sample; and
- ECMWF SEAS5 seasonal forecast ensemble realisations for winters 1982--2016.

For SEAS5, 25 ensemble members are available for each of 35 winters, giving 875 winter-member realisations.

The analysis includes:

- processing of SEAS5 daily weather variables;
- population weighting over Great Britain;
- deterministic weather-to-demand modelling;
- rolling-mean winter demand severity;
- mean alignment between historical and SEAS5 samples;
- empirical upper-tail comparisons;
- diagnostics of medium-duration upper-tail structure;
- stationary generalised extreme-value (GEV) modelling;
- winter-year cluster bootstrap uncertainty intervals;
- non-stationary GEV sensitivity analysis;
- mean residual life diagnostics; and
- additional sensitivity and definition-integrity checks.

## Repository structure

```text
persistent-winter-demand-seas5/
│
├── config.py
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── ECMWF SEAS5 winter NetCDF files
│   ├── population/
│   │   └── WorldPop population raster
│   └── processed/
│       └── population_weights_used_by_02.csv
│
├── outputs/
│   ├── daily/
│   │   ├── ecmwf_daily_weather_1982_2016.csv
│   │   ├── ecmwf_daily_demand_Nov08_1982_2016.csv
│   │   ├── hannah_uk_daily_weather_inputs.csv
│   │   └── hannah_daily_demand_deterministic_NDJFM.csv
│   │
│   └── severity/
│       ├── ecmwf_severity_summary_Nov08_1982_2016_extended_windows.csv
│       └── hannah_severity_summary_Nov08_extended_windows.csv
│
└── scripts/
    ├── data_processing/
    └── final_analysis/


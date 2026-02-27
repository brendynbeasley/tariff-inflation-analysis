# Sectoral Inflation & Tariff Dynamics

This repository contains Python and Julia scripts designed to analyze and visualize U.S. sectoral inflation dynamics against the implementation dates of major trade tariffs. 

The scripts automatically fetch disaggregated Consumer Price Index (CPI) data from the Federal Reserve Economic Data (FRED) API, calculate inflation metrics, and generate publication-ready charts mapping price changes alongside policy events.

![Inflation Plot Example](inflation_plot_second_term.png) 
*(Note: Upload your generated PNG to the repo and this image will display here)*

## Features

* **Automated Data Fetching:** Directly pulls Headline, Core, Food, and Energy CPI series from FRED. Uses `vintage_date` pinning (in the Julia version) for strict academic reproducibility.
* **Custom Calculations:** Easily toggle between Month-over-Month (MoM) or Year-over-Year (YoY) calculations, outputting either a Gross Rate ($P_t / P_{t-1}$) or a percentage change. Includes optional rolling-average smoothing.
* **Collision-Free Visualization:** Implements a numbered reference system and formatted data table to elegantly handle dense clusters of tariff implementation dates without inline text overlap.
* **Policy Encoding:** Tariff events are weighted by estimated import exposure ($B) and color-coded by legal authority (e.g., IEEPA, Section 232, Section 301).
* **Multi-Era Support:** Built-in databases for both the 2018–2019 trade war (Trump 1.0) and the 2025–2026 tariffs (Trump 2.0).

## Data Sources
* **Inflation Data:** [FRED (Federal Reserve Bank of St. Louis)](https://fred.stlouisfed.org/)
* **Tariff Dates & Scope:** * [Atlantic Council Trump Tariff Tracker](https://www.atlanticcouncil.org/programs/geoeconomics-center/trump-tariff-tracker/)
  * [Congressional Research Service (CRS R48549)](https://www.congress.gov/crs-product/R48549)

## Installation & Requirements

You can run this analysis using either Python or Julia. 

### Python Setup
Requires Python 3.8+ and the following packages:
```bash
pip install pandas matplotlib pandas-datareader

# Inside the Julia REPL, run:
using Pkg
Pkg.add(["DataFrames", "CSV", "HTTP", "PyPlot", "PyCall"])

# For Python
python tariff_inflation_analysis.py

# For Julia
julia tariff_inflation_analysis.jl

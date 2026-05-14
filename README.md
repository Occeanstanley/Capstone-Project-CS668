# Capstone-Project-CS668

# 🏠 Economic Drivers of U.S. Housing Prices

## 📌 Overview
This project examines how major economic factors relate to housing prices across the 50 U.S. states from 2020 to 2024. The analysis uses a **50-state panel dataset** and applies a **two-way fixed effects panel regression** as the main model, with **Random Forest** used as a comparison model. The goal is to identify which economic variables are most strongly associated with housing prices over time.

## 🎯 Objective
To examine how cost of living, income growth, unemployment, population, and migration relate to the Housing Price Index (HPI) across the United States, while using a panel-data approach that accounts for differences across states and across years.

## 📄 Project Files
- [Final Paper](./Final_Paper.pdf)
- [Poster](./Poster.pdf)
- [Capstone Presentation](./Capstone_Presentation.pdf)
- [Master Regression Workbook](./capstone_master_regression_workbook.xlsx)

## ❓ Research Question
In a 50-state panel from 2020 to 2024, how are unemployment, income growth, population, migration, and cost-of-living differences associated with housing prices over time?

## 📊 Data Sources
This project combines state-level data from the following sources:

- **Federal Housing Finance Agency (FHFA)** – House Price Index (HPI)
- **Bureau of Economic Analysis (BEA)** – Regional Price Parities (RPP)
- **Bureau of Labor Statistics (BLS)** – Unemployment Rate
- **U.S. Census Bureau** – Population Data
- **IRS** – State-to-State Migration Data

## 🧠 Methodology
The project follows a structured panel-data workflow:

1. **Data Collection and Cleaning**
   - Load and clean the core state-year dataset
   - Standardize state names
   - Remove District of Columbia to keep a consistent 50-state sample

2. **Dataset Merging**
   - Merge FHFA, BLS, Census, BEA, and IRS-based migration data into one **state-year panel**

3. **Exploratory Data Analysis (EDA)**
   - Correlation heatmap
   - Scatter plots and regression grids
   - Trend analysis
   - Geographic visualization

4. **Main Statistical Model**
   - **Two-way fixed effects panel regression**
   - State fixed effects control for stable differences across states
   - Year fixed effects control for common shocks across years

5. **Comparison Model**
   - **Random Forest Regression**

6. **Model Evaluation**
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² Score

## 🧾 Variables Used
- **Housing Price Index (HPI)** – dependent variable
- **Regional Price Parity (RPP)** – cost-of-living measure
- **Income Change**
- **Unemployment Rate**
- **Population**
- **Net Migration** – included as a supporting variable from IRS migration files

## 📈 Key Findings
- **Regional Price Parity (RPP)** showed one of the strongest positive relationships with housing prices
- **Unemployment rate** was negatively associated with housing prices
- **Income growth** contributed to the housing-price story, though less strongly than RPP
- **Population** was weaker than the main labor and cost-of-living variables
- **Migration** was included as supporting information, but the main panel-regression model focused on the full 2020–2024 panel structure
- Housing prices increased steadily from 2020 to 2024

## 🤖 Model Insights
- The **two-way fixed effects panel regression** was the main model because it better matched the structure of the 50-state panel dataset
- **Random Forest** was used only as a comparison model for predictive performance
- Model results showed that the relationship between housing prices and economic conditions was clearer when state and year effects were taken into account

## 🌍 Spatial Analysis
A geographic visualization (ArcGIS map) was used to show how housing prices vary across states. The map supports the broader finding that higher-cost regions tend to have higher housing prices.

## ⚠️ Limitations
- State-level data may hide city-level or county-level variation
- Migration data is limited to the IRS flow-year files for **2020–2021**, **2021–2022**, and **2022–2023**
- Migration is included as supporting information, but not emphasized as the central regression result
- The models identify statistical relationships, not causal effects
- The sample begins in 2020, so COVID-19 may have influenced housing demand and market conditions during the study period
- Important housing-market variables such as interest rates, housing supply, and mortgage availability were not directly included

## 🚀 Future Work
- Use city-level or county-level data
- Incorporate interest rates, housing supply, or mortgage availability
- Apply additional machine learning models such as XGBoost or Gradient Boosting
- Extend migration analysis with longer time coverage
- Explore additional panel-data specifications and robustness checks

## ▶️ How to Run
1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
---
   ```bash
python housing_project_analysis.py

```
## 📁 Project Structure
```
Capstone-Project-CS668/
├── README.md
├── Final_Paper.pdf
├── Poster.pdf
├── Capstone_Presentation.pdf
├── housing_project_analysis.py
├── requirements.txt
├── RPP_2020_2024_clean.xlsx
├── economic_data_rpp_analysis.xlsx
├── hpi_master.csv
├── capstone_master_regression_workbook.xlsx
├── state_year_unemployment_clean.csv
├── stateinflow2020-2021.csv
├── stateinflow2021-2022.csv
├── stateinflow2223.csv
├── stateoutflow2020-2021.csv
├── stateoutflow2021-2022.csv
└── stateoutflow2223.csv


---

## 👥 Authors
- Stanley Occean  
- Jonah Liautaud  

Pace University – M.S. Data Science

---

## 📄 License
This project is for academic purposes.

# Capstone-Project-CS668

# 🏠 Economic Drivers of U.S. Housing Prices

## 📌 Overview
This project examines how major economic factors relate to housing prices across the 50 U.S. states from 2020 to 2024. The goal is to identify which variables are most strongly associated with differences in housing prices using state-level data, exploratory data analysis, multiple linear regression, and Random Forest Regression.

## 🎯 Objective
To examine how cost of living, income growth, unemployment, population, and migration relate to the Housing Price Index (HPI) across the United States.

## 📄 Project Files
- [Final Paper](./Final_Paper.pdf)
- [Poster](./Poster.pdf)

## ❓ Research Question
How are unemployment, income growth, population change, migration, and cost-of-living differences associated with housing prices in U.S. states between 2020 and 2024?

## 📊 Data Sources
This project combines state-level data from the following sources:

- **Federal Housing Finance Agency (FHFA)** – House Price Index (HPI)
- **Bureau of Economic Analysis (BEA)** – Regional Price Parities (RPP)
- **Bureau of Labor Statistics (BLS)** – Unemployment Rate
- **U.S. Census Bureau** – Population Data
- **IRS** – State-to-State Migration Data

## 🧠 Methodology
The project follows a structured data analysis pipeline:

1. **Data Collection and Cleaning**
   - Load and clean the core state-year dataset
   - Standardize state names
   - Remove District of Columbia to keep a consistent 50-state sample

2. **Dataset Merging**
   - Merge FHFA, BLS, Census, BEA, and IRS-based migration data into one state-year panel

3. **Exploratory Data Analysis (EDA)**
   - Correlation heatmap
   - Scatter plots
   - Trend analysis
   - Geographic visualization

4. **Statistical Modeling**
   - Multiple Linear Regression
   - Random Forest Regression

5. **Model Evaluation**
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² Score

## 🧾 Variables Used
- **Housing Price Index (HPI)** – dependent variable
- **Regional Price Parity (RPP)** – cost-of-living measure
- **Income Change**
- **Unemployment Rate**
- **Population**
- **Net Migration**

## 📈 Key Findings
- **Regional Price Parity (RPP)** showed the strongest relationship with housing prices
- **Income growth** had a positive but weaker relationship with housing prices
- **Unemployment rate** was negatively associated with housing prices
- **Migration** appeared weaker than the other main variables and was limited to the available IRS flow years
- Housing prices increased steadily from 2020 to 2024

## 🤖 Model Insights
- Random Forest outperformed linear regression on the full baseline dataset
- Linear regression performed better on the smaller migration subset
- Model performance depended on dataset size and feature availability

## 🌍 Spatial Analysis
A geographic visualization (ArcGIS map) was used to show how housing prices vary across states. The map supports the broader finding that higher-cost regions tend to have higher housing prices.

## ⚠️ Limitations
- State-level data may hide city-level or county-level variation
- Migration data is limited to the IRS flow-year files for **2020–2021**, **2021–2022**, and **2022–2023**
- In the merged panel, migration is only analyzed for **2021–2023**
- The models identify statistical relationships, not causal effects
- The sample begins in 2020, so COVID-19 may have influenced housing demand and market conditions during the study period

## 🚀 Future Work
- Use city-level or county-level data
- Incorporate interest rates, housing supply, or mortgage availability
- Apply additional machine learning models such as XGBoost or Gradient Boosting
- Explore panel-data methods such as fixed effects models

## ▶️ How to Run
1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   
---

## 📁 Project Structure


---

## 👥 Authors
- Stanley Occean  
- Jonah Liautaud  

Pace University – M.S. Data Science

---

## 📄 License
This project is for academic purposes.

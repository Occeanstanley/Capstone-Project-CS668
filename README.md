# Capstone-Project-CS668
# 🏠 Economic Drivers of U.S. Housing Prices

## 📌 Overview
This project analyzes the key economic factors influencing housing prices across U.S. states from 2020 to 2024. The goal is to identify which variables most strongly impact housing price differences using data analysis and machine learning techniques.

---

## 🎯 Objective
To determine how economic indicators such as cost of living, income, unemployment, and migration affect the Housing Price Index (HPI) across the United States.

---

## 📊 Data Sources
- Federal Housing Finance Agency (FHFA) – Housing Price Index  
- Bureau of Economic Analysis (BEA) – Regional Price Parity (RPP)  
- Bureau of Labor Statistics (BLS) – Unemployment Rate  
- U.S. Census Bureau – Population Data  
- IRS – Migration Data  

---

## 🧠 Methodology
The project follows a structured data analysis pipeline:

1. Data Collection and Cleaning  
2. Dataset Merging (state-level data)  
3. Exploratory Data Analysis (EDA)  
   - Correlation heatmap  
   - Scatter plots  
   - Trend analysis  
4. Statistical Modeling  
   - Multiple Linear Regression  
   - Random Forest Regression  
5. Model Evaluation  
   - RMSE (Root Mean Squared Error)  
   - MAE (Mean Absolute Error)  
   - R² Score  

---

## 📈 Key Findings
- **Regional Price Parity (RPP)** is the strongest predictor of housing prices  
- **Income growth** has a positive but moderate effect  
- **Unemployment rate** is negatively associated with housing prices  
- **Migration** has the weakest impact  
- Housing prices increased steadily from 2020 to 2024  

---

## 🤖 Model Insights
- Random Forest outperformed linear regression on the full dataset  
- Linear regression performed better on smaller datasets (with migration)  
- Model performance depends on dataset size and feature availability  

---

## 🌍 Spatial Analysis
A geographic visualization (ArcGIS map) was used to show how housing prices vary across states, confirming that high-cost regions tend to have higher housing prices.

---

## ⚠️ Limitations
- State-level data may hide city-level variations  
- Migration data limited to a shorter time period  
- COVID-19 impacted economic conditions during the study period  

---

## 🚀 Future Work
- Use city-level or county-level data  
- Incorporate interest rates and housing supply  
- Apply advanced machine learning models (XGBoost, etc.)  

---

## 📁 Project Structure

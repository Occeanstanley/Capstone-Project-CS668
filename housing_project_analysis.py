# =========================================================
# CAPSTONE PROJECT ANALYSIS SCRIPT
# Economic Drivers of U.S. Housing Prices
#
# Main model: Two-way fixed effects panel regression
# Comparison model: Random Forest
# Dataset: 50-state panel, 2020-2024
# =========================================================

from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =========================================================
# 1. FILE PATHS AND PROJECT SETUP
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "capstone_master_regression_workbook.xlsx"

FILES = {
    "merged_workbook": BASE_DIR / "economic_data_rpp_analysis.xlsx",
    "inflow_2021": BASE_DIR / "stateinflow2020-2021.csv",
    "outflow_2021": BASE_DIR / "stateoutflow2020-2021.csv",
    "inflow_2022": BASE_DIR / "stateinflow2021-2022.csv",
    "outflow_2022": BASE_DIR / "stateoutflow2021-2022.csv",
    "inflow_2023": BASE_DIR / "stateinflow2223.csv",
    "outflow_2023": BASE_DIR / "stateoutflow2223.csv",
}

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY"
}

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def normalize_state_names(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.title()
        .replace({
            "District Of Columbia": "District of Columbia",
            "D.C.": "District of Columbia",
        })
    )


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def first_match(columns: Iterable[str], candidates: Iterable[str]) -> str:
    cols = list(columns)
    lowered = {c.lower(): c for c in cols}
    for candidate in candidates:
        if candidate in cols:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise KeyError(f"Could not find columns {list(candidates)} in available columns: {cols}")


# =========================================================
# 3. DATA COLLECTION / LOADING
# =========================================================

def load_primary_dataset() -> pd.DataFrame:
    workbook = FILES["merged_workbook"]
    if not workbook.exists():
        raise FileNotFoundError(
            "economic_data_rpp_analysis.xlsx was not found. Place it in the repo root."
        )

    xls = pd.ExcelFile(workbook)
    preferred_sheet_order = ["Merged_Data", "Merged", "Data", "Sheet1"]
    chosen_sheet = None

    for sheet in preferred_sheet_order:
        if sheet in xls.sheet_names:
            chosen_sheet = sheet
            break

    if chosen_sheet is None:
        chosen_sheet = xls.sheet_names[0]

    df = pd.read_excel(workbook, sheet_name=chosen_sheet)

    rename_map = {}
    for col in df.columns:
        low = str(col).strip().lower()
        if low in {"state", "geoname"}:
            rename_map[col] = "state"
        elif low == "year":
            rename_map[col] = "year"
        elif low in {"price_index", "hpi", "house_price_index", "housing_price_index"}:
            rename_map[col] = "price_index"
        elif low in {"rpp", "regional_price_parity"}:
            rename_map[col] = "rpp"
        elif low in {"income_change", "income_growth"}:
            rename_map[col] = "income_change"
        elif low in {"unemployment_rate", "unemployment"}:
            rename_map[col] = "unemployment_rate"
        elif low == "population":
            rename_map[col] = "population"
        elif low in {"state_fips", "geofips"}:
            rename_map[col] = "state_fips"

    df = df.rename(columns=rename_map)

    required = [
        "state",
        "year",
        "price_index",
        "unemployment_rate",
        "income_change",
        "population",
        "rpp",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    df["state"] = normalize_state_names(df["state"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)

    for col in ["price_index", "unemployment_rate", "income_change", "population", "rpp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "state_fips" in df.columns:
        df["state_fips"] = pd.to_numeric(df["state_fips"], errors="coerce")

    df = df[df["state"] != "District of Columbia"].copy()
    return df


# =========================================================
# 4. MIGRATION DATA COLLECTION AND PROCESSING
# =========================================================

def build_migration_for_year(inflow_path: Path, outflow_path: Path, panel_year: int) -> pd.DataFrame:
    if not inflow_path.exists() or not outflow_path.exists():
        return pd.DataFrame(columns=["state_fips", "inflow_people", "outflow_people", "net_migration", "year"])

    inflow = pd.read_csv(inflow_path)
    outflow = pd.read_csv(outflow_path)

    inflow_origin = first_match(inflow.columns, ["y1_statefips", "statefips1", "origin_state_fips"])
    inflow_dest = first_match(inflow.columns, ["y2_statefips", "statefips2", "destination_state_fips"])
    inflow_people = first_match(inflow.columns, ["n2", "num_returns_exemptions", "exmpt_num"])

    outflow_origin = first_match(outflow.columns, ["y1_statefips", "statefips1", "origin_state_fips"])
    outflow_dest = first_match(outflow.columns, ["y2_statefips", "statefips2", "destination_state_fips"])
    outflow_people = first_match(outflow.columns, ["n2", "num_returns_exemptions", "exmpt_num"])

    inflow = inflow[inflow[inflow_origin] != inflow[inflow_dest]].copy()
    outflow = outflow[outflow[outflow_origin] != outflow[outflow_dest]].copy()

    inflow_agg = (
        inflow.groupby(inflow_dest, as_index=False)[inflow_people]
        .sum()
        .rename(columns={inflow_dest: "state_fips", inflow_people: "inflow_people"})
    )

    outflow_agg = (
        outflow.groupby(outflow_origin, as_index=False)[outflow_people]
        .sum()
        .rename(columns={outflow_origin: "state_fips", outflow_people: "outflow_people"})
    )

    mig = inflow_agg.merge(outflow_agg, on="state_fips", how="outer").fillna(0)
    mig["state_fips"] = pd.to_numeric(mig["state_fips"], errors="coerce")
    mig["net_migration"] = mig["inflow_people"] - mig["outflow_people"]
    mig["year"] = panel_year
    return mig


def build_all_migration() -> pd.DataFrame:
    mig_2021 = build_migration_for_year(FILES["inflow_2021"], FILES["outflow_2021"], 2021)
    mig_2022 = build_migration_for_year(FILES["inflow_2022"], FILES["outflow_2022"], 2022)
    mig_2023 = build_migration_for_year(FILES["inflow_2023"], FILES["outflow_2023"], 2023)

    return pd.concat([mig_2021, mig_2022, mig_2023], ignore_index=True)


def merge_migration(df: pd.DataFrame, migration: pd.DataFrame) -> pd.DataFrame:
    if migration.empty:
        df["inflow_people"] = np.nan
        df["outflow_people"] = np.nan
        df["net_migration"] = np.nan
        return df

    merged = df.merge(
        migration[["state_fips", "year", "inflow_people", "outflow_people", "net_migration"]],
        on=["state_fips", "year"],
        how="left"
    )
    return merged


# =========================================================
# 5. DATA CLEANING / MINING / PANEL PREPARATION
# =========================================================

def prepare_panel_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "state",
        "year",
        "price_index",
        "unemployment_rate",
        "income_change",
        "population",
        "rpp",
    ]
    panel_df = df.dropna(subset=required).copy()
    panel_df["state"] = panel_df["state"].astype("category")
    return panel_df


def build_descriptive_summary(df: pd.DataFrame) -> pd.DataFrame:
    net_mig_corr = np.nan
    if "net_migration" in df.columns and df["net_migration"].notna().any():
        temp = df[["price_index", "net_migration"]].dropna()
        if len(temp) > 1:
            net_mig_corr = float(temp.corr().iloc[0, 1])

    summary_rows = [
        {"metric": "Number of states", "value": df["state"].nunique()},
        {"metric": "Number of years", "value": df["year"].nunique()},
        {"metric": "Number of observations", "value": len(df)},
        {"metric": "Average HPI in 2020", "value": float(df[df["year"] == 2020]["price_index"].mean())},
        {"metric": "Average HPI in 2024", "value": float(df[df["year"] == 2024]["price_index"].mean())},
        {"metric": "Correlation: HPI and RPP", "value": float(df[["price_index", "rpp"]].corr().iloc[0, 1])},
        {"metric": "Correlation: HPI and income change", "value": float(df[["price_index", "income_change"]].corr().iloc[0, 1])},
        {"metric": "Correlation: HPI and unemployment", "value": float(df[["price_index", "unemployment_rate"]].corr().iloc[0, 1])},
        {"metric": "Correlation: HPI and population", "value": float(df[["price_index", "population"]].corr().iloc[0, 1])},
        {"metric": "Correlation: HPI and net migration", "value": net_mig_corr},
    ]
    return pd.DataFrame(summary_rows)


def build_state_period_averages(df: pd.DataFrame) -> pd.DataFrame:
    period_df = df.copy()
    period_df["period"] = ""
    period_df.loc[period_df["year"].between(2020, 2022), "period"] = "COVID Period (2020–2022)"
    period_df.loc[period_df["year"].between(2023, 2024), "period"] = "Post-COVID Period (2023–2024)"
    period_df = period_df[period_df["period"] != ""].copy()

    state_period = period_df.groupby(["period", "state"], as_index=False).agg({
        "price_index": "mean",
        "unemployment_rate": "mean",
        "income_change": "mean",
        "population": "mean",
        "rpp": "mean",
    })
    state_period["state_abbr"] = state_period["state"].map(STATE_ABBR)
    return state_period


# =========================================================
# 6. METHODOLOGY: PANEL REGRESSION
# =========================================================

def run_panel_regression(df: pd.DataFrame):
    formula = "price_index ~ unemployment_rate + income_change + population + rpp + C(state) + C(year)"
    fe_model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["state"]}
    )

    conf = fe_model.conf_int()
    coef_table = pd.DataFrame({
        "term": fe_model.params.index,
        "coef": fe_model.params.values,
        "std_err": fe_model.bse.values,
        "t_or_z": fe_model.tvalues.values,
        "p_value": fe_model.pvalues.values,
        "ci_low": conf[0].values,
        "ci_high": conf[1].values,
    })

    main_vars = ["unemployment_rate", "income_change", "population", "rpp"]
    main_coef_table = coef_table[coef_table["term"].isin(main_vars)].copy().reset_index(drop=True)
    main_coef_table["r2"] = fe_model.rsquared
    main_coef_table["r2_percent"] = fe_model.rsquared * 100
    main_coef_table["adj_r2"] = fe_model.rsquared_adj
    main_coef_table["adj_r2_percent"] = fe_model.rsquared_adj * 100

    fitted_df = df.copy()
    fitted_df["predicted_fe"] = fe_model.predict(df)
    fitted_df["residual_fe"] = fitted_df["price_index"] - fitted_df["predicted_fe"]

    panel_metrics = pd.DataFrame([{
        "model": "Panel Regression (State FE + Year FE)",
        "rmse": rmse(fitted_df["price_index"], fitted_df["predicted_fe"]),
        "mae": float(mean_absolute_error(fitted_df["price_index"], fitted_df["predicted_fe"])),
        "r2": float(r2_score(fitted_df["price_index"], fitted_df["predicted_fe"])),
        "r2_percent": float(r2_score(fitted_df["price_index"], fitted_df["predicted_fe"])) * 100,
        "notes": "In-sample fitted values from fixed-effects panel regression",
    }])

    return main_coef_table, fitted_df, panel_metrics


# =========================================================
# 7. COMPARISON MODEL: RANDOM FOREST
# =========================================================

def run_random_forest_comparison(df: pd.DataFrame):
    rf_df = df.copy()

    X = pd.get_dummies(
        rf_df[["unemployment_rate", "income_change", "population", "rpp", "state", "year"]],
        columns=["state", "year"],
        drop_first=True
    )
    y = rf_df["price_index"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_metrics = pd.DataFrame([{
        "model": "Random Forest (state/year dummies)",
        "rmse": rmse(y_test, rf_pred),
        "mae": float(mean_absolute_error(y_test, rf_pred)),
        "r2": float(r2_score(y_test, rf_pred)),
        "r2_percent": float(r2_score(y_test, rf_pred)) * 100,
        "notes": "Out-of-sample test-set performance",
    }])

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return rf_metrics, feature_importance


# =========================================================
# 8. SAVE MASTER WORKBOOK
# =========================================================

def save_master_workbook(
    raw_df: pd.DataFrame,
    migration_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    state_period_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    panel_coef_df: pd.DataFrame,
    fitted_df: pd.DataFrame,
    panel_metrics: pd.DataFrame,
    rf_metrics: pd.DataFrame,
    rf_importance: pd.DataFrame,
) -> None:
    model_comparison = pd.concat([panel_metrics, rf_metrics], ignore_index=True)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        raw_df.to_excel(writer, sheet_name="MergedDataWithMigration", index=False)
        migration_df.to_excel(writer, sheet_name="MigrationData", index=False)
        panel_df.to_excel(writer, sheet_name="PanelData", index=False)
        state_period_df.to_excel(writer, sheet_name="StatePeriodAverages", index=False)
        panel_coef_df.to_excel(writer, sheet_name="PanelCoefficients", index=False)
        panel_metrics.to_excel(writer, sheet_name="PanelMetrics", index=False)
        rf_metrics.to_excel(writer, sheet_name="RandomForestMetrics", index=False)
        model_comparison.to_excel(writer, sheet_name="ModelComparison", index=False)
        fitted_df.to_excel(writer, sheet_name="FittedValues", index=False)
        rf_importance.to_excel(writer, sheet_name="RFImportance", index=False)


# =========================================================
# 9. MAIN SCRIPT
# =========================================================

def main() -> None:
    # -----------------------------
    # Data collection / loading
    # -----------------------------
    raw_df = load_primary_dataset()

    # -----------------------------
    # Migration data collection and merge
    # -----------------------------
    migration_df = build_all_migration()
    merged_df = merge_migration(raw_df, migration_df)

    # -----------------------------
    # Data cleaning / mining / panel prep
    # -----------------------------
    panel_df = prepare_panel_dataset(merged_df)
    summary_df = build_descriptive_summary(merged_df)
    state_period_df = build_state_period_averages(merged_df)

    # -----------------------------
    # Main methodology: panel regression
    # -----------------------------
    panel_coef_df, fitted_df, panel_metrics = run_panel_regression(panel_df)

    # -----------------------------
    # Comparison model: Random Forest
    # -----------------------------
    rf_metrics, rf_importance = run_random_forest_comparison(panel_df)

    # -----------------------------
    # Save one organized workbook
    # -----------------------------
    save_master_workbook(
        raw_df=merged_df,
        migration_df=migration_df,
        panel_df=panel_df,
        state_period_df=state_period_df,
        summary_df=summary_df,
        panel_coef_df=panel_coef_df,
        fitted_df=fitted_df,
        panel_metrics=panel_metrics,
        rf_metrics=rf_metrics,
        rf_importance=rf_importance,
    )

    print("Analysis complete.")
    print(f"Master workbook saved to: {OUTPUT_FILE}")

    print("\nPanel regression coefficients:")
    print(panel_coef_df.to_string(index=False))

    print("\nPanel regression metrics:")
    print(panel_metrics.to_string(index=False))

    print("\nRandom Forest metrics:")
    print(rf_metrics.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Capstone project analysis script for:
Analyzing the Impact of Economic Factors on Housing Prices in the United States

- 50 U.S. states only (District of Columbia removed)
- Full baseline panel: 2020-2024
- Baseline predictors: unemployment_rate, income_change, population, rpp
- Migration subset: 2021-2023
- Migration built from IRS flow-year files:
    2020-2021 -> panel year 2021
    2021-2022 -> panel year 2022
    2022-2023 -> panel year 2023
- Models:
    1) Multiple Linear Regression
    2) Random Forest Regression
- Outputs:
    merged dataset, figures, coefficients, model comparison CSV

"""

from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =========================================================
# 1. FILE PATHS AND PROJECT SETUP
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FILES = {
    "merged_workbook": BASE_DIR / "economic_data_rpp_analysis.xlsx",
    "rpp_workbook": BASE_DIR / "RPP_2020_2024_clean.xlsx",
    "hpi_csv": BASE_DIR / "hpi_master.csv",
    "unemployment_csv": BASE_DIR / "state_year_unemployment_clean.csv",
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

def first_match(columns: Iterable[str], candidates: Iterable[str]) -> str:
    cols = list(columns)
    lowered = {c.lower(): c for c in cols}
    for candidate in candidates:
        if candidate in cols:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise KeyError(f"Could not find any of these columns: {list(candidates)}. Available: {cols}")


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


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# =========================================================
# 3. DATA LOADING AND CLEANING
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

    required = ["state", "year", "price_index", "unemployment_rate", "income_change", "population", "rpp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    df["state"] = normalize_state_names(df["state"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    for col in ["price_index", "unemployment_rate", "income_change", "population", "rpp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "state_fips" not in df.columns:
        raise KeyError("The merged workbook must include 'state_fips' to merge migration data.")

    df["state_fips"] = pd.to_numeric(df["state_fips"], errors="coerce").astype("Int64")

    # Keep only the 50 states
    df = df[df["state"] != "District of Columbia"].copy()

    return df

# =========================================================
# 4. MIGRATION DATA PROCESSING
# =========================================================

def build_migration_for_year(inflow_path: Path, outflow_path: Path, panel_year: int) -> pd.DataFrame:
    inflow = pd.read_csv(inflow_path)
    outflow = pd.read_csv(outflow_path)

    inflow_origin = first_match(inflow.columns, ["y1_statefips", "statefips1", "origin_state_fips"])
    inflow_dest = first_match(inflow.columns, ["y2_statefips", "statefips2", "destination_state_fips"])
    inflow_people = first_match(inflow.columns, ["n2", "num_returns_exemptions", "exmpt_num"])

    outflow_origin = first_match(outflow.columns, ["y1_statefips", "statefips1", "origin_state_fips"])
    outflow_dest = first_match(outflow.columns, ["y2_statefips", "statefips2", "destination_state_fips"])
    outflow_people = first_match(outflow.columns, ["n2", "num_returns_exemptions", "exmpt_num"])

    # Remove within-state moves
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
    mig["state_fips"] = pd.to_numeric(mig["state_fips"], errors="coerce").astype("Int64")
    mig["net_migration"] = mig["inflow_people"] - mig["outflow_people"]
    mig["year"] = panel_year
    return mig


def build_all_migration() -> pd.DataFrame:
    mig_2021 = build_migration_for_year(FILES["inflow_2021"], FILES["outflow_2021"], 2021)
    mig_2022 = build_migration_for_year(FILES["inflow_2022"], FILES["outflow_2022"], 2022)
    mig_2023 = build_migration_for_year(FILES["inflow_2023"], FILES["outflow_2023"], 2023)

    migration = pd.concat([mig_2021, mig_2022, mig_2023], ignore_index=True)
    migration.to_csv(OUTPUT_DIR / "migration_panel_2021_2023.csv", index=False)
    return migration


def merge_migration(df: pd.DataFrame) -> pd.DataFrame:
    if "net_migration" in df.columns:
        df["net_migration"] = pd.to_numeric(df["net_migration"], errors="coerce")
        return df

    migration = build_all_migration()
    merged = df.merge(
        migration[["state_fips", "year", "inflow_people", "outflow_people", "net_migration"]],
        on=["state_fips", "year"],
        how="left"
    )
    return merged

# =========================================================
# 5. VISUALIZATION FUNCTIONS
# =========================================================

def make_scatter(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, filename: str) -> None:
    plot_df = df[[x, y]].dropna().copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df[x], plot_df[y], alpha=0.7, s=10)

    if len(plot_df) > 1:
        coef = np.polyfit(plot_df[x], plot_df[y], 1)
        x_line = np.linspace(plot_df[x].min(), plot_df[x].max(), 100)
        y_line = coef[0] * x_line + coef[1]
        plt.plot(x_line, y_line, linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    save_plot(OUTPUT_DIR / filename)


def make_line(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, filename: str) -> None:
    plot_df = df[[x, y]].dropna().copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df[x], plot_df[y], marker="o")
    for _, row in plot_df.iterrows():
        plt.text(row[x], row[y] + 4, f"{int(round(row[y]))}", ha="center", fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    save_plot(OUTPUT_DIR / filename)


def make_heatmap(df: pd.DataFrame, filename: str) -> None:
    cols = ["price_index", "rpp", "income_change", "unemployment_rate", "population", "net_migration"]
    available = [c for c in cols if c in df.columns]
    corr_df = df[available].corr()

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    im = ax.imshow(corr_df.values, aspect="auto")
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Correlation Heatmap (50 States)")
    plt.colorbar(im)
    save_plot(OUTPUT_DIR / filename)


def make_top10_rpp(df: pd.DataFrame, filename: str) -> None:
    top10 = (
        df.groupby("state", as_index=False)["rpp"]
        .mean()
        .sort_values("rpp", ascending=False)
        .head(10)
        .sort_values("rpp", ascending=True)
    )

    plt.figure(figsize=(8, 5))
    plt.barh(top10["state"], top10["rpp"])
    plt.title("Top 10 States by Average Cost of Living (RPP)")
    plt.xlabel("Average RPP")
    plt.ylabel("")
    save_plot(OUTPUT_DIR / filename)


def make_spatial_map(df: pd.DataFrame, filename_html: str) -> None:
    try:
        import plotly.express as px
    except ImportError:
        return

    map_df = (
        df.groupby("state", as_index=False)["price_index"]
        .mean()
        .rename(columns={"price_index": "avg_hpi"})
    )
    map_df["abbr"] = map_df["state"].map(STATE_ABBR)
    map_df = map_df.dropna(subset=["abbr"])

    fig = px.choropleth(
        map_df,
        locations="abbr",
        locationmode="USA-states",
        color="avg_hpi",
        scope="usa",
        hover_name="state",
        title="Housing Price Index Across U.S. States"
    )
    fig.write_html(OUTPUT_DIR / filename_html)


def create_all_figures(df: pd.DataFrame) -> None:
    # Correlation heatmap
    heatmap_df = df[df["year"].isin([2021, 2022, 2023])].copy()
    make_heatmap(heatmap_df, "correlation_heatmap_50_states.png")

    # Cost of living vs housing prices
    make_scatter(
        df, "rpp", "price_index",
        "Cost of Living vs Housing Prices",
        "Regional Price Parity", "Housing Price Index",
        "cost_of_living_vs_housing_prices.png"
    )

    # Unemployment vs housing prices
    make_scatter(
        df, "unemployment_rate", "price_index",
        "Unemployment vs Housing Prices",
        "Unemployment Rate", "Housing Price Index",
        "unemployment_vs_housing_prices.png"
    )

    # Income growth vs housing prices
    make_scatter(
        df, "income_change", "price_index",
        "Income Growth vs Housing Prices",
        "Income Change", "Housing Price Index",
        "income_growth_vs_housing_prices.png"
    )

    # Housing price trend
    yearly = df.groupby("year", as_index=False)["price_index"].mean()
    make_line(
        yearly, "year", "price_index",
        "Average Housing Price Trend Over Time",
        "Year", "Average HPI",
        "housing_price_trend.png"
    )

    # Top 10 states by average RPP
    make_top10_rpp(df, "top10_states_by_average_rpp.png")

    # Spatial analysis
    make_spatial_map(df, "spatial_analysis_map.html")

# =========================================================
# 6. MODELING FUNCTIONS
# =========================================================

def fit_and_evaluate(df: pd.DataFrame, features: list[str], label: str):
    model_df = df[features + ["price_index"]].dropna().copy()
    if model_df.empty:
        raise ValueError(f"No usable rows found for model: {label}")

    X = model_df[features]
    y = model_df["price_index"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Multiple Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    lr_metrics = {
        "Model": "Multiple Linear Regression",
        "Dataset": label,
        "RMSE": rmse(y_test, lr_pred),
        "MAE": float(mean_absolute_error(y_test, lr_pred)),
        "R2": float(r2_score(y_test, lr_pred)),
    }

    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": lr.coef_,
    })
    coef_df["intercept"] = float(lr.intercept_)
    coef_df["dataset"] = label

    # Random Forest Regression
    rf = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    rf_metrics = {
        "Model": "Random Forest Regression",
        "Dataset": label,
        "RMSE": rmse(y_test, rf_pred),
        "MAE": float(mean_absolute_error(y_test, rf_pred)),
        "R2": float(r2_score(y_test, rf_pred)),
    }

    return lr_metrics, rf_metrics, coef_df


def run_models(df: pd.DataFrame):
    results = []
    coef_tables = []

    # Baseline model: 2020-2024
    baseline_df = df[df["year"].between(2020, 2024, inclusive="both")].copy()
    baseline_features = ["unemployment_rate", "income_change", "population", "rpp"]
    lr_base, rf_base, coef_base = fit_and_evaluate(
        baseline_df, baseline_features, "2020–2024 (Baseline)"
    )
    results.extend([lr_base, rf_base])
    coef_tables.append(coef_base)

    # Migration subset model: 2021-2023
    migration_df = df[df["year"].isin([2021, 2022, 2023])].copy()
    migration_features = ["unemployment_rate", "income_change", "population", "rpp", "net_migration"]
    lr_mig, rf_mig, coef_mig = fit_and_evaluate(
        migration_df, migration_features, "2021–2023 (Migration Subset)"
    )
    results.extend([lr_mig, rf_mig])
    coef_tables.append(coef_mig)

    results_df = pd.DataFrame(results)
    coef_df = pd.concat(coef_tables, ignore_index=True)

    results_df.to_csv(OUTPUT_DIR / "model_performance_comparison.csv", index=False)
    coef_df.to_csv(OUTPUT_DIR / "linear_regression_coefficients.csv", index=False)

    return results_df, coef_df

# =========================================================
# 7. SUMMARY OUTPUTS
# =========================================================

def make_summary_text(df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    corr_cols = ["price_index", "rpp", "income_change", "unemployment_rate", "population", "net_migration"]
    corr_df = df[corr_cols].corr(numeric_only=True)

    summary_lines = []
    summary_lines.append("Key summary from automated analysis\n")
    summary_lines.append(f"Average HPI in 2020: {df[df['year'] == 2020]['price_index'].mean():.0f}")
    summary_lines.append(f"Average HPI in 2024: {df[df['year'] == 2024]['price_index'].mean():.0f}")
    summary_lines.append(f"Correlation of HPI with RPP: {corr_df.loc['price_index', 'rpp']:.2f}")
    summary_lines.append(f"Correlation of HPI with income change: {corr_df.loc['price_index', 'income_change']:.2f}")
    summary_lines.append(f"Correlation of HPI with unemployment: {corr_df.loc['price_index', 'unemployment_rate']:.2f}")
    if "net_migration" in corr_df.columns:
        summary_lines.append(f"Correlation of HPI with migration: {corr_df.loc['price_index', 'net_migration']:.2f}")

    summary_lines.append("\nModel comparison")
    summary_lines.append(results_df.to_string(index=False))

    (OUTPUT_DIR / "analysis_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

# =========================================================
# 8. MAIN SCRIPT
# =========================================================

def main() -> None:
    # Load and merge data
    df = load_primary_dataset()
    df = merge_migration(df)

    # Save final merged analysis dataset
    df.to_csv(OUTPUT_DIR / "merged_analysis_dataset.csv", index=False)

    # Create all visualizations
    create_all_figures(df)

    # Run models and save outputs
    results_df, coef_df = run_models(df)

    # Save written summary
    make_summary_text(df, results_df)

    # Print summary to console
    print("Analysis complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("\nModel performance:")
    print(results_df.to_string(index=False))
    print("\nLinear regression coefficients:")
    print(coef_df.to_string(index=False))


if __name__ == "__main__":
    main()

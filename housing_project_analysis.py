# =========================================================
# 8A. ACTUAL VS PREDICTED PANEL REGRESSION CHART
# =========================================================

def save_actual_vs_predicted_chart(fitted_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plot_df = fitted_df.copy()

    plt.figure(figsize=(12, 9))
    plt.scatter(plot_df["price_index"], plot_df["predicted_fe"], alpha=0.75, s=22)

    if "state" in plot_df.columns:
        for _, row in plot_df.iterrows():
            plt.text(
                row["price_index"],
                row["predicted_fe"],
                str(row["state"]),
                fontsize=6,
                alpha=0.5
            )

    min_val = min(plot_df["price_index"].min(), plot_df["predicted_fe"].min())
    max_val = max(plot_df["price_index"].max(), plot_df["predicted_fe"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2)

    rmse_val = rmse(plot_df["price_index"], plot_df["predicted_fe"])
    mae_val = float(mean_absolute_error(plot_df["price_index"], plot_df["predicted_fe"]))
    r2_val = float(r2_score(plot_df["price_index"], plot_df["predicted_fe"]))

    plt.xlabel("Actual HPI")
    plt.ylabel("Predicted HPI")
    plt.title("Actual vs Predicted HPI (Panel Regression with State Labels)")
    plt.text(
        0.05, 0.95,
        f"RMSE = {rmse_val:.2f}\nMAE = {mae_val:.2f}\nR² = {r2_val:.2f}\nR² = {r2_val*100:.2f}%",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )
    plt.tight_layout()
    plt.savefig(BASE_DIR / "actual_vs_predicted_hpi_panel_regression.png", dpi=220, bbox_inches="tight")
    plt.close()

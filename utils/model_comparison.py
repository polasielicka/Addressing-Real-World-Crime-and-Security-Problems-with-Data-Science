import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
from pathlib import Path

def evaluate_models():
    """
    Compare the two models' predictions against each other and against actual values.
    """
    print("Evaluating models...")

    # get directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    output_dir = os.path.abspath(os.path.join(script_dir, '..', 'output'))

    path1 = os.path.join(output_dir, "results.csv")
    path2 = os.path.join(repo_root, "Pola", "results", "ward_backtest_forecasts.csv")
    output_path = os.path.join(output_dir, "model_comparison.csv")

    # Load CSVs
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    # Clean and rename df1
    df1 = df1.rename(columns={
        "ward_name": "ward",
        "month_num": "month",
        "predicted_burglaries": "pred1",
        "actual_burglaries": "actual"
    })
    df1["ward"] = df1["ward"].astype(str).str.replace(r"\s*ward$", "", case=False, regex=True).str.strip()
    df1["month"] = df1["month"].astype(int)

    # Clean and rename df2
    df2 = df2.rename(columns={"predicted": "pred2"})
    df2["ward"] = df2["ward"].astype(str).str.strip()
    df2["month"] = pd.to_datetime(df2["month"], errors="coerce").dt.month

    # Merge on ward and month
    merged = pd.merge(
        df1[["ward", "month", "pred1", "actual"]],
        df2[["ward", "month", "pred2"]],
        on=["ward", "month"],
        how="inner"
    )

    if merged.empty:
        print("No matching (ward, month) entries found.")
    else:
        merged = merged.dropna(subset=["pred1", "pred2", "actual"])
        
        # Compute model-to-model similarity
        merged["abs_diff"] = (merged["pred1"] - merged["pred2"]).abs()
        merged["rel_diff_pct"] = merged["abs_diff"] / merged[["pred1", "pred2"]].mean(axis=1) * 100
        correlation_models = merged["pred1"].corr(merged["pred2"])
        mean_abs_diff = merged["abs_diff"].mean()
        mean_rel_diff = merged["rel_diff_pct"].mean()

        # Evaluate models vs actual
        mae1 = mean_absolute_error(merged["actual"], merged["pred1"])
        mae2 = mean_absolute_error(merged["actual"], merged["pred2"])
        corr1 = merged["actual"].corr(merged["pred1"])
        corr2 = merged["actual"].corr(merged["pred2"])

        print("\nModel-to-Model Similarity:")
        print(f"Correlation (pred1 vs pred2): {correlation_models:.3f}")
        print(f"Mean Absolute Difference: {mean_abs_diff:.2f} burglaries")
        print(f"Mean Relative Difference: {mean_rel_diff:.1f}%")

        print("\nModel Accuracy vs Actual:")
        print(f"Model 1 (XGBoost) MAE: {mae1:.2f}, Correlation: {corr1:.3f}")
        print(f"Model 2 (SARIMA) MAE: {mae2:.2f}, Correlation: {corr2:.3f}")

        # Save output
        merged.to_csv(output_path, index=False)
        print(f"\nOutput saved to: {output_path}")

def main():
    evaluate_models()

if __name__ == "__main__":
    main()
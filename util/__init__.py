import pandas as pd

def nan_ratio_report(
    df: pd.DataFrame,
    sort: bool = True,
    with_row_stats: bool = True,
):

    total_rows = len(df)
    if total_rows == 0:
        col_nan_df = pd.DataFrame(columns=["column", "nan_ratio"])
        row_stats = {
            "rows_nan_ge_30": 0,
            "rows_nan_ge_60": 0,
            "rows_nan_ge_90": 0,
            "rows_nan_all": 0,
        }
        return col_nan_df, row_stats if with_row_stats else col_nan_df


    nan_ratios = df.isna().sum() / total_rows
    col_nan_df = (
        nan_ratios.reset_index()
        .rename(columns={"index": "column", 0: "nan_ratio"})
    )
    if sort:
        col_nan_df = col_nan_df.sort_values("nan_ratio", ascending=False).reset_index(drop=True)

    if not with_row_stats:
        return col_nan_df


    total_cols = df.shape[1]
    if total_cols == 0:
        row_stats = {
            "rows_nan_ge_30": 0,
            "rows_nan_ge_60": 0,
            "rows_nan_ge_90": 0,
            "rows_nan_all": 0,
        }
        return col_nan_df, row_stats

    nan_count_per_row = df.isna().sum(axis=1)
    frac_nan_per_row = nan_count_per_row / total_cols

    row_stats = {
        "rows_nan_ge_30": int((frac_nan_per_row >= 0.30).sum()),
        "rows_nan_ge_60": int((frac_nan_per_row >= 0.60).sum()),
        "rows_nan_ge_90": int((frac_nan_per_row >= 0.90).sum()),
        "rows_nan_all":   int((frac_nan_per_row == 1.00).sum()),
    }

    return col_nan_df, row_stats

def clean_dataframe(
    df: pd.DataFrame,
    dropna_cols: list[str] | None = None,
    outlier_rules: dict[str, tuple[float, float]] | None = None,
    quantile_mode: bool = False,
) -> pd.DataFrame:


    df_clean = df.copy()


    if dropna_cols:
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=dropna_cols)
        print(f"[NaN] Dropped {before - len(df_clean)} rows with NaN in {dropna_cols}")


    if outlier_rules:
        for col, bounds in outlier_rules.items():
            if col not in df_clean.columns:
                print(f"[Skip] Column '{col}' not found in DataFrame.")
                continue

            lower, upper = bounds
            if quantile_mode:
                lower_val, upper_val = df_clean[col].quantile([lower, upper])
            else:
                lower_val, upper_val = lower, upper

            before = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_val) & (df_clean[col] <= upper_val)]
            after = len(df_clean)

            print(f"[Outlier] {col}: kept in range ({lower_val:.3f}, {upper_val:.3f}), "
                  f"removed {before - after} rows.")

    return df_clean.reset_index(drop=True)
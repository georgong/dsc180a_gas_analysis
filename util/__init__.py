import pandas as pd

def nan_ratio_report(
    df: pd.DataFrame,
    sort: bool = True,
    with_row_stats: bool = True,
):
    """
    返回:
    - col_nan_df: 每个列的 NaN 比例 (0~1)，按比例排序（可选）
    - row_stats: 行级 NaN 分布统计（可选）

    row_stats 包含:
      - rows_nan_ge_30:  NaN 比例 >= 30% 的行数
      - rows_nan_ge_60:  NaN 比例 >= 60% 的行数
      - rows_nan_ge_90:  NaN 比例 >= 90% 的行数
      - rows_nan_all:    全 NaN 行数
    """
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

    # ------ 列级 NaN 比例 ------
    nan_ratios = df.isna().sum() / total_rows
    col_nan_df = (
        nan_ratios.reset_index()
        .rename(columns={"index": "column", 0: "nan_ratio"})
    )
    if sort:
        col_nan_df = col_nan_df.sort_values("nan_ratio", ascending=False).reset_index(drop=True)

    if not with_row_stats:
        return col_nan_df

    # ------ 行级 NaN 比例 ------
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
    """
    清洗 DataFrame：
      1. 删除指定列中存在 NaN 的行；
      2. 按指定范围过滤掉 outliers。

    参数：
      df : pd.DataFrame
          输入表。
      dropna_cols : list[str]
          哪些列要求无 NaN；为 None 时不执行 dropna。
      outlier_rules : dict[str, tuple[float, float]]
          指定列及其取值范围，形如：
          {"duration": (0, 100), "delay": (-10, 10)}
          若 quantile_mode=True，则区间值视为分位数。
      quantile_mode : bool
          True → 把范围解释为 quantile（如 (0.01, 0.99)）
          False → 把范围解释为实际值区间。
    返回：
      清洗后的 DataFrame（索引重置）
    """

    df_clean = df.copy()

    # --- Step 1: dropna ---
    if dropna_cols:
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=dropna_cols)
        print(f"[NaN] Dropped {before - len(df_clean)} rows with NaN in {dropna_cols}")

    # --- Step 2: outlier filtering ---
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
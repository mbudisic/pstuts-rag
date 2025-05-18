import pandas as pd
import numpy as np
import functools


def with_summary(df, mean_label="Mean", std_label="StdDev"):
    """
    Adds summary rows (mean and standard deviation) to a pandas DataFrame.

    This function calculates the mean of numeric columns and estimates the standard
    deviation using a jackknife resampling method. It then appends these summary
    statistics as new rows to the original DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to summarize
        mean_label (str, optional): Label for the mean row. Defaults to "Mean".
        std_label (str, optional): Label for the standard deviation row. Defaults to "StdDev".

    Returns:
        pd.DataFrame: Original DataFrame with summary rows appended

    Raises:
        ValueError: If the DataFrame has fewer than 2 rows (needed for jackknife estimation)
    """
    # 1. Identify numeric vs non-numeric columns
    df = df.infer_objects()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.columns.difference(numeric_cols)

    n = len(df)
    if n < 2:
        raise ValueError("Need at least two rows for jackknife estimate")

    # 2. Compute the plain mean
    means = df[numeric_cols].mean()

    # 3. Compute jackknife leave-one-out std dev
    jackknife_std = {}
    # total sums up front for speed
    col_sums = df[numeric_cols].sum()
    for col in numeric_cols:
        x = df[col].values
        # leave-one-out means: (sum(x) - x[i])/(n-1)
        loo_means = (col_sums[col] - x) / (n - 1)
        # jackknife variance formula: (n-1)/n * Σ (θ_i – θ̄)²
        theta = loo_means.mean()
        var = (n - 1) / n * np.sum((loo_means - theta) ** 2)
        jackknife_std[col] = np.sqrt(var)

    jackknife_std = pd.Series(jackknife_std)

    # 4. Build a 2×cols DataFrame for the summary rows
    summary = pd.DataFrame(index=[mean_label, std_label], columns=df.columns)

    # fill numeric summaries
    for col in numeric_cols:
        summary.at[mean_label, col] = means[col]
        summary.at[std_label, col] = jackknife_std[col]
    # fill non-numeric with labels
    for col in non_numeric_cols:
        summary.at[mean_label, col] = mean_label
        summary.at[std_label, col] = std_label

    # 5. Concatenate and return
    retval = pd.concat([df, summary], axis=0)
    retval = retval.apply(
        functools.partial(pd.to_numeric, errors="ignore")
    )
    retval = retval.infer_objects()
    return retval

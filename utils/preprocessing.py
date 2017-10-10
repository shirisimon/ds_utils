import pandas as pd
import numpy as np

def distribution_imputation_by_class(missing_vals_df, missing_vals_col, target_df, target_col, key_col):
    """
    impute missing values based on the distribution of the values in the example target label.
    E.g. for a customer with missing age that with a target label = 1, the imputed age will be drawn
    from the distribution of ages for target label = 1
    :param missing_vals_df: pandas DataFrame. include the column to impute (i.e. the context table)
    :param missing_vals_col: String. name of column to impute
    :param target_df: pandas DataFrame. include the target column (i.e. the main)
    :param target_col: String. Name of target column
    :param key_col: String. Name of context key column
    :return:
    """
    np.random.seed(0)
    filled = pd.DataFrame()
    for part_label in target_df[target_col].unique():
        target_part = target_df[target_df[target_col] == part_label]
        df2hist = missing_vals_df[missing_vals_df[key_col].isin(target_part[key_col].unique())].dropna()
        hist = df2hist.groupby(missing_vals_col).apply(lambda x: len(x)) / float(
            df2hist.groupby(missing_vals_col).apply(lambda x: len(x)).sum())
        missing_vals_part = missing_vals_df[missing_vals_df[key_col].isin(target_part[key_col].unique())]
        merged_part = pd.merge(target_part, missing_vals_part, how='outer', on=key_col)[[key_col, missing_vals_col]]
        merged_part = merged_part.drop_duplicates()
        part2impute = merged_part[merged_part[missing_vals_col].isnull()]
        part2impute[missing_vals_col] = np.random.choice(hist.index, part2impute.shape[0], p=hist)
        part_filled = pd.concat([part2impute, merged_part[np.isfinite(merged_part[missing_vals_col])]], axis=0)
        filled = pd.concat([filled, part_filled], axis=0)
    return filled

def downsample_context_chunks_by_key(sampled_main_df, context_df, key_column):
    pass
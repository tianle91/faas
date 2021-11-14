import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_target_scatter(
    df_prediction: pd.DataFrame,
    df_actual: pd.DataFrame,
    target_column: str,
    date_column: str,
) -> Figure:

    PRED_TARGET_COL = '__PRED_TARGET__'
    pred = df_prediction[[date_column, target_column]].rename(columns={
        target_column: PRED_TARGET_COL})

    ACTUAL_TARGET_COL = '__ACTUAL_TARGET__'
    actual = df_actual[[date_column, target_column]].rename(columns={
        target_column: ACTUAL_TARGET_COL})

    joined = actual.merge(right=pred, how='left', on=date_column)
    joined = joined.sort_values(by=date_column)

    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(joined[date_column], joined[PRED_TARGET_COL], label='Predicted', alpha=.5)
        ax.plot(joined[date_column], joined[ACTUAL_TARGET_COL], label='Actual', alpha=.5)
        ax.set_title(target_column)
        ax.set_ylabel(target_column)
        ax.legend()

    return fig

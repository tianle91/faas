import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_target_scatter(
    df_prediction: pd.DataFrame,
    df_actual: pd.DataFrame,
    target_column: str
) -> Figure:
    pred = df_prediction[target_column].to_list()
    actual = df_actual[target_column].to_list()
    all_vals = pred + actual
    min_val, max_val = min(all_vals), max(all_vals)
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(5, 5))
        v = sorted(all_vals)
        ax.plot(v, v, ls=':', color='red')

        ax.scatter(x=actual, y=pred, alpha=.5)
        ax.set_xlabel('Actual')
        ax.set_xlim(min_val, max_val)
        ax.set_ylabel('Predicted')
        ax.set_ylim(min_val, max_val)
        ax.set_title(target_column)
    return fig
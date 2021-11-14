from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_day_of_week(df: pd.DataFrame, date_column: str, target_column: str) -> Figure:

    df = df[[date_column, target_column]].sort_values(by=date_column)

    DAY_OF_WEEK_COL = '__DAY_OF_WEEK__'
    df[DAY_OF_WEEK_COL] = df[date_column].apply(lambda dt: dt.weekday())
    START_OF_WEEK_COL = '__START_OF_WEEK__'
    df[START_OF_WEEK_COL] = df[date_column].apply(lambda dt: dt - timedelta(days=dt.weekday()))

    with plt.xkcd():
        fig, ax = plt.subplots()
        for start_of_week_dt, subdf in df.groupby(START_OF_WEEK_COL):
            dow, tgt = subdf[DAY_OF_WEEK_COL], subdf[target_column]
            ax.plot(dow, tgt, label=start_of_week_dt)
        ax.legend()
    return fig


def plot_day_of_month(df: pd.DataFrame, date_column: str, target_column: str) -> Figure:
    pass


def plot_day_of_year(df: pd.DataFrame, date_column: str, target_column: str) -> Figure:
    pass


def plot_week_of_year(df: pd.DataFrame, date_column: str, target_column: str) -> Figure:
    pass

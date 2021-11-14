from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

DOW_MAPPING = {
    0: 'Mon',
    1: 'Tue',
    2: 'Wed',
    3: 'Thu',
    4: 'Fri',
    5: 'Sat',
    6: 'Sun',
}


def plot_day_of_week(df: pd.DataFrame, date_column: str, target_column: str) -> Figure:
    df = df[[date_column, target_column]].sort_values(by=date_column)
    DAY_OF_WEEK_COL = '__DAY_OF_WEEK__'
    df[DAY_OF_WEEK_COL] = df[date_column].apply(lambda dt: dt.weekday())
    START_OF_WEEK_COL = '__START_OF_WEEK__'
    df[START_OF_WEEK_COL] = df.apply(
        lambda row: row[date_column] - timedelta(days=row[DAY_OF_WEEK_COL]),
        axis=1
    )

    with plt.xkcd():
        fig, ax = plt.subplots()
        x = []
        labels = []
        for dow, subdf in df.groupby(DAY_OF_WEEK_COL):
            x.append(subdf[target_column])
            labels.append(DOW_MAPPING[dow])
        ax.boxplot(x=x, labels=labels)
        ax.set_title(f'{target_column} by Day of Week')
        ax.set_ylabel(target_column)
    return fig

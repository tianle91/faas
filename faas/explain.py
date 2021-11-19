import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMModel
from matplotlib.figure import Figure


def plot_feature_importances(m: LGBMModel, top_n: int = 10) -> Figure:
    df = pd.DataFrame({
        'name': m.feature_name_,
        'importance': m.feature_importances_,
    }).sort_values('importance', ascending=False)
    top_df = df.iloc[:top_n]
    with plt.xkcd():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x=top_df['name'], height=top_df['importance'])
        ax.tick_params(labelrotation=90)
        ax.set_title('Feature Importances')
    return fig

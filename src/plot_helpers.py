import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from IPython.display import display
FIGSIZE = (6, 4)


def fraction_plot(df, col, title):
    plt.figure(figsize=FIGSIZE)
    vals = df.iloc[:, col].copy()
    na_count = vals.isna().sum()
    vals_non_na = vals.dropna()
    vc = vals_non_na.value_counts(normalize=True)
    # Add NA as a separate class
    if na_count > 0:
        vc['NA'] = na_count / len(vals)
    sns.barplot(y=vc.index, x=vc.values, orient='h')
    plt.ylabel('Fraction of Participants')
    plt.xlabel(None)
    plt.title(title)
    plt.tight_layout()

def drop_parenthesis(text):
    return re.sub(r'\([^)]*\)', '', text)


def df_display(df, i_col):
    df = pd.DataFrame(df.iloc[:, i_col]).dropna()
    display(df)



def plot_dist_by_age(df_num, age_col='What is your age?', figsize=(12, 12)):
    plot_cols = [col for col in df_num.columns if col != age_col]
    num_plots = len(plot_cols)
    num_rows = (num_plots + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, col in zip(axes, plot_cols):
        df_plot = df_num[[age_col, col]].dropna()
        sns.boxplot(x=col, y=age_col, data=df_plot, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('Age')
        title_str = col
        if len(title_str) > 65:
            break_idx = title_str.rfind(' ', 0, 65)
            break_idx = break_idx if break_idx != -1 else 65
            title_str = title_str[:break_idx] + '\n' + title_str[break_idx+1:]
        ax.set_title(title_str)

    for ax in axes[len(plot_cols):]:
        ax.set_visible(False)

    fig.suptitle("Age Distribution by Rating (1- very low; 5 - very high)", fontsize=16)
    plt.tight_layout()
    plt.show()
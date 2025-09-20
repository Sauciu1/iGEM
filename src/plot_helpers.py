import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
from IPython.display import display
from scipy.stats import pearsonr
from itertools import combinations
import textwrap
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


def compute_corr_p(df):

    cols = df.columns
    n = len(cols)
    corr = pd.DataFrame(np.eye(n), index=cols, columns=cols, dtype=float)
    pvals = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for a, b in combinations(cols, 2):
        x = df[a]
        y = df[b]
        mask = x.notna() & y.notna()
        if mask.sum() > 1:
            r, p = pearsonr(x[mask], y[mask])
        else:
            r, p = np.nan, np.nan
        corr.loc[a, b] = corr.loc[b, a] = r
        pvals.loc[a, b] = pvals.loc[b, a] = p

    # Bonferroni correction for multiple comparisons
    m = n * (n - 1) // 2  # number of unique pairs
    pvals_corrected = pvals * m
    pvals_corrected = pvals_corrected.clip(upper=1.0)
    pvals[:] = pvals_corrected
    
    return corr, pvals

def format_annotation(corr_df, p_df):
    import numpy as np
    def fmt_p(v):
        if np.isnan(v):
            return ""
        if v < 1e-3:
            return "p<0.001"
        return f"p={v:.3f}".rstrip('0').rstrip('.')
    return corr_df.round(2).astype(str) + "\n" + p_df.applymap(fmt_p)

def plot_corr_heatmap(corr_matrix, pval_matrix, title="Pearson Correlation Triangle", figsize=(14,8)):

    # Remove first row and last column for strict lower triangle
    corr_plot = corr_matrix.iloc[1:, :-1]
    pval_plot = pval_matrix.iloc[1:, :-1]
    annot = format_annotation(corr_plot, pval_plot)
    mask = ~np.tril(np.ones_like(corr_plot, dtype=bool))
    wrap_label = lambda s, w=40: "\n".join(textwrap.wrap(s, w)) if len(s) > w else s
    plt.figure(figsize=figsize)
    with sns.axes_style("white"):
        ax = sns.heatmap(
            corr_plot,
            annot=annot,
            mask=mask,
            fmt='',
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0,
            cbar_kws={'shrink': 0.8}
        )
        ax.set_xticklabels([wrap_label(t.get_text(), 40) for t in ax.get_xticklabels()], rotation=90, ha='center')
        ax.set_yticklabels([wrap_label(t.get_text(), 40) for t in ax.get_yticklabels()], rotation=0, va='center')
        ax.set_title(title)

    return ax
# %%
import pandas as pd
from pandas.api.types import CategoricalDtype
import os, re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from models import GatedAttentionMIL, MultiHeadGatedAttentionMIL
import torch

os.chdir("users/project1/pt01190/TOMPEI-CMMD/code")
results_dir = "../results/results_quant"

pattern = re.compile(r"(?P<metric>.+)_MCDO-(?P<id>\d+)\.pkl")

records = []

for filename in os.listdir(results_dir):
    match = pattern.match(filename)
    if match:
        metric = match.group("metric")
        id_ = int(match.group("id"))
        file_path = os.path.join(results_dir, filename)
        if os.path.getsize(file_path) == 0:
            continue
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        records.append({
            "metric": metric,
            "id": id_,
            "value": data
        })

df = pd.DataFrame(records)

metric_rename = {
    "relevance_rank_accuracy": "Relevance Rank Accuracy",
    "topkintersection": "Top-K Intersection",
    "sparseness": "Sparseness",
    "faithfulness_correlation": "Faithfulness Correlation",
    "mprt": "Mean Perturbation Relevance Test"
}

df["metric"] = df["metric"].replace(metric_rename)

id_rename = {
    490: 'SH#224@0.50',
    493: 'SH#128@0.50',
    497: 'SH#128@0.75',
    491: 'MH#224@0.50',
    492: 'MH#128@0.50',
    496: 'MH#128@0.75',
}
# 490 - SH#224@0.50
# 493 - SH#128@0.50
# 497 - SH#128@0.75
# 491 - MH#224@0.50
# 492 - MH#128@0.50
# 496 - MH#128@0.75

df['models'] = df['id'].replace(id_rename)


print(df.head())

rows = []
for _, row in df.iterrows():
    for label in row['value']:
        for val in row['value'][label]:
            rows.append({
                'metric': row['metric'],
                'id': row['id'],
                'label': label,
                'score': val
            })

flat_df = pd.DataFrame(rows)

order = list(id_rename.values())
flat_df['model_name'] = flat_df['id'].map(id_rename)
flat_df['model_name'] = flat_df['model_name'].astype(CategoricalDtype(categories=order, ordered=True))

# %%
def plot_single_metric_boxplot(flat_df, metric_name, use_labels=True):
    """
    Plots a stylized boxplot for a single metric across models, optionally using label-based distinction (hatches).

    Parameters:
        flat_df (pd.DataFrame): DataFrame with columns ['metric', 'id', 'score', 'label']
        metric_name (str): Name of the metric to plot (as appears in flat_df['metric'])
        use_labels (bool): Whether to split by 'label' (positive/negative) and apply hatches
    """

    df_plot = flat_df[flat_df['metric'] == metric_name].copy()

    if df_plot.empty:
        print(f"[Warning] No data found for metric: {metric_name}")
        return

    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(
        x='model_name',
        y='score',
        hue='label' if use_labels else None,
        hue_order=['positive', 'negative'] if use_labels else None,
        data=df_plot,
        fill=True,
        linewidth=0.5,
        linecolor='black',
        medianprops={'color': 'red', 'linewidth': 3},
        width=0.4,
    )

    plt.xlabel('')
    plt.ylabel("Score", fontsize=16)
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(False)

    ax.grid(axis='y', linestyle=':', alpha=0.8)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    for patch in ax.patches:
        patch.set_facecolor('white')

    if use_labels and 'label' in df_plot.columns:
        labels_present = df_plot['label'].unique().tolist()
        num_ids = df_plot['id'].nunique()

        hatch_map = {'positive': '//', 'negative': ' '}
        hatches = []

        for i in range(num_ids):
            for label in labels_present:
                hatches.append(hatch_map.get(label, ''))

        for pat, bar in zip(hatches, ax.patches):
            bar.set_hatch(pat)

        legend_handles = [
            mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Positive'),
            mpatches.Patch(facecolor='white', edgecolor='black', hatch=' ', label='Negative'),
        ]
        plt.legend(
            handles=legend_handles,
            title='Label',
            loc='center',
            bbox_to_anchor=(0.1, 1.05),
            fontsize=12,
            title_fontsize=14,
            ncol=2,
            frameon=True
            )

        bars = ax.patches
        hatches = ['//'] * 6 + [' '] * 6
        for pat, bar in zip(hatches, bars):
            bar.set_hatch(pat)

    else:
        if ax.get_legend():
            ax.get_legend().remove()
        bars = ax.patches
        hatches = ['//'] * len(bars)
        for pat, bar in zip(hatches, bars):
            bar.set_hatch(pat)

    plt.title(metric_name, fontsize=18, pad=10)
    plt.xticks(rotation=45)  # or rotation=30, 60, 90, etc. as needed
    plt.show()


plot_single_metric_boxplot(flat_df, "Sparseness", use_labels=True)
plot_single_metric_boxplot(flat_df, "Top-K Intersection", use_labels=False)
plot_single_metric_boxplot(flat_df, "Relevance Rank Accuracy", use_labels=False)




df = df[df["metric"] == "Mean Perturbation Relevance Test"].copy()

model_SH = GatedAttentionMIL()
model_MH = MultiHeadGatedAttentionMIL()

layer_order_SH = [
    name for name, module in model_SH.named_modules()
    if not isinstance(module, torch.nn.Sequential) and name != ""
]
layer_order_MH = [
    name for name, module in model_MH.named_modules()
    if not isinstance(module, torch.nn.Sequential) and name != ""
]

def flatten_scores(row):
    records = []
    model_name = row["models"]
    for polarity in ["positive"]: #, "negative"]:
        for item in row["value"][polarity]:
            flat = {"models": model_name, "score_type": polarity}
            for layer, score in item.items():
                if layer != "original":
                    flat[layer] = float(score[0])
            records.append(flat)
    return records

flat_all = [item for _, row in df.iterrows() for item in flatten_scores(row)]
df_flat = pd.DataFrame(flat_all)

df_long = df_flat.melt(id_vars=["models", "score_type"], var_name="layer", value_name="score")

for model in df_long["models"].unique():
    sub_df = df_long[df_long["models"] == model].copy()

    layer_order = layer_order_SH if "SH" in model else layer_order_MH

    sub_df['layer'] = sub_df['layer'].str.replace('^model\.', '', regex=True)
    used_layers = [l for l in layer_order if l in sub_df["layer"].unique()]
    sub_df["layer"] = pd.Categorical(sub_df["layer"], categories=used_layers, ordered=True)
    sub_df = sub_df.sort_values("layer")

    plt.figure(figsize=(14, 5))
    sns.lineplot(data=sub_df, x="layer", y="score", hue="score_type", marker="o")
    plt.title(f"MPRT Scores per Layer — Model {model}")
    plt.xlabel("Layer")
    plt.ylabel("MPRT Score")
    plt.xticks(rotation=60, ha="right")
    plt.legend(title="Score Type")
    plt.tight_layout()
    plt.show()

# %%import pandas as pd

df_positive = df_long[df_long['score_type'] == 'positive']

summary_positive = df_positive.groupby('models').agg(
    mean_score=('score', 'mean'),
    std_score=('score', 'std'),
    min_score=('score', 'min'),
    max_score=('score', 'max'),
    median_score=('score', 'median'),
    count=('score', 'count')
).reset_index()

print(summary_positive)

# %%
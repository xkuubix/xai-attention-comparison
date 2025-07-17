# %%
import pandas as pd
import os, re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['ps.fonttype'] = 3  # eps @ problem in latex
import pickle
import neptune
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

results_dir = "cv_results"

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

metrics_df = pd.DataFrame(records)

metric_rename = {
    "relevance_rank_accuracy": "Relevance Rank Accuracy",
    "topkintersection": "Top-K Intersection",
    "sparseness": "Sparseness",
    "faithfulness_correlation": "Faithfulness Correlation",
    "avg_sensitivity": "Average Sensitivity",
    # "mprt": "Mean Perturbation Relevance Test"
}

metrics_df["metric"] = metrics_df["metric"].replace(metric_rename)



project = neptune.init_project(project="ProjektMMG/MCDO")

runs_table_df = project.fetch_runs_table(
    id=[f"MCDO-{id}" for id in range(517, 565)],
    owner="jakub-buler",
    state="inactive",
    trashed=False,
    ).to_pandas()

runs_df = runs_table_df.copy()
runs_df["id"] = runs_df["sys/name"].str.replace("MCDO-", "")

def extract_variant_and_fold(path):
    # Example path: /.../SH_128_at_0.75/fold_5/...
    match = re.search(r'/(SH|MH)_(\d+)_at_(\d\.\d+)/fold_(\d+)/', path)
    if match:
        variant, patch, overlap, fold = match.groups()
        variant_name = f"{variant}#{patch}@{str(overlap).replace('.', '')}"
        return pd.Series([variant_name, int(fold)])
    else:
        return pd.Series([None, None])

runs_df[["training_variant", "fold"]] = runs_df["best_model_path"].apply(extract_variant_and_fold)
from io import StringIO

def parse_classification_report(report_str):
    try:
        df = pd.read_fwf(StringIO(report_str), index_col=0)
        df.index = df.index.str.strip()  # clean whitespace
        return df
    except Exception as e:
        print(f"Error parsing report: {e}")
        return pd.DataFrame()
def flatten_report_df(df):
    flat = {}
    for row in df.index:
        for col in df.columns:
            flat[f"{row}/{col}"] = df.loc[row, col]
    return pd.Series(flat)


report_tables = runs_df["test/classification_report"].apply(parse_classification_report)
report_flat = report_tables.apply(flatten_report_df)

meta_df = pd.DataFrame()

meta_df["id"] = runs_df["sys/name"].str.replace("MCDO-", "", regex=False)
variant_parts = runs_df["best_model_path"].str.extract(r"/([A-Z]+)_([0-9]+)_at_([0-9.]+)/")
meta_df["training_variant"] = (
    variant_parts[0] + " #" + variant_parts[1] + 
    variant_parts[2].apply(lambda x: f" @{float(x):.2f}")
)
meta_df["fold"] = runs_df["best_model_path"].str.extract(r"/fold_(\d+)/")[0].astype(int)
report_df = pd.concat([meta_df, report_flat], axis=1)
report_df['id'] = report_df['id'].astype(int)

merged_df = pd.merge(metrics_df, report_df, on="id", how="left")


def unpack_dict(row):
    if isinstance(row['value'], dict):
        # Flatten list of lists for 'positive' and 'negative'
        positive = [item[0] for item in row['value'].get('positive', []) if isinstance(item, list) and item]
        negative = [item[0] for item in row['value'].get('negative', []) if isinstance(item, list) and item]

        return pd.Series({
            'score_positive': np.array(positive),
            'score_negative': np.array(negative)
        })
    else:
        return pd.Series({
            'score_positive': np.array(row['value']),
            'score_negative': None
        })

flat_df = merged_df.join(merged_df.apply(unpack_dict, axis=1))

metric_cols = [
    # 'Negative/precision', 'Negative/recall', 'Negative/f1-score',
    # 'Positive/precision', 'Positive/recall', 'Positive/f1-score',
    'accuracy/precision', 'accuracy/recall', 'accuracy/f1-score',
    'macro avg/precision', 'macro avg/recall', 'macro avg/f1-score',
    'weighted avg/precision', 'weighted avg/recall', 'weighted avg/f1-score'
]
valid_metric_cols = [col for col in metric_cols if not flat_df[col].isna().all()]
df = (
    flat_df.groupby('training_variant')[valid_metric_cols]
    .agg(['mean', 'std'])  # Get mean and standard deviation
    .round(3)               # Round for display
)
print(df)
# %%
def plot_single_metric_boxplot(flat_df, metric_name, use_labels=True):
    """
    Plots stylized heatmaps for both positive and negative scores for a single metric across models.

    Parameters:
        flat_df (pd.DataFrame): DataFrame with columns ['metric', 'id', 'score', 'label']
        metric_name (str): Name of the metric to plot (as appears in flat_df['metric'])
        use_labels (bool): If True, plot positive and negative as subplots; else only positive.
    """
    plt.rcParams['font.family'] = 'Nimbus Roman'
    df_plot = flat_df[flat_df['metric'] == metric_name].copy()

    if df_plot.empty:
        print(f"[Warning] No data found for metric: {metric_name}")
        return

    variant_order = [
        'SH #224 @0.50',
        'SH #224 @0.75',
        'SH #128 @0.50',
        'SH #128 @0.75',
        'MH #224 @0.50',
        'MH #224 @0.75',
        'MH #128 @0.50',
        'MH #128 @0.75',
    ]

    def custom_formatter(value):
        if value == 0.:
            return '0'
        elif value == 1.0:
            return '1'
        # elif value < 0.0013 and value > -0.0013:
        #     return f'{value:.1e}'
        else:
            return f'{value:.2f}'

    # Prepare positive data
    df_pos = df_plot[['training_variant', 'fold', 'score_positive']].explode('score_positive')
    df_pos = df_pos.rename(columns={'score_positive': 'score'})
    df_pos['score'] = df_pos['score'].astype(float)
    heatmap_data_pos = df_pos.pivot_table(
        index='training_variant',
        columns='fold',
        values='score',
        aggfunc='mean'
    ).reindex(variant_order)

    if use_labels and 'score_negative' in df_plot.columns:
        # Prepare negative data
        df_neg = df_plot[['training_variant', 'fold', 'score_negative']].explode('score_negative')
        df_neg = df_neg.rename(columns={'score_negative': 'score'})
        df_neg['score'] = df_neg['score'].astype(float)
        heatmap_data_neg = df_neg.pivot_table(
            index='training_variant',
            columns='fold',
            values='score',
            aggfunc='mean'
        ).reindex(variant_order)

        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
        annot_pos = np.vectorize(custom_formatter)(heatmap_data_pos)
        sns.heatmap(
            heatmap_data_pos,
            annot=annot_pos,
            annot_kws={'size': 18, 'font': 'Nimbus Roman'},
            fmt="",
            cmap="gray",
            linewidths=0.5,
            linecolor='black',
            ax=axes[0]
        )
        axes[0].set_title(f'{metric_name} Heatmap (Positive)', fontdict={'fontsize': 16})
        axes[0].set_xlabel('Fold', fontsize=14)
        axes[0].set_ylabel('Training Variant', fontsize=14)
        axes[0].tick_params(axis='x', labelrotation=0, labelsize=14)
        axes[0].tick_params(axis='y', labelrotation=0, labelsize=14)

        annot_neg = np.vectorize(custom_formatter)(heatmap_data_neg)
        sns.heatmap(
            heatmap_data_neg,
            annot=annot_neg,
            annot_kws={'size': 18, 'font': 'Nimbus Roman'},
            fmt="",
            cmap="gray",
            linewidths=0.5,
            linecolor='black',
            ax=axes[1]
        )
        axes[1].set_title(f'{metric_name} Heatmap (Negative)', fontdict={'fontsize': 16})
        axes[1].set_xlabel('Fold', fontsize=14)
        axes[1].set_ylabel('Training Variant', fontsize=14)
        axes[1].tick_params(axis='x', labelrotation=0, labelsize=14)
        axes[1].tick_params(axis='y', labelrotation=0, labelsize=14)

        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        annot_pos = np.vectorize(custom_formatter)(heatmap_data_pos)
        sns.heatmap(
            heatmap_data_pos,
            annot=annot_pos,
            annot_kws={'size': 18, 'font': 'Nimbus Roman'},
            fmt="",
            cmap="gray",
            linewidths=0.5,
            linecolor='black',
        )
        plt.title(f'{metric_name} Heatmap (Positive)', fontdict={'fontsize': 16})
        plt.xlabel('Fold', fontsize=14)
        plt.ylabel('Training Variant', fontsize=14)
        plt.xticks(rotation=0, fontsize=14)
        plt.yticks(rotation=0, fontsize=14)
        plt.tight_layout()
        plt.show()
        if use_labels:
            print(f"[Info] No negative scores found for metric: {metric_name}")
        
    plt.savefig(f"heatmap_{metric_name}_positive.eps", format='eps', bbox_inches='tight')

plot_single_metric_boxplot(flat_df, "Sparseness", use_labels=True)
plot_single_metric_boxplot(flat_df, "Top-K Intersection", use_labels=False)
plot_single_metric_boxplot(flat_df, "Relevance Rank Accuracy", use_labels=False)
plot_single_metric_boxplot(flat_df, "Faithfulness Correlation", use_labels=True)
plot_single_metric_boxplot(flat_df, "Average Sensitivity", use_labels=True)


# %%
df_pos = flat_df[['metric', 'training_variant', 'fold', 'score_positive']].explode('score_positive')
df_pos = df_pos.rename(columns={'score_positive': 'score'})
df_pos['class'] = 'Positive'

df_neg = flat_df[['metric', 'training_variant', 'fold', 'score_negative']].explode('score_negative')
df_neg = df_neg.rename(columns={'score_negative': 'score'})
df_neg['class'] = 'Negative'

df_long = pd.concat([df_pos, df_neg], ignore_index=True)
df_long['score'] = df_long['score'].astype(float)


def parse_variant(variant_str):
    # Example variant: "MH#224@0.75"
    arch = variant_str.split('#')[0]  # MH or SH
    patch = variant_str.split('#')[1] if '#' in variant_str else variant_str
    return arch, patch

df_long[['architecture', 'patch_variant']] = df_long['training_variant'].apply(
    lambda x: pd.Series(parse_variant(x))
)

agg_df = df_long.groupby(['architecture', 'class', 'metric']).agg(
    mean_score=('score', 'mean'),
    std_score=('score', 'std'),
    n=('score', 'count')
).reset_index()


agg_df['df'] = agg_df['n'] - 1
agg_df['t_crit'] = agg_df['df'].apply(lambda df: stats.t.ppf(0.975, df) if df > 0 else float('nan'))
agg_df['sem'] = agg_df['std_score'] / agg_df['n']**0.5
agg_df['ci95_margin'] = agg_df['t_crit'] * agg_df['sem']

# CI bounds
agg_df['ci95_low'] = agg_df['mean_score'] - agg_df['ci95_margin']
agg_df['ci95_high'] = agg_df['mean_score'] + agg_df['ci95_margin']
agg_df['mean_ci95_str'] = agg_df.apply(
    lambda row: f"{row['mean_score']:.2f} [{row['ci95_low']:.2f}–{row['ci95_high']:.2f}]", axis=1
)

# %%

def compare_architectures(df_long, metric_name):
    """
    Compare MH vs SH architectures for each class using paired tests per fold.
    Returns a DataFrame with test results and FDR-corrected p-values.
    """

    df_long = df_long[df_long['metric'] == metric_name].copy()
    df_long['architecture'] = df_long['training_variant'].str.extract(r'^(MH|SH)')
    agg = df_long.groupby(['architecture', 'fold', 'class'])['score'].mean().reset_index()
    results = []

    print(f"Comparing architectures for metric: {metric_name}", end=' ')
    classes = agg['class'].unique()
   
    for c in classes:
        print(f" class={c}")
        data_class = agg[agg['class'] == c]
        pivot = data_class.pivot(index='fold', columns='architecture', values='score')

        # Skip if any architecture has missing scores for this class
        if pivot[['MH', 'SH']].isnull().any().any():
            print(f"Skipping class '{c}' due to missing data.")
            continue

        diffs = pivot['MH'] - pivot['SH']
        print('Mean 1:', pivot['MH'].mean())
        print('Mean 2:', pivot['SH'].mean())
        stat_norm, p_norm = stats.shapiro(diffs)

        if p_norm > 0.05:
            # differences are normal, use paired t-test
            print(f"Normality of differences test passed for class '{c}' with p-value {p_norm:.4f}. Using paired t-test.")
            stat_test, p_test = stats.ttest_rel(pivot['MH'], pivot['SH'])
            test_name = 'paired t-test'
        else:
            # not normal, use Wilcoxon signed-rank test
            print(f"Normality of differences test failed for class '{c}' with p-value {p_norm:.4f}. Using Wilcoxon signed-rank test.")
            stat_test, p_test = stats.wilcoxon(pivot['MH'], pivot['SH'])
            test_name = 'Wilcoxon signed-rank'

        results.append({
            'class': c,
            'normality_p': p_norm,
            'test': test_name,
            'statistic': stat_test,
            'p_value_raw': p_test
        })

    if results:  # Only correct if there's at least one test
        results_df = pd.DataFrame(results)
        reject, pvals_corrected, _, _ = multipletests(results_df['p_value_raw'], method='fdr_bh')
        results_df['p_value_adj'] = pvals_corrected
        results_df['reject_null'] = reject
    else:
        results_df = pd.DataFrame()

    print("Results DataFrame:")
    print(results_df)
    return results_df

results_df = compare_architectures(df_long, "Sparseness")
results_df = compare_architectures(df_long, "Top-K Intersection")
results_df = compare_architectures(df_long, "Relevance Rank Accuracy")
results_df = compare_architectures(df_long, "Faithfulness Correlation")
results_df = compare_architectures(df_long, "Average Sensitivity")
# %%

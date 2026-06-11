from argparse import ArgumentParser
from pathlib import Path
from itertools import combinations
import json

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import shapiro, friedmanchisquare, wilcoxon, ttest_rel
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM, AnovaResults

columns_map = {
    "eeg": "Model EEG",
    "ecg": "Model EKG",
    "ppg": "Model PPG",
    "gsr": "Model GSR",
    "image": "Model wizyjny",
    "question": "Model pytania",
    "answer": "Model odpowiedzi",
}


def plot_f1_by_fusion(contrib_df: pd.DataFrame, save_path: Path | None = None, title: str | None = None) -> None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=contrib_df, palette="Set2")
    plt.ylabel("F1-score")
    plt.xlabel("Model")
    if title is not None:
        plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def posthoc_friedman_matrix(df: pd.DataFrame, alpha_correction: str | None = "fdr_bh") -> pd.DataFrame:
    cols = df.columns
    pvals = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)

    pairs = []
    raw_ps = []

    for a, b in combinations(cols, 2):
        stat, p = wilcoxon(df[a], df[b], zero_method="wilcox")
        pairs.append((a, b))
        raw_ps.append(p)

    corrected = multipletests(raw_ps, method=alpha_correction)[1] if alpha_correction is not None else raw_ps

    for (a, b), p in zip(pairs, corrected):
        pvals.loc[a, b] = p
        pvals.loc[b, a] = p

    return pvals


def posthoc_anova_matrix(df: pd.DataFrame, alpha_correction: str | None = "fdr_bh") -> pd.DataFrame:
    cols = df.columns
    pvals = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)

    pairs = []
    raw_ps = []

    for a, b in combinations(cols, 2):
        stat, p = ttest_rel(df[a], df[b], nan_policy="omit")
        pairs.append((a, b))
        raw_ps.append(p)

    corrected = multipletests(raw_ps, method=alpha_correction)[1] if alpha_correction is not None else raw_ps

    for (a, b), p in zip(pairs, corrected):
        pvals.loc[a, b] = p
        pvals.loc[b, a] = p

    return pvals


def all_normal(coop_df: pd.DataFrame) -> bool:
    for col in coop_df.columns:
        stat, p = shapiro(coop_df[col])
        if p < 0.05:
            return False
    return True


def repeated_anova(df: pd.DataFrame) -> AnovaResults:
    long = df.reset_index(names="run").melt(id_vars="run", var_name="modality", value_name="contribution")

    return AnovaRM(data=long, depvar="contribution", subject="run", within=["modality"]).fit()


def statistical_diff(df: pd.DataFrame, posthoc_save_path: Path, alpha_correction: str | None = "fdr_bh") -> bool:
    if normal := all_normal(df):
        result = repeated_anova(df)

        p = result.anova_table["Pr > F"].iloc[0]
        print("Repeated-measures ANOVA")
        print(result)
    else:
        stat, p = friedmanchisquare(*(df[col] for col in df.columns))
        print(f"Friedman: stat={stat:.3f}, p={p:.3f}")

    if p >= 0.05:
        with open(posthoc_save_path, "w") as f:
            f.write("No significant differences found, post-hoc test not performed.\n")
        return False

    posthoc = (
        posthoc_anova_matrix(df, alpha_correction=alpha_correction)
        if normal
        else posthoc_friedman_matrix(df, alpha_correction=alpha_correction)
    )
    posthoc.to_csv(posthoc_save_path)

    return True


def load_modality_metrics(base_dir: Path, expr: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    base_metrics: dict[str, list[float]] = {}
    modalities_metrics: dict[str, dict[str, list[float]]] = {}

    for subdir in base_dir.glob(expr):

        metrics_json = subdir / "metrics_full.json"
        if not metrics_json.exists():
            metrics_json = subdir / "metrics.json"

        if not metrics_json.exists():
            print(f"Warning: {metrics_json} does not exist, skipping")
            continue

        with open(metrics_json, "r") as f:
            metrics = json.load(f)

        for k, v in metrics.items():
            if k == "individual_metrics":
                for modality, modality_metrics in v.items():
                    if modality not in modalities_metrics:
                        modalities_metrics[modality] = {}
                    for metric_name, metric_value in modality_metrics.items():
                        if metric_name not in modalities_metrics[modality]:
                            modalities_metrics[modality][metric_name] = []
                        modalities_metrics[modality][metric_name].append(metric_value)
                continue

            if k not in base_metrics:
                base_metrics[k] = []
            base_metrics[k].append(v)

    base_metrics_df = pd.DataFrame(base_metrics)
    modalities_metrics_dfs = {modality: pd.DataFrame(metrics) for modality, metrics in modalities_metrics.items()}
    return base_metrics_df, modalities_metrics_dfs


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--base-dir", type=str, required=True, help="Base directory for input and output files")
    parser.add_argument("--mode", type=int, required=True, help="Mode for processing")

    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    mode = args.mode

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory {base_dir} does not exist")



    results_dir = Path(f"data_{mode}")
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)


    fusion_metrics = {}
    fusion_types = ["early", "hybrid", "late"]
    for fusion_type in fusion_types:
        expr = (
            f"{fusion_type}_fusion_mode_{mode}*/results"
            if fusion_type != "early"
            else f"{fusion_type}_fusion2_mode_{mode}*/results"
        )

        base_metrics_df, modalities_metrics_dfs = load_modality_metrics(base_dir, expr)
        base_metrics_df.to_csv(results_dir / f"{fusion_type}_base_metrics.csv", index=False)
        fusion_metrics[fusion_type] = {
            "base_metrics": base_metrics_df,
            "modalities_metrics": modalities_metrics_dfs,
        }


if __name__ == "__main__":
    main()

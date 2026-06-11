from pathlib import Path

import pandas as pd
import seaborn as sns
from scipy.stats import shapiro, friedmanchisquare, studentized_range
from scikit_posthocs import posthoc_nemenyi_friedman, critical_difference_diagram
from matplotlib import pyplot as plt
plt.rcParams["font.size"] = 14

GROUPS = ["wczesna", "późna", "hybrydowa"]
FN_MAPPING = {"wczesna": "early", "późna": "late", "hybrydowa": "hybrid"}
SUFIX = "_base_metrics.csv"
RESULTS_DIR = Path("results/friedman/")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df_dict = {g: [] for g in GROUPS}
for group in GROUPS:
    df = pd.read_csv(f"data_1/{FN_MAPPING[group]}{SUFIX}")
    df_dict[group] = list(df["accuracy"].round(3))


df = pd.DataFrame(df_dict)


sns.boxplot(data=df, palette="Set2")
plt.title("Dokładność między grupami")
plt.xlabel("Grupa")
plt.ylabel("Dokładność")
plt.savefig(RESULTS_DIR / "boxplot-parametric-rel.png", bbox_inches="tight")
plt.show()

for group in GROUPS:
    # Check normality
    stat, p = shapiro(df[group])
    is_normal = p > 0.05
    print(f"Grup {group} is normal: {is_normal}")


stat, p = friedmanchisquare(*[df[g] for g in GROUPS])
is_significant = p < 0.05
print(f"Difference between groups is significant: {is_significant} (p={p:.3f})")

# Post-hoc test -- Nemenyi test
posthoc_results = posthoc_nemenyi_friedman(df)
posthoc_results.to_csv(RESULTS_DIR / "posthoc_nemenyi_results.csv", index=True)
print("Post-hoc Nemenyi test results:")
print(posthoc_results)

k = len(GROUPS)
N = len(df)

q_alpha = studentized_range.ppf(0.95, k, df=float('inf')) / (2 ** 0.5)
critical_difference = q_alpha * ((k * (k + 1)) / (6 * N)) ** 0.5

critical_difference_diagram(df.rank(axis=1).mean().sort_values(), posthoc_results, alpha=0.05, cd=critical_difference)
plt.title("Critical Difference Diagram")
plt.savefig(RESULTS_DIR / "critical_difference_diagram.png", bbox_inches="tight")
plt.show()


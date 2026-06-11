from pathlib import Path

import pandas as pd
import seaborn as sns
from scipy.stats import shapiro, wilcoxon, mannwhitneyu
from matplotlib import pyplot as plt

plt.rcParams["font.size"] = 14

GROUPS = ["późna", "hybrydowa"]
FN_MAPPING = {"wczesna": "early", "późna": "late", "hybrydowa": "hybrid"}
SUFIX = "_base_metrics.csv"
RESULTS_DIR = Path("results/nonparametric/")

HFP_PATH = "hfp/heart_failure_clinical_records_dataset.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


df_dict = {g: [] for g in GROUPS}
for group in GROUPS:
    df = pd.read_csv(f"data_2/{FN_MAPPING[group]}{SUFIX}")
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
    print(f"Grup {group} is normal: {is_normal}, p={p:.3f}")


group_a = "hybrydowa"
group_b = "późna"

stat, p = wilcoxon(df[group_a], df[group_b])
is_significant = p < 0.05
print(f"Difference between {group_a} and {group_b} is significant: {is_significant} (p={p:.3f})")


hfp_df = pd.read_csv(HFP_PATH)
hfp_df = hfp_df[hfp_df["age"].notna()]
x1 = hfp_df[(hfp_df["sex"] == 1)]["age"]
x2 = hfp_df[(hfp_df["sex"] == 0)]["age"]

if len(x1) != len(x2):
    min_len = min(len(x1), len(x2))
    x1 = x1.sample(n=min_len, random_state=42)
    x2 = x2.sample(n=min_len, random_state=42)

df = pd.DataFrame({
    "Mężczyźni": x1,
    "Kobiety": x2,
})

sns.boxplot(data=df, palette="Set2")
plt.title("Wiek między płciami")
plt.xlabel("Płeć")
plt.ylabel("Wiek")
plt.savefig(RESULTS_DIR / "boxplot-parametric-rel-hfp.png", bbox_inches="tight")
plt.show()

for idx, group in enumerate((x1, x2)):
    gn = "Mężczyźni" if idx == 0 else "Kobiety"
    stat, p = shapiro(group)
    is_normal = p > 0.05
    print(f"Age in group {gn} is normal: {is_normal}, p={p:.3f}")

stat, p = mannwhitneyu(x1, x2)
is_significant = p < 0.05
print(f"Difference between Mężczyźni and Kobiety is significant: {is_significant} (p={p:.3f})")

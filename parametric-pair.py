from pathlib import Path

from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro, levene, chisquare, ttest_rel, ttest_ind
from matplotlib import pyplot as plt
plt.rcParams["font.size"] = 14

GROUPS = ["wczesna", "późna"]
FN_MAPPING = {"wczesna": "early", "późna": "late", "hybrydowa": "hybrid"}
SUFIX = "_base_metrics.csv"
RESULTS_DIR = Path("results/parametric/")

HFP_PATH = "hfp/heart_failure_clinical_records_dataset.csv"

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
    print(f"Grup {group} is normal: {is_normal}, p={p:.3f}")

# Variance test -- Levene's test
stat, p = levene(*[df[g] for g in GROUPS])
is_homogeneous = p > 0.05
print(f"Variance is homogeneous: {is_homogeneous}, p={p:.3f}")

# T-test
group_a = "wczesna"
group_b = "późna"
stat, p = ttest_rel(df[group_a], df[group_b])
is_significant = p < 0.05
print(f"Difference between {group_a} and {group_b} is significant: {is_significant} (p={p:.3f})")

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = iris.target
col = "sepal width (cm)"

species = df["species"].unique()[:2]
sns.boxplot(x="species", y=col, data=df[df["species"].isin(species)], palette="Set2")
plt.title("Szerokość działki w zależności od gatunku")
plt.xlabel("Gatunek")
plt.ylabel("Szerokość działki (cm)")
plt.savefig(RESULTS_DIR / "boxplot-parametric-ind.png", bbox_inches="tight")
plt.show()

for s in species:
    group = df[df["species"] == s][col]
    stat, p = shapiro(group)
    is_normal = p > 0.05
    print(f"Count of species {s}: {len(group)}")
    print(f"Sepal length in species {s}, col {col} is normal: {is_normal}, p={p:.3f}")

stat, p = levene(*[df[df["species"] == s][col] for s in species])
is_homogeneous = p > 0.05
print(f"Variance is homogeneous: {is_homogeneous}, p={p:.3f}")

group_a = species[0]
group_b = species[1]
stat, p = ttest_ind(df[df["species"] == group_a][col], df[df["species"] == group_b][col])
is_significant = p < 0.05
print(f"Difference between {group_a} and {group_b} is significant: {is_significant} (p={p:.3f})")


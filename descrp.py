from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams["font.size"] = 14

GROUPS = ["wczesna", "późna", "hybrydowa"]
FN_MAPPING = {"wczesna": "early", "późna": "late", "hybrydowa": "hybrid"}
SUFIX = "_base_metrics.csv"
RESULTS_DIR = Path("results/description/")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df_dict = {g: [] for g in GROUPS}
for group in GROUPS:
    df = pd.read_csv(f"data/{FN_MAPPING[group]}{SUFIX}")
    df_dict[group] = list(df["accuracy"].round(3))


df = pd.DataFrame(df_dict)


descriptions_dict = {
    "Średnia": [df[group].mean() for group in GROUPS],
    "Mediana": [df[group].median() for group in GROUPS],
    "Moda": [df[group].mode()[0] for group in GROUPS],
    "Odchylenie standardowe": [df[group].std() for group in GROUPS],
    "Wariancja": [df[group].var() for group in GROUPS],
    "Współczynnik zmienności": [int(df[group].std() / df[group].mean() * 100) for group in GROUPS],
    "Kwartyl 1": [df[group].quantile(0.25) for group in GROUPS],
    "Kwartyl 3": [df[group].quantile(0.75) for group in GROUPS],
    "Odstęp międzykwartylowy": [df[group].quantile(0.75) - df[group].quantile(0.25) for group in GROUPS],
    "Minimum": [df[group].min() for group in GROUPS],
    "Maximum": [df[group].max() for group in GROUPS],
    "Rozstęp": [df[group].max() - df[group].min() for group in GROUPS],
    "Skosność": [df[group].skew() for group in GROUPS],
    "Kurtoza": [df[group].kurtosis() for group in GROUPS],
}

descriptions_df = pd.DataFrame(descriptions_dict, index=GROUPS)
descriptions_df = descriptions_df.transpose().round(4)
print(descriptions_df)
descriptions_df.to_csv(RESULTS_DIR / "description.csv")

sns.boxplot(data=df, palette="Set2")
plt.xlabel("Rodzaj fuzji")
plt.ylabel("Dokładność")
plt.title("Porównanie dokładności dla różnych rodzajów fuzji")
plt.savefig(RESULTS_DIR / "boxplot.png", bbox_inches="tight")
plt.show()

for group in GROUPS:
    sns.histplot(df[group], kde=True, color="skyblue", bins=10)  # type: ignore
    plt.xlabel("Dokładność")
    plt.ylabel("Częstość")
    plt.title(f"Histogram dokładności dla grupy {group}")
    plt.savefig(RESULTS_DIR / f"histogram_{group}.png", bbox_inches="tight")
    plt.show()

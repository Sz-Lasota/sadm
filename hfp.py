import pandas as pd
import seaborn as sns
from scipy.stats import shapiro, wilcoxon
from matplotlib import pyplot as plt
from pathlib import Path


HFP_PATH = "hfp/heart_failure_clinical_records_dataset.csv"
COLUMNS = ["ejection_fraction", "creatinine_phosphokinase", "platelets", "serum_creatinine", "serum_sodium"]
COLUMN = COLUMNS[4]

for seed_ in range(10000):
    hfp_df = pd.read_csv(HFP_PATH)

    hfp_df = hfp_df[hfp_df[COLUMN].notna()]
    x1 = hfp_df[(hfp_df["sex"] == 1)][COLUMN]
    x2 = hfp_df[(hfp_df["sex"] == 0)][COLUMN]

    if len(x1) != len(x2):
        min_len = min(len(x1), len(x2))
        x1 = x1.sample(n=min_len, random_state=seed_)
        x2 = x2.sample(n=min_len, random_state=seed_)

    df = pd.DataFrame({
        "Mężczyźni": x1,
        "Kobiety": x2,
    })

    # sns.histplot(data=df, palette="Set2")
    # plt.xlabel("Płeć")
    # plt.ylabel(COLUMN)
    # plt.show()
    both_normal = True

    for idx, group in enumerate((x1, x2)):
        gn = "Mężczyźni" if idx == 0 else "Kobiety"
        stat, p = shapiro(group)
        is_normal = p > 0.05
        if not is_normal:
            both_normal = False
        # print(f"Age in group {gn} is normal: {is_normal}, p={p:.3f}")

    if both_normal:
        print(f"Both groups are normal for column {col}")
        break


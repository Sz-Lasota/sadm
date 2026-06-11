from pathlib import Path
import sys
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

RESULT_DIR = Path("results/kaplan-meier")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
hfp_df = pd.read_csv("hfp/heart_failure_clinical_records_dataset.csv")
kmf = KaplanMeierFitter()

crit = sys.argv[1]

kmf.fit(
    durations=hfp_df["time"],
    event_observed=hfp_df["DEATH_EVENT"],
    label="Przeżywalność pacjentów z niewydolnością serca",
)
ax = plt.subplot(111)

for group_value in hfp_df[crit].unique():
    mask = hfp_df[crit] == group_value

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=hfp_df[mask]["time"],
        event_observed=hfp_df[mask]["DEATH_EVENT"],
        label="Nadciśnienie" if group_value == 1 else "Brak nadciśnienia",
    )

    kmf.plot(ax=ax)

plt.title("Kaplan-Meier ze względu na płeć")
plt.xlabel("Czas (dni)")
plt.ylabel("Prawdopodobieństwo przeżycia")
plt.savefig(RESULT_DIR / "km-plot.pdf", bbox_inches="tight")
plt.show()



a = hfp_df[hfp_df[crit] == 1]
b = hfp_df[hfp_df[crit] == 0]

result = logrank_test(a["time"], b["time"],
                      event_observed_A=a["DEATH_EVENT"],
                      event_observed_B=b["DEATH_EVENT"])

print(f"p-value: {result.p_value:.4f}")
print(f"Test statistic: {result.test_statistic:.4f}")

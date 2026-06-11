import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import shapiro, wilcoxon, levene, chisquare

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = iris.target
col = "sepal width (cm)"

species = df["species"].unique()[:2]

for s in species:
    group = df[df["species"] == s][col]
    stat, p = shapiro(group)
    is_normal = p > 0.05
    print(f"Count of species {s}: {len(group)}")
    print(f"Sepal length in species {s}, col {col} is normal: {is_normal}, p={p:.3f}")

stat, p = levene(*[df[df["species"] == s][col] for s in species])
is_homogeneous = p > 0.05
print(f"Variance is homogeneous: {is_homogeneous}, p={p:.3f}")



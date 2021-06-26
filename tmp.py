import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

plt.style.use("seaborn-whitegrid")

df = pd.read_excel("dataset.xls")

X = df.copy()
y = X.pop("Stage")

y = y.apply(LabelEncoder().fit_transform)



convert_numeric = X.apply(LabelEncoder().fit_transform)
discrete_features = convert_numeric.dtypes == int


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


mi_scores = make_mi_scores(convert_numeric, y, discrete_features)
#mi_scores[::3]  # show a few features with their MI scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)

"""
Real-data validation of the assessment-performance trajectory axis.

Sequence L = [ssc_p, hsc_p, degree_p, etest_p, mba_p]  (5 real sequential stages)
Outcome    = status (Placed / Not Placed), and salary among the placed.

This tests ONLY the trajectory axis (no sentiment data exists in this dataset).
N=215 -> use stratified k-fold CV, not a single split, and report CV spread honestly.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_THIS_DIR, "..", "data", "Placement_Data_Full_Class.csv")

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

stage_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
L = df[stage_cols].to_numpy(dtype=float)
y_status = (df["status"].str.strip() == "Placed").astype(int).to_numpy()

def phi(V):
    n, T = V.shape
    stages = np.arange(T)
    slope = np.array([np.polyfit(stages, V[i], 1)[0] for i in range(n)])
    volatility = V.std(axis=1)
    diffs = np.diff(V, axis=1)
    recovery = np.array([
        np.mean(diffs[i][np.where(diffs[i][:-1] < 0)[0] + 1])
        if np.any(diffs[i][:-1] < 0) else 0.0
        for i in range(n)
    ])
    consistency = -np.mean(np.abs(diffs), axis=1)
    return np.stack([slope, volatility, recovery, consistency], axis=1)

phi_L = phi(L)

# static-only baseline: final-stage score + simple average (what a snapshot system would use)
x_static = np.stack([L[:, -1], L.mean(axis=1)], axis=1)  # [mba_p, mean_of_all_stages]
x_fusion = np.concatenate([x_static, phi_L], axis=1)

print(f"N = {len(df)}, placed = {y_status.sum()}, not placed = {(1-y_status).sum()}")
print(f"Missing values in stage columns: {df[stage_cols].isna().sum().sum()}")

# ---- classification: predicting placement status ----
def cv_auc(X, y, n_splits=5, n_repeats=20):
    aucs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for tr, te in skf.split(X, y):
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
            clf = LogisticRegression(max_iter=1000)
            clf.fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
            if len(np.unique(y[te])) == 2:
                aucs.append(roc_auc_score(y[te], p))
    return np.array(aucs)

auc_static = cv_auc(x_static, y_status)
auc_fusion = cv_auc(x_fusion, y_status)

from scipy.stats import ttest_rel
n = min(len(auc_static), len(auc_fusion))
t, p = ttest_rel(auc_fusion[:n], auc_static[:n])

print("\n=== Predicting placement status (5-fold CV x 20 repeats, AUC) ===")
print(f"static-only  AUC = {auc_static.mean():.4f} (sd {auc_static.std():.4f})")
print(f"traj-fusion  AUC = {auc_fusion.mean():.4f} (sd {auc_fusion.std():.4f})")
print(f"paired t-test on matched folds: t={t:.4f}, p={p:.6f}")

# ---- regression: predicting salary among placed candidates ----
placed_mask = y_status == 1
Ls = L[placed_mask]
salary = df.loc[placed_mask, "salary"].to_numpy(dtype=float)
phi_Ls = phi(Ls)
x_static_s = np.stack([Ls[:, -1], Ls.mean(axis=1)], axis=1)
x_fusion_s = np.concatenate([x_static_s, phi_Ls], axis=1)

def cv_corr(X, y, n_splits=5, n_repeats=20):
    from scipy.stats import pearsonr
    corrs = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for tr, te in kf.split(X):
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
            reg = Ridge(alpha=1.0)
            reg.fit(Xtr, y[tr])
            pred = reg.predict(Xte)
            if np.std(pred) > 1e-8:
                r, _ = pearsonr(pred, y[te])
                corrs.append(r)
    return np.array(corrs)

corr_static = cv_corr(x_static_s, salary)
corr_fusion = cv_corr(x_fusion_s, salary)
n2 = min(len(corr_static), len(corr_fusion))
t2, p2 = ttest_rel(corr_fusion[:n2], corr_static[:n2])

print(f"\n=== Predicting salary among placed (N={placed_mask.sum()}), 5-fold CV x 20 repeats, Pearson r ===")
print(f"static-only  r = {corr_static.mean():.4f} (sd {corr_static.std():.4f})")
print(f"traj-fusion  r = {corr_fusion.mean():.4f} (sd {corr_fusion.std():.4f})")
print(f"paired t-test on matched folds: t={t2:.4f}, p={p2:.6f}")

# ---- mechanism check: what does the trajectory actually look like for placed vs not? ----
print("\n=== Shape-feature means by outcome group ===")
for k, lab in enumerate(["slope", "volatility", "recovery", "consistency"]):
    print(f"  {lab}: placed={phi_L[y_status==1, k].mean():.4f}  "
          f"not_placed={phi_L[y_status==0, k].mean():.4f}")

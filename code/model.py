import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, ttest_rel

def phi(V):
    """Shape-feature extractor: [slope, volatility, recovery, consistency]
    V: (n, n_stages) array. Sequence-agnostic -- no domain-specific assumptions."""
    n, T = V.shape
    stages = np.arange(T)
    slope = np.array([np.polyfit(stages, V[i], 1)[0] for i in range(n)])
    volatility = V.std(axis=1)
    diffs = np.diff(V, axis=1)
    recovery = np.array([
        np.sum(diffs[i][:-1] < 0) and np.mean(diffs[i][np.where(diffs[i][:-1] < 0)[0] + 1])
        if np.any(diffs[i][:-1] < 0) else 0.0
        for i in range(n)
    ])
    consistency = -np.mean(np.abs(diffs), axis=1)  # higher = more consistent
    return np.stack([slope, volatility, recovery, consistency], axis=1)

def fit_eval(x_train, y_train, x_test, y_test):
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    r, _ = pearsonr(pred, y_test)
    return r, model

def run_seed(data, seed, target_key="outcome_latent"):
    rng = np.random.default_rng(seed + 9999)
    n = data["x_static"].shape[0]
    idx = rng.permutation(n)
    split = int(n * 0.7)
    tr, te = idx[:split], idx[split:]

    y = data[target_key]
    x_static = data["x_static"]
    phi_L = phi(data["L"])
    phi_S = phi(data["S"])

    x_static_only = x_static
    x_fusion = np.concatenate([x_static, phi_L, phi_S], axis=1)

    r_static, _ = fit_eval(x_static_only[tr], y[tr], x_static_only[te], y[te])
    r_fusion, fusion_model = fit_eval(x_fusion[tr], y[tr], x_fusion[te], y[te])

    return r_static, r_fusion, (x_fusion, fusion_model, te)

def multiseed_compare(data, n_seeds=10, target_key="outcome_latent"):
    statics, fusions = [], []
    last_fusion_bundle = None
    for s in range(n_seeds):
        r_s, r_f, bundle = run_seed(data, s, target_key)
        statics.append(r_s)
        fusions.append(r_f)
        last_fusion_bundle = bundle
    statics = np.array(statics)
    fusions = np.array(fusions)
    t, p = ttest_rel(fusions, statics)
    return dict(
        static_mean=statics.mean(), static_sd=statics.std(),
        fusion_mean=fusions.mean(), fusion_sd=fusions.std(),
        diff_mean=(fusions - statics).mean(), diff_sd=(fusions - statics).std(),
        t=t, p=p, n_seeds=n_seeds, last_fusion_bundle=last_fusion_bundle,
    )

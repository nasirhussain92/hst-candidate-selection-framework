import numpy as np
from sklearn.neural_network import MLPRegressor
from scipy.stats import pearsonr, ttest_rel
from generator import generate_candidates
from model import phi, fit_eval

def run_seed_with_neural(data, seed, target_key="outcome_latent"):
    rng = np.random.default_rng(seed + 9999)
    n = data["x_static"].shape[0]
    idx = rng.permutation(n)
    split = int(n * 0.7)
    tr, te = idx[:split], idx[split:]
    y = data[target_key]

    x_static = data["x_static"]
    phi_L, phi_S = phi(data["L"]), phi(data["S"])
    x_fusion = np.concatenate([x_static, phi_L, phi_S], axis=1)
    x_raw = np.concatenate([x_static, data["L"], data["S"]], axis=1)  # raw, unengineered

    r_static, _ = fit_eval(x_static[tr], y[tr], x_static[te], y[te])
    r_fusion, _ = fit_eval(x_fusion[tr], y[tr], x_fusion[te], y[te])

    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000,
                        random_state=seed, early_stopping=True)
    mlp.fit(x_raw[tr], y[tr])
    pred = mlp.predict(x_raw[te])
    r_neural, _ = pearsonr(pred, y[te])

    return r_static, r_fusion, r_neural

def compare(data, n_seeds=10, target_key="outcome_latent"):
    S, F, Nn = [], [], []
    for s in range(n_seeds):
        rs, rf, rn = run_seed_with_neural(data, s, target_key)
        S.append(rs); F.append(rf); Nn.append(rn)
    S, F, Nn = np.array(S), np.array(F), np.array(Nn)
    t_fn, p_fn = ttest_rel(F, Nn)
    t_sn, p_sn = ttest_rel(S, Nn)
    print(f"static  r mean={S.mean():.4f} sd={S.std():.4f}")
    print(f"fusion  r mean={F.mean():.4f} sd={F.std():.4f}")
    print(f"neural  r mean={Nn.mean():.4f} sd={Nn.std():.4f}")
    print(f"fusion vs neural: t={t_fn:.4f} p={p_fn:.6f}")
    print(f"static vs neural: t={t_sn:.4f} p={p_sn:.6f}")

print("=== GENERAL scenario ===")
data_general = generate_candidates(4000, seed=1, favorable=False)
compare(data_general, target_key="outcome_latent")

print("\n=== FAVORABLE scenario ===")
data_fav = generate_candidates(4000, seed=2, favorable=True)
compare(data_fav, target_key="outcome_latent")

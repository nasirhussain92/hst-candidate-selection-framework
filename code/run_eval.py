import numpy as np
from generator import generate_candidates, N_STAGES
from model import multiseed_compare, phi

N = 4000

def report(name, res):
    print(f"\n=== {name} ===")
    print(f"static-only  r = {res['static_mean']:.4f} (sd {res['static_sd']:.4f})")
    print(f"fusion       r = {res['fusion_mean']:.4f} (sd {res['fusion_sd']:.4f})")
    print(f"diff (fusion-static) mean={res['diff_mean']:.4f} sd={res['diff_sd']:.4f}")
    print(f"paired t-test: t={res['t']:.4f}, p={res['p']:.6f}")

# --- General scenario ---
data_general = generate_candidates(N, seed=1, favorable=False)
res_general_latent = multiseed_compare(data_general, target_key="outcome_latent")
report("GENERAL scenario vs true (latent) outcome", res_general_latent)
res_general_real = multiseed_compare(data_general, target_key="realized_outcome")
report("GENERAL scenario vs noisy realized outcome", res_general_real)

# --- Favorable (growth/composure-weighted) scenario ---
data_fav = generate_candidates(N, seed=2, favorable=True)
res_fav_latent = multiseed_compare(data_fav, target_key="outcome_latent")
report("FAVORABLE (growth-weighted) scenario vs true (latent) outcome", res_fav_latent)
res_fav_real = multiseed_compare(data_fav, target_key="realized_outcome")
report("FAVORABLE (growth-weighted) scenario vs noisy realized outcome", res_fav_real)

# --- Fairness audit: does the trajectory shape-feature disadvantage the
#     accommodation-need group at matched true_quality? ---
print("\n=== FAIRNESS AUDIT (favorable scenario, fusion model) ===")
bundle = res_fav_latent["last_fusion_bundle"]
x_fusion, fusion_model, te = bundle
acc = data_fav["accommodation_need"][te]
tq = data_fav["true_quality"][te]
pred = fusion_model.predict(x_fusion[te])

# match on true_quality via quintile bins, compare predicted score by group within bin
bins = np.quantile(tq, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
bins[-1] += 1e-6
bin_idx = np.digitize(tq, bins) - 1

gaps = []
for b in range(5):
    mask = bin_idx == b
    if mask.sum() < 10:
        continue
    g1 = pred[mask & (acc == 1)]
    g0 = pred[mask & (acc == 0)]
    if len(g1) >= 3 and len(g0) >= 3:
        gap = g1.mean() - g0.mean()
        gaps.append(gap)
        print(f"quality-quintile {b}: n_acc={len(g1)}, n_no_acc={len(g0)}, "
              f"mean_pred_acc={g1.mean():.4f}, mean_pred_no_acc={g0.mean():.4f}, gap={gap:.4f}")

if gaps:
    print(f"\nMean within-quintile predicted-score gap "
          f"(accommodation-need minus no-need, matched true_quality): {np.mean(gaps):.4f}")

# also report the raw phi(L) slope/consistency gap directly (mechanism check)
phiL = phi(data_fav["L"])
acc_full = data_fav["accommodation_need"]
print("\nRaw trajectory shape-feature gap by group (whole sample, unmatched):")
labels = ["slope", "volatility", "recovery", "consistency"]
for k, lab in enumerate(labels):
    g1 = phiL[acc_full == 1, k]
    g0 = phiL[acc_full == 0, k]
    print(f"  {lab}: acc_need mean={g1.mean():.4f}, no_need mean={g0.mean():.4f}, "
          f"diff={g1.mean()-g0.mean():.4f}")

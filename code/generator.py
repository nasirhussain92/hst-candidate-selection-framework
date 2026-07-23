"""
Synthetic candidate data generator.

Two temporal sequences per candidate:
  L = multi-stage assessment performance trajectory
      (screening -> coding/technical test -> case study -> work sample)
  S = interview sentiment trajectory (affective dynamics across stages)

Plus static features and a protected-like "accommodation_need" flag used
ONLY for the fairness audit (never used as a model input).
"""
import numpy as np

N_STAGES = 4

def generate_candidates(n, seed, favorable=False, fatigue_strength=0.35):
    rng = np.random.default_rng(seed)

    true_quality = rng.normal(0, 1, n)
    growth_true = rng.normal(0.15, 1, n)      # tendency to improve across stages
    composure_true = rng.normal(0, 1, n)      # affective stability latent

    # static features correlated with true ability, plus CQ/AQ-style scores
    resume_score = 0.6 * true_quality + rng.normal(0, 0.8, n)
    cq_score = 0.4 * true_quality + rng.normal(0, 0.9, n)
    aq_score = 0.3 * true_quality + rng.normal(0, 0.9, n)
    x_static = np.stack([resume_score, cq_score, aq_score], axis=1)

    # accommodation-need flag: unrelated to true_quality by construction
    accommodation_need = rng.binomial(1, 0.15, n)

    # --- assessment performance trajectory L (N_STAGES scores) ---
    L = np.zeros((n, N_STAGES))
    for j in range(N_STAGES):
        stage_frac = j / (N_STAGES - 1)
        growth_component = growth_true * stage_frac
        # fatigue: performance erodes across a long multi-stage process;
        # candidates needing accommodations experience a larger fatigue penalty
        # under the (unaccommodated) synthetic process -- this is the fairness
        # risk flagged in the write-up, modeled explicitly so it can be audited.
        fatigue = fatigue_strength * stage_frac * (1 + 0.8 * accommodation_need)
        noise = rng.normal(0, 0.5, n)
        L[:, j] = true_quality + growth_component - fatigue + noise

    # --- sentiment trajectory S (N_STAGES scores) ---
    S = np.zeros((n, N_STAGES))
    for j in range(N_STAGES):
        stage_frac = j / (N_STAGES - 1)
        recovery_wobble = 0.3 * np.sin(stage_frac * np.pi) * (composure_true > 0)
        noise = rng.normal(0, 0.5, n)
        S[:, j] = composure_true + recovery_wobble + noise

    # --- outcome ---
    if favorable:
        # growth/composure-weighted, analogous to the earlier "leadership" scenario
        outcome_latent = 0.4 * true_quality + 0.3 * growth_true + 0.3 * composure_true
    else:
        outcome_latent = 0.85 * true_quality + 0.15 * growth_true

    realized_outcome = outcome_latent + rng.normal(0, 0.6, n)  # noisy real-world label

    return dict(
        x_static=x_static, L=L, S=S,
        true_quality=true_quality, growth_true=growth_true,
        composure_true=composure_true, accommodation_need=accommodation_need,
        outcome_latent=outcome_latent, realized_outcome=realized_outcome,
    )

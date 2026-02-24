export async function onRequestGet() {
  return Response.json({
    quality_weights: {
      entity_density:      0.22,
      topic_clarity:       0.18,
      informational_value: 0.16,
      freshness_signal:    0.12,
      engagement_depth:    0.10,
      title_formatting:    0.08,
      natural_authority:   0.08,
      visual_promise:      0.06,
    },
    beta_weight:    0.35,
    ctr_floor:      0.005,
    ctr_ceil:       0.22,
    sigmoid_alpha:  0.65,
    sigmoid_mu:     5.5,
    model:          "claude-opus-4-6",
    formula: [
      "quality   = Σ(wᵢ × fᵢ)  — 8 quality dimensions",
      "β_penalty = 1 − 0.35 × (clickbait_score / 10)",
      "raw       = quality × β_penalty",
      "pCTR      = 0.5% + (22% − 0.5%) × σ(0.65 × (raw − 5.5))",
    ],
  });
}

# iter14 — Deferred mitigations (handle-if-needed only)

> Activate ONLY if Δ2 < 0 (surgery doesn't beat pretrain) in T6b eval. All "implement-now" items landed in `surgery_base.yaml` + `m09c_surgery.py` + `run_train.sh` (T4, 2026-05-08).

| Mitigation | What | Trigger / How |
|---|---|---|
| 🎓 (e) KL distillation | aux KL(student_logits ‖ pretrain_logits) | encoder drifts off pretrain basin (CKA < 0.7) |
| 🪶 (f) LoRA / adapter-only | freeze base, train rank-r delta | Δ2 < 0 with anchor at λ=0.01; reuse `src/legacy/m09b_explora.py` LoRA primitive |
| 🧮 (g) EWC / Fisher | replace uniform L2 with Fisher-weighted | uniform L2 (λ=0.005) insufficient at Δ2 |
| 🚦 Backbone LR cap 1e-5 | override `base_optimization.yaml:143` lr 5e-5 → 1e-5 | drift_loss > 0.05 mid-training |
| 🌡️ Per-stage warmup | currently global `warmup_pct=0.12` — add stage-boundary warmup | val_jepa spike at stage transitions |
| ⚠️ val_jepa rise > 5% abort | flip `probe.kill_switch_enabled.full: false → true` (already wired in m09c:845) | use after first FULL run shows trajectory |

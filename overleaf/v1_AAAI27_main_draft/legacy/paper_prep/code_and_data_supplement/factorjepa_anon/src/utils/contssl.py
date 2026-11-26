"""iter18 B3 baseline helpers — continual-SSL regularizers: CaSSLe distillation + EWC Fisher.

Both are anti-forgetting baselines that vjepa_surgery must beat. Technique-agnostic (#49):
m09d_contssl_encoder.py calls these config-gated; ZERO `if technique` branches live here.

Gold standards (cited per src/CLAUDE.md "Training scripts MUST cite official gold-standard repo"):
  - CaSSLe : Fini et al., "Self-Supervised Models are Continual Learners", CVPR 2022.
             https://github.com/DonkeyShot21/cassle
             A learned predictor g maps the CURRENT model's representation onto the FROZEN
             previous model's representation; the SSL criterion is reused as the distillation
             loss with stop-grad on the frozen target. Here the FROZEN teacher (SALT slot) IS
             the previous model, and the criterion is the JEPA L1 (matching the pure-JEPA scaffold).
  - EWC    : Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017.
             reference impl https://github.com/moskomule/ewc.pytorch
             Diagonal empirical Fisher F_i = E[(∂L/∂θ_i)^2] weights a quadratic anchor to the
             pretrained init θ*:  L_ewc = λ Σ_i F_i (θ_i − θ*_i)^2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────── CaSSLe (Fini CVPR'22) ─────────────────────────

def build_cassle_predictor(d_encoder: int, hidden_dim: int) -> nn.Module:
    """CaSSLe predictor g — the BN-MLP that maps current-feature → frozen-prev-feature.

    Matches the cassle repo's `Predictor` (Linear→BN→ReLU→Linear). Trainable (added to the
    optimizer via attach_cassle_to_optimizer); discarded at export (eval uses the encoder only).
    """
    return nn.Sequential(
        nn.Linear(d_encoder, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, d_encoder),
    )


def attach_cassle_to_optimizer(optimizer, cassle_head: "nn.Module | None",
                               base_lr: float, head_lr_multiplier: float) -> None:
    """Add g's params as a separate optimizer group (own LR, no weight decay). No-op when None.

    Mirrors motion_aux_loss.attach_motion_aux_to_optimizer so g's LR scales independently of
    the encoder LR.
    """
    if cassle_head is None:
        return
    optimizer.add_param_group({
        "params":       list(cassle_head.parameters()),
        "lr":           base_lr * head_lr_multiplier,
        "weight_decay": 0.0,
        "name":         "cassle_predictor",
    })


def run_cassle_step(student, teacher, cassle_head: "nn.Module | None",
                    batch_clips: torch.Tensor, scaler, mp_cfg: dict, dtype, device,
                    weight_cassle: float) -> float:
    """One CaSSLe distillation forward+backward over the macro-batch. Returns the distill loss.

    g(pool(student(x))) is trained to predict pool(teacher(x)) — the FROZEN previous model's
    feature (stop-grad). Scaled backward accumulates onto the SAME param.grad buffer as the JEPA
    grads (single optimizer.step consumes both), exactly like run_motion_aux_step.

      - returns 0.0 when cassle_head is None.
      - re-raises torch.cuda.OutOfMemoryError so the caller's per-step OOM handler wins.

    Both encoders emit hierarchical (B, N, 4*D) at train time; we toggle return_hierarchical OFF
    for these forwards so the pooled feature is (B, D) — g expects D, not 4*D. try/finally restores.
    """
    if cassle_head is None:
        return 0.0
    # PEFT-safety parity with run_motion_aux_step (no-op here — m09d is full-FT, no PeftModel).
    if hasattr(student, "get_base_model"):
        student = student.get_base_model()
    s_had_hier = getattr(student, "return_hierarchical", None)
    t_had_hier = getattr(teacher, "return_hierarchical", None)
    if s_had_hier is True:
        student.return_hierarchical = False
    if t_had_hier is True:
        teacher.return_hierarchical = False
    try:
        with torch.amp.autocast("cuda", enabled=mp_cfg["enabled"], dtype=dtype):
            z_student = student(batch_clips)
            if isinstance(z_student, (list, tuple)):
                z_student = z_student[-1]
            z_student = z_student.mean(dim=1)                 # (B, D)
            with torch.no_grad():
                z_teacher = teacher(batch_clips)
                if isinstance(z_teacher, (list, tuple)):
                    z_teacher = z_teacher[-1]
                z_teacher = z_teacher.mean(dim=1)             # (B, D) — frozen prev-model feature
            pred = cassle_head(z_student)                     # g: current → predict frozen
            cassle_loss = F.l1_loss(pred, z_teacher.detach())  # stop-grad on the target
            cassle_scaled = cassle_loss * float(weight_cassle)
        loss_value = float(cassle_loss.detach().item())
        if scaler is not None and cassle_scaled.requires_grad and loss_value > 0.0:
            scaler.scale(cassle_scaled).backward()
        return loss_value
    finally:
        if s_had_hier is not None:
            student.return_hierarchical = s_had_hier
        if t_had_hier is not None:
            teacher.return_hierarchical = t_had_hier


# ───────────────────────── EWC (Kirkpatrick PNAS'17) ─────────────────────────

def init_fisher(student) -> dict:
    """Zero diagonal-Fisher accumulator over the trainable params (online-EWC accumulation)."""
    return {n: torch.zeros_like(p, device=p.device)
            for n, p in student.named_parameters() if p.requires_grad}


def accumulate_fisher(student, fisher: dict, n_total: int, grad_scale: float) -> None:
    """Add (∂L_jepa/∂θ)^2 / n_total of the CURRENT grads into `fisher` (call right after the JEPA
    backward, BEFORE any auxiliary backward contaminates the grads).

    grad_scale = scaler.get_scale() (1.0 when the GradScaler is disabled, e.g. bf16). Live grads are
    loss-scaled under fp16; dividing by grad_scale puts the Fisher in the true-gradient domain (a
    CONSTANT scale would fold into λ, but the fp16 scaler is dynamic → unscale to be safe; no-op at bf16).

    Online empirical Fisher at θ≈θ* (the first n_total steps sit near the pretrained init, so this
    approximates the Kirkpatrick Fisher evaluated at θ* without a separate previous-task loader).
    """
    inv = 1.0 / float(grad_scale)
    for n, p in student.named_parameters():
        if p.requires_grad and p.grad is not None and n in fisher:
            g = p.grad.detach() * inv
            fisher[n] += (g * g) / float(n_total)


def add_ewc_grads(student, fisher: dict, theta_star: dict, lam: float) -> float:
    """Add the analytic EWC gradient 2λ F_i (θ_i − θ*_i) DIRECTLY onto p.grad; return the scalar penalty
    λ Σ_i F_i (θ_i − θ*_i)^2 for logging (Kirkpatrick 2017).

    Closed-form grad bypasses autograd — materializing (θ−θ*)^2 over a 2B-param model is ~15 GB of
    intermediates, and the analytic grad is exact. MUST be called AFTER scaler.unscale_(optimizer) so the
    JEPA grads are already in the true-gradient domain — the added grad is unscaled to match. theta_star =
    pretrained-init CPU clones, moved to GPU one tensor at a time (no full-model 7.4 GB GPU copy).
    """
    device = next(student.parameters()).device
    total = torch.zeros((), device=device)
    for n, p in student.named_parameters():
        if p.requires_grad and n in fisher and n in theta_star:
            diff = p.detach() - theta_star[n].to(device)
            grad = (2.0 * float(lam)) * fisher[n] * diff
            if p.grad is None:
                p.grad = grad
            else:
                p.grad.add_(grad)
            total += (float(lam) * fisher[n] * diff * diff).sum()
    return float(total.item())

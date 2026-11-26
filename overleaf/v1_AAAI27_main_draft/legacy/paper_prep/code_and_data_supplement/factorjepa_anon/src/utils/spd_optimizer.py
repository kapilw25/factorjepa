"""Selective Projection Decay (SPD) optimizer wrapper. CPU/GPU.

Reference:
    Tian, Y. et al. "Selective Projection Decay for Anchor-Aware Fine-Tuning"
    NeurIPS 2024 — arXiv:2411.01713
    https://github.com/gt-ripl/selective-projection-decay

Why SPD over uniform L2 anchor:
    The legacy `compute_drift_loss(student, init_params, λ)` adds `λ‖θ - θ₀‖²`
    to the loss — penalizes ALL parameters uniformly regardless of whether
    their gradient direction is anchor-aware. This causes the Δ2 ≈ 0 trap
    seen in iter13 v10 (anchor-saturation collapse): if λ is large enough to
    prevent forgetting, it's also large enough to freeze surgery into
    pretrain-equivalent → no Δ2 signal.

    SPD replaces uniform L2 with a SELECTIVE per-parameter pull: after the
    AdamW step, for each anchored parameter, check whether the gradient
    direction is ALSO pushing away from anchor. If yes (gradient and
    anchor-displacement are aligned), apply a normalized pull-back toward
    the anchor proportional to alpha_spd × lr. If no (gradient is already
    pulling toward anchor), skip the pull-back — don't fight a gradient
    that's already doing the right thing.

USAGE (used by src/utils/training.py:build_optimizer when cfg_opt.spd.enabled):
    optimizer = SPDAdamW(
        params=model.parameters(),
        anchor_state_dict={name: pretrain_state[name].detach().clone()
                           for name in pretrain_state},
        anchor_param_names=list(model.named_parameters()),  # for tensor→name map
        alpha_spd=0.05,             # 0.0 = pure AdamW (legacy)
        lr=1e-5, weight_decay=0.01,
    )

Backward-compat: alpha_spd=0.0 → SPD post-step is a no-op → bit-identical
to torch.optim.AdamW (verified by numerical test in T5 smoke).

iter14 recipe-v3 audit + intervention #4 (2026-05-09).
See iter/iter14_surgery_on_pretrain/plan_surgery_wins.md §0 row 4️⃣ + §4 #4.
"""
import sys
from typing import Optional

import torch
from torch.optim import AdamW


class SPDAdamW(AdamW):
    """AdamW with Selective Projection Decay applied as a post-update pull-back.

    Args (in addition to AdamW):
        anchor_state_dict: optional {name: tensor} mapping parameter NAMES to
            the anchor (pretrain-weight) tensor. Detached + cloned by caller.
        anchor_param_names: optional list of (name, tensor) — used to build
            the param-id → anchor mapping at init time. Caller usually passes
            list(model.named_parameters()).
        alpha_spd: SPD strength. 0.0 = pure AdamW (legacy). Typical: 1e-3 to 1e-1.

    Step semantics:
        1. Snapshot p_pre for each anchored parameter (small mem cost).
        2. Standard AdamW step (super().step()) — updates p in place.
        3. If alpha_spd > 0.0: for each anchored parameter `p`:
           - actual step = p − p_pre   (post-AdamW displacement vector)
           - disp_post   = p − anchor   (post-step distance from anchor)
           - selectivity: if <step, disp_post> > 0, the AdamW step moved p
             FURTHER from anchor → apply normalized pull-back:
                  p ← p − alpha_spd · lr · disp_post / (‖disp_post‖₂ + 1e-8)
           - else: skip (AdamW step already pulled toward anchor — don't
             fight a gradient that's doing the right thing).

        The per-parameter inner product is a scalar reduction over the
        entire parameter tensor (matches the paper's per-tensor selectivity
        granularity).

    State: stores anchor tensors indexed by id(param). Caller must NOT
    re-allocate parameter tensors between optimizer construction and step()
    calls (rare in practice — only happens with model.to(device) AFTER
    optimizer.__init__, which is a known footgun for any param-id-based
    optimizer).
    """

    def __init__(self, params, anchor_state_dict: Optional[dict] = None,
                 anchor_param_names: Optional[list] = None,
                 alpha_spd: float = 0.0,
                 **adamw_kwargs):
        super().__init__(params, **adamw_kwargs)
        self.alpha_spd = float(alpha_spd)
        # _anchor maps id(param_tensor) -> anchor_tensor (detached, frozen).
        # Built only if alpha_spd > 0 AND caller provided both anchor sources.
        self._anchor: dict = {}
        if self.alpha_spd > 0.0:
            if anchor_state_dict is None or anchor_param_names is None:
                sys.exit("❌ FATAL: SPDAdamW with alpha_spd > 0 requires "
                         "anchor_state_dict (dict) AND anchor_param_names "
                         "(list of (name, tensor)). Got "
                         f"anchor_state_dict={'set' if anchor_state_dict else 'None'}, "
                         f"anchor_param_names={'set' if anchor_param_names else 'None'}.")
            n_matched = 0
            for name, p in anchor_param_names:
                if name in anchor_state_dict:
                    a = anchor_state_dict[name]
                    if a.shape != p.shape:
                        sys.exit(f"❌ FATAL: SPD anchor shape mismatch for '{name}': "
                                 f"anchor {tuple(a.shape)} vs param {tuple(p.shape)}")
                    self._anchor[id(p)] = a.detach().to(p.device).clone()
                    n_matched += 1
            if n_matched == 0:
                sys.exit("❌ FATAL: SPDAdamW could not match ANY parameter to "
                         "anchor_state_dict — likely a name-mismatch bug. Check "
                         "model.named_parameters() vs anchor_state_dict keys.")
            self._n_anchored = n_matched
        else:
            self._n_anchored = 0

    @torch.no_grad()
    def step(self, closure=None):
        # 1. When SPD is active, snapshot p_pre BEFORE the AdamW step so we
        #    can compute the true step direction (= p_post - p_pre). This
        #    avoids the gradient-vs-step sign ambiguity that affected an
        #    earlier draft of this code (where <grad, disp> had the wrong
        #    sign correspondence to "moved away from anchor"). Memory cost:
        #    one extra tensor copy per anchored param, only during step().
        pre_step: dict = {}
        if self.alpha_spd > 0.0 and self._anchor:
            for group in self.param_groups:
                for p in group["params"]:
                    aid = id(p)
                    if aid in self._anchor and p.grad is not None:
                        pre_step[aid] = p.detach().clone()

        # 2. Standard AdamW update — bit-identical to torch.optim.AdamW.
        loss = super().step(closure)

        # 3. SPD post-step pull-back (no-op when alpha_spd == 0 or empty anchor map).
        if self.alpha_spd > 0.0 and self._anchor:
            for group in self.param_groups:
                lr = group["lr"]
                for p in group["params"]:
                    aid = id(p)
                    if aid not in self._anchor or aid not in pre_step:
                        continue
                    anchor = self._anchor[aid]
                    p_pre = pre_step[aid]
                    step_vec = p - p_pre                     # actual AdamW step direction
                    disp_post = p - anchor                   # post-step distance from anchor
                    # Selectivity: did the step push us AWAY from anchor?
                    # <step, disp_post> > 0  ⇔  step in direction of disp_post  ⇔  moved away
                    selector = (step_vec * disp_post).sum()
                    if selector.item() > 0:
                        disp_norm = disp_post.norm().clamp(min=1e-8)
                        p.sub_(self.alpha_spd * lr * disp_post / disp_norm)
        return loss

    def __repr__(self):
        return (f"SPDAdamW(alpha_spd={self.alpha_spd}, "
                f"n_anchored={self._n_anchored}, "
                f"groups={len(self.param_groups)})")

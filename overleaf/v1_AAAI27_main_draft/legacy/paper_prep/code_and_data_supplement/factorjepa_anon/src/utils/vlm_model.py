"""VJepaLlavaVLM — V-JEPA 2.1 (frozen) → spatial-pool → MLP projector → Qwen3-8B (+LoRA in stage 2).

V-JEPA 2's OWN MLLM recipe (arxiv 2506.09985: align the encoder to an LLM with LLaVA visual-
instruction-tuning — per-patch embeddings projected to LLM token space via an MLP). LLaVA-1.5
2-layer GELU projector (github.com/haotian-liu/LLaVA). The frozen encoder is the ONLY difference
between the FROZEN-arm and OURS-arm VLM → a controlled encoder swap.

Runs on the 96GB box (Qwen3-8B + V-JEPA-G). SpatialPoolProjector + encode_video() are testable in
isolation on the 3060 (encoder ~2 GB fits); the full LLM fusion/forward needs the big box, so the
LLM is only loaded when load_llm=True.

No hardcoded values — every dim/id/hparam arrives via the merged configs/vlm.yaml (CLAUDE.md).
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src")
from utils.predictor_eval import load_encoder_only

VIDEO_TOKEN = "<video>"


def build_chat(tokenizer, user_prompt, answer, enable_thinking):
    """Chat-templated token ids for an instruct LLM. user_prompt CONTAINS the <video> token (survives the
    template as text → its special id after tokenize). enable_thinking=False primes an empty <think></think>
    so the model answers directly — a FAITHFUL encoder readout (CoT would let the LLM guess from text priors
    and mask the OURS-vs-FROZEN gap). Returns (input_ids, labels): labels = -100 on the prompt, answer ids
    (+eos) on the answer; answer=None → (input_ids, None) for generation. Falls back gracefully on tokenizers
    without enable_thinking (e.g. Qwen2)."""
    msgs = [{"role": "user", "content": user_prompt}]
    try:
        s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                          enable_thinking=enable_thinking)
    except TypeError:
        s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    p_ids = tokenizer(s, add_special_tokens=False).input_ids
    if answer is None:
        return p_ids, None
    a_ids = tokenizer(str(answer).strip(), add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    return p_ids + a_ids, [-100] * len(p_ids) + a_ids


class SpatialPoolProjector(nn.Module):
    """(B, grid_t*grid_hw², D_concat) V-JEPA tokens → (B, grid_t*pool_hw², hidden) LLM tokens.
    Spatial 24×24→8×8 avg-pool (temporal grid kept — V-JEPA's perf is temporal-sensitive), then a
    LLaVA-1.5 2-layer GELU MLP. Pure-torch, no encoder/LLM → unit-testable on CPU."""

    def __init__(self, in_dim, hidden, grid_t, grid_hw, pool_hw, activation, depth):
        super().__init__()
        self.grid_t, self.grid_hw, self.pool_hw = grid_t, grid_hw, pool_hw
        act = {"gelu": nn.GELU}[activation]()
        assert depth == 2, f"projector depth {depth} unsupported (LLaVA-1.5 = 2)"
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), act, nn.Linear(hidden, hidden))

    def forward(self, tokens):
        B, N, D = tokens.shape
        exp = self.grid_t * self.grid_hw * self.grid_hw
        if N != exp:
            raise ValueError(f"token count {N} != grid_t*grid_hw² ({exp}) — check encoder/frames")
        x = tokens.view(B, self.grid_t, self.grid_hw, self.grid_hw, D).permute(0, 1, 4, 2, 3)   # B,T,D,H,W
        x = x.reshape(B * self.grid_t, D, self.grid_hw, self.grid_hw)
        x = F.adaptive_avg_pool2d(x, (self.pool_hw, self.pool_hw))                                # B*T,D,ph,pw
        x = x.reshape(B, self.grid_t, D, self.pool_hw * self.pool_hw).permute(0, 1, 3, 2)         # B,T,pp,D
        x = x.reshape(B, self.grid_t * self.pool_hw * self.pool_hw, D)                            # B,512,D
        return self.mlp(x)


class VJepaLlavaVLM(nn.Module):
    """Full VLM. arm ∈ {frozen, ours} selects the (only-differing) encoder ckpt."""

    def __init__(self, cfg, arm, load_llm=True, lora=False):
        super().__init__()
        v = cfg["vlm"]
        enc = v["encoder"]
        if arm not in enc["arms"]:
            raise KeyError(f"arm '{arm}' not in vlm.encoder.arms {list(enc['arms'])}")
        self.encoder, _ckpt, embed_concat = load_encoder_only(
            enc["arms"][arm], enc["num_frames"], enc["model_config"])
        if embed_concat != enc["embed_dim_concat"]:
            sys.exit(f"FATAL: encoder concat {embed_concat} != cfg embed_dim_concat {enc['embed_dim_concat']}")
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        tk = v["tokens"]
        self.n_video_tokens = tk["n_video_tokens"]
        if self.n_video_tokens != tk["grid_t"] * tk["pool_hw"] ** 2:
            sys.exit(f"FATAL: n_video_tokens {self.n_video_tokens} != grid_t*pool_hw²")

        # LLM FIRST → projector out-dim = the LOADED model's hidden (dynamic; swapping the LLM is 1 config line).
        self.tokenizer = self.llm = self.video_token_id = None
        self.enable_thinking = v["llm"]["enable_thinking"]
        llm_hidden = v["llm"]["hidden_size"]                       # fallback for load_llm=False shape tests only
        if load_llm:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            mid = v["llm"]["model_id"]
            self.tokenizer = AutoTokenizer.from_pretrained(mid)
            if VIDEO_TOKEN not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({"additional_special_tokens": [VIDEO_TOKEN]})
            self.video_token_id = self.tokenizer.convert_tokens_to_ids(VIDEO_TOKEN)
            self.llm = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda")
            self.llm.resize_token_embeddings(len(self.tokenizer))
            llm_hidden = self.llm.config.hidden_size              # source of truth (read BEFORE the LoRA wrap)
            for p in self.llm.parameters():
                p.requires_grad_(False)
            if lora:
                from peft import LoraConfig, get_peft_model
                lc = v["lora"]
                self.llm = get_peft_model(self.llm, LoraConfig(
                    r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                    target_modules=lc["target_modules"], task_type="CAUSAL_LM"))

        pj = v["projector"]
        self.projector = SpatialPoolProjector(
            embed_concat, llm_hidden, tk["grid_t"], tk["grid_hw"], tk["pool_hw"],
            pj["activation"], pj["depth"]).to("cuda", dtype=torch.bfloat16)

    def encode_video(self, pixels):
        """pixels (B,C,T,H,W) bf16 cuda → (B, n_video_tokens, hidden). Encoder is no-grad; projector trains."""
        with torch.no_grad():
            h = self.encoder(pixels)
            if isinstance(h, (list, tuple)):
                h = torch.cat(list(h), dim=-1)
        return self.projector(h)

    def _fuse(self, video_embeds, input_ids, attention_mask, labels):
        """LLaVA merge: replace the single <video> token per sample with its n_video_tokens rows.
        Returns right-padded (inputs_embeds, attention_mask, labels). github.com/haotian-liu/LLaVA
        (prepare_inputs_labels_for_multimodal)."""
        embed = self.llm.get_input_embeddings()
        B, H = input_ids.shape[0], video_embeds.shape[-1]
        seqs, masks, lbls = [], [], []
        for b in range(B):
            ids = input_ids[b]
            pos = (ids == self.video_token_id).nonzero(as_tuple=True)[0]
            if len(pos) != 1:
                raise ValueError(f"sample {b}: expected exactly one <video> token, got {len(pos)}")
            p = int(pos)
            te = embed(ids)                                          # (L, H)
            vid = video_embeds[b].to(te.dtype)                       # (nv, H)
            seqs.append(torch.cat([te[:p], vid, te[p + 1:]], 0))
            am = attention_mask[b]
            masks.append(torch.cat([am[:p], torch.ones(self.n_video_tokens, device=am.device, dtype=am.dtype), am[p + 1:]]))
            if labels is not None:
                lb = labels[b]
                ig = torch.full((self.n_video_tokens,), -100, device=lb.device, dtype=lb.dtype)
                lbls.append(torch.cat([lb[:p], ig, lb[p + 1:]]))
        L = max(s.shape[0] for s in seqs)
        ie = torch.zeros(B, L, H, device=seqs[0].device, dtype=seqs[0].dtype)
        am = torch.zeros(B, L, device=masks[0].device, dtype=masks[0].dtype)
        lb = torch.full((B, L), -100, device=seqs[0].device, dtype=torch.long) if labels is not None else None
        for b in range(B):
            n = seqs[b].shape[0]
            ie[b, :n], am[b, :n] = seqs[b], masks[b]                 # right-pad
            if labels is not None:
                lb[b, :n] = lbls[b]
        return ie, am, lb

    def forward(self, pixels, input_ids, attention_mask, labels):
        """Training forward → causal-LM loss on the answer tokens only (video+prompt = -100)."""
        vid = self.encode_video(pixels)
        ie, am, lb = self._fuse(vid, input_ids, attention_mask, labels)
        return self.llm(inputs_embeds=ie, attention_mask=am, labels=lb)

    @torch.no_grad()
    def generate(self, pixels, input_ids, attention_mask, max_new_tokens=32):
        """Eval: greedy decode the answer given the fused video+prompt. input_ids end at 'Answer:'."""
        vid = self.encode_video(pixels)
        ie, am, _ = self._fuse(vid, input_ids, attention_mask, labels=None)
        out = self.llm.generate(inputs_embeds=ie, attention_mask=am, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def trainable_parameters(self, stage):
        """align → projector only; instruct → projector + LoRA. Encoder always frozen."""
        params = list(self.projector.parameters())
        if stage == "instruct" and self.llm is not None:
            params += [p for p in self.llm.parameters() if p.requires_grad]
        return params

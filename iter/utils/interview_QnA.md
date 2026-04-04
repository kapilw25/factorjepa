# FactorJEPA — Interview Q&A with Real-Life Analogies

> Quick-recall cheat sheet for research scientist interviews.
> Each question has: (1) the technical answer in 2-3 lines, (2) a real-life analogy you can explain to anyone.
> Add new questions as they come up during the project.

---

## Q1: What issues did you face creating embeddings for a pretrained/adapted V-JEPA model? How did you fix them?

### The one-liner

> "The frozen HF model is like an all-inclusive resort — everything handled for you. The adapted model is like camping — you bring the wrong tent size (64 vs 16 frames), your flashlight has the wrong batteries (float32 vs float16), and there's a gate across the trail that does nothing but slows you down (deprecated sdp_kernel). Once we matched every parameter to the training config instead of the HF defaults, throughput went from crashing to 15 clips/sec."

### Technical summary (for follow-ups)

The frozen model uses HF `AutoModel` which handles frame count, dtype, and attention internally. The adapted model uses Meta's native `vit_giant_xformers()` — we had to manually match every parameter to the training config. 7 bugs fixed across ~8 OOM crashes in one session.

### The 7 issues as "The Wrong Uniform Problem"

Think of the adapted V-JEPA model as a **basketball player who transferred from another team**. The frozen model is a player who's been on YOUR team since day one.

**1. Wrong jersey size (ROOT CAUSE — 64 frames vs 16)**

> You hired a player from the NBA (HF model uses 64-frame videos). But your team plays street basketball with 16-frame clips. Nobody told the equipment manager, so he gave the player an NBA-sized jersey (64 frames). The player could barely move (4x more data to process = OOM). **Fix: check the team's own playbook (training config), not the NBA rulebook.**

**2. Unnecessary security checkpoints (sdp_kernel graph breaks)**

> The player had to pass through a metal detector every time he crossed half-court — a legacy rule from the old arena that does literally nothing. But it broke the flow of the game (torch.compile couldn't optimize across these stops). **Fix: removed the decorative metal detectors.**

**3. Wrong currency at the door (float32 vs float16)**

> The player showed up with dollars but the arena only accepts euros. The cashier had to convert currency AND keep the dollars — using two wallets instead of one (both compiled graphs in memory). **Fix: convert to euros before arriving.**

**4. Wearing the jersey backwards (wrong tensor permute)**

> The warmup jersey was put on correctly (C, T, H, W), then someone helpfully flipped it inside out. The coach saw "64" on the back instead of "3" and rejected the player. **Fix: stop "helping" — it was already right.**

**5. No warmup before the game (missing torch.compile warmup)**

> The player walked straight onto the court cold — first play took forever and he cramped up. **Fix: do a quick shootaround first (BS=2 warmup).**

**6. Too many fans in the stadium (CPU memory SIGKILL)**

> With the oversized 64-frame jerseys, each fan took 4x as much space. 8 rows x 176 seats x 4x = 152GB of fans in a 120GB stadium. Fire marshal shut it down (SIGKILL). **Fix: once jerseys were right-sized (#1), fans fit fine.**

**7. Couldn't clean the court after a spill (OOM retry failure)**

> After the first OOM crash, the janitor tried to mop up (empty_cache), but the torch.compile debris was bolted to the floor. Second attempt also slipped. **Fix: the other fixes prevented the spill entirely.**

### Why the frozen model never had these issues

| Aspect | Frozen = All-inclusive resort | Adapted = Camping |
|--------|-----|------|
| Frame count | Hotel knows your room size | You brought the wrong tent |
| Attention | Elevator works automatically | Gate across the trail (sdp_kernel) |
| torch.compile | Concierge handles everything | You assemble it yourself |
| Input dtype | Room service converts for you | Wrong batteries in your flashlight |
| Model loading | Check in with your name | Build the cabin from blueprints |

### Result

- **Before:** OOM at BS=176 (89GB GPU + 152GB CPU), 0 clips processed
- **After:** 15.3 clips/s at BS=176, ~30GB VRAM, ~2.1h for 115K clips (10x faster than crashing)

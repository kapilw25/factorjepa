# 🧪 iter15 — Head-only vs Encoder-update surgery (paper §3 Method figures)
> ## 🎯 Paper goal:  `vjepa_surgery` [X_epochs(surgery) +X_epochs(pretrain)] ≫ `vjepa_pretrain` [2X epochs] ≫ `vjepa_frozen` on motion / temporal features
> 🎯 Claim: `head-only surgery` ≈ `encoder-update surgery` on motion features ⇒ 1/40× GPU.
> Diagrams only — paper-figure aesthetic. One concept per diagram.

---

## § 1 — 🔬 Research question

```mermaid
flowchart LR
    Q["❓ Does factor surgery still help<br/>🧊 when the backbone is frozen?"]
    Q --> H1["🧪 Δ1: 🏋️ continual SSL &gt; 🧊 frozen<br/>✅ proven (iter13)"]
    Q --> H2["🧪 Δ2 / Δ3: 🔧 surgery &gt; 🏋️ pretrain · 🔁 pretrain_2X<br/>✅ proven (iter14 recipe-v3)"]
    Q --> H3["⭐ Δ5 / Δ6 / Δ7: 🧠 head-only ≈ 🔓 encoder-update<br/>🎯 iter15 headline"]
    classDef node fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b,stroke-width:1.5px
    class Q,H1,H2,H3 node
```

---

## § 2 — 🧬 Encoder zoo (5 arms compared at eval)

```mermaid
flowchart LR
    M["🦣 V-JEPA 2.1 ViT-G<br/>🧮 1.84 B · 🏛️ Meta init"]
    M --> A0["0️⃣ 🧊 frozen<br/>🎯 zero-shot baseline"]
    M --> A1["1️⃣ 🏋️ pretrain · 5 ep<br/>🔄 continual SSL"]
    M --> A2["2️⃣ 🔁 pretrain_2X · 10 ep<br/>⚖️ compute control"]
    A1 --> A3["3️⃣ 🔧 surgery_3stage_DI · 5 ep<br/>🧩 D_L → D_A → D_I"]
    A1 --> A4["4️⃣ ✂️ surgery_noDI · 5 ep<br/>🧩 D_L → D_A"]
    classDef node fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b,stroke-width:1.5px
    class M,A0,A1,A2,A3,A4 node
```

---

## § 3 — 🔗 Sequential composition + paired-Δ tests

```mermaid
flowchart LR
    F["🧊 frozen"]
    P["🏋️ pretrain"]
    P2["🔁 pretrain_2X"]
    S["🔧 surgery"]
    F -.->|"🧪 Δ1"| P
    P -.->|"🧪 Δ2"| S
    P2 -.->|"⭐ Δ3 · 🎯 causal"| S
    classDef node fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b,stroke-width:1.5px
    class F,P,P2,S node
```

---

## § 4 — 🧬 Recipe-v3 winning recipe (iter14 R1 · 🏆 top-1 = 0.8456)

```mermaid
flowchart LR
    subgraph Teacher["🧊 Teacher"]
        T["🧊 SALT · frozen pretrain encoder<br/>🚫 no EMA · 🚫 no updates"]
    end
    subgraph Student["🔓 Student"]
        H["🧠 LP-FT Stage 0 · head-only warmup"]
        L["✂️ Surgical subset · 4 / 8 / 8 blocks"]
        W["🔥 Single warmup over total budget"]
        O["🛡️ SPD optimizer · selective projection decay"]
    end
    subgraph Data["📥 Data"]
        FV["🧩 factor views · D_L · D_A · D_I"]
        R["🔁 50% raw mp4 replay · CLEAR"]
    end
    subgraph Loss["🎯 Loss"]
        J["🎯 saliency-weighted JEPA"]
        AA["🧠 motion_aux head · CE + MSE"]
    end
    FV --> Student
    R --> Student
    T -->|"📡 target latents"| J
    Student -->|"📡 predicted latents"| J
    Student --> AA
    classDef node fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b,stroke-width:1.5px
    classDef sg fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b
    class T,H,L,W,O,FV,R,J,AA node
    class Teacher,Student,Data,Loss sg
```

---

## § 5 — 🧪 iter15 6-cell paired-Δ matrix

```mermaid
flowchart LR
    subgraph EU["🔓 encoder-update · ViT-G backward · ⏱️ ~80 min / cell"]
        D["🅳 🏋️ pretrain_encoder<br/>📦 m09a1"]
        E["🅴 🔧 surg_3stage_DI_enc<br/>📦 m09c1"]
        Ff["🅵 ✂️ surg_noDI_enc<br/>📦 m09c1"]
    end
    subgraph HO["🧊 head-only · encoder frozen · ⚡ ~9 min / cell"]
        A["🅰️ 🏋️ pretrain_head<br/>📦 m09a2"]
        B["🅱️ 🔧 surg_3stage_DI_head<br/>📦 m09c2"]
        C["🅲 ✂️ surg_noDI_head<br/>📦 m09c2"]
    end
    D -.->|"🧪 Δ6"| A
    E -.->|"⭐ Δ5 headline"| B
    Ff -.->|"🧪 Δ7"| C
    classDef node fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b,stroke-width:1.5px
    classDef sg fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b
    class D,E,Ff,A,B,C node
    class EU,HO sg
```

---

## § 6 — 📊 Paired-Δ tests (paper §4 Results)

```mermaid
flowchart LR
    subgraph Tests["📐 paired BCa 95% CI · 🎲 10K resample"]
        D5["⭐ Δ5 = 🔧🔓 surg_DI_enc − 🔧🧊 surg_DI_head"]
        D6["🧪 Δ6 = 🏋️🔓 pretrain_enc − 🏋️🧊 pretrain_head"]
        D7["🧪 Δ7 = ✂️🔓 surg_noDI_enc − ✂️🧊 surg_noDI_head"]
    end
    D5 --> O5{"🔍 sign of Δ5"}
    O5 -->|"🟰 ≈ 0 · CI ∋ 0"| W["🟢 🧠 head-only WINS<br/>⚡ 1/40× GPU"]
    O5 -->|"➕ Δ5 &gt; 0"| Enc["🔵 🔓 encoder margin"]
    O5 -->|"➖ Δ5 &lt; 0"| Hd["🔴 🧠 head outperforms"]
    classDef node fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b,stroke-width:1.5px
    classDef sg fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b
    class D5,D6,D7,O5,W,Enc,Hd node
    class Tests sg
```

---

## § 7 — 🔭 Probe-trio evaluation protocol

```mermaid
flowchart LR
    enc["🧬 encoder<br/>(any arm)"]
    enc -->|"🎞️ N = 1000 val clips"| pool["🧮 pooled features"]
    pool --> p1["🎯 probe_top1<br/>🧪 LOOCV kNN · 🏷️ 14 motion cls"]
    pool --> p2["🧭 motion_cos<br/>📐 cos(feat, RAFT flow)"]
    pool --> p3["⏭️ future_l1<br/>📏 next-frame latent L1"]
    p1 --> headline["⭐ 🏆 paper headline metric"]
    classDef node fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b,stroke-width:1.5px
    class enc,pool,p1,p2,p3,headline node
```

---

## § 8 — 🧠 motion_aux head (auxiliary supervision)

```mermaid
flowchart LR
    raw["🎬 raw mp4 clip"] -->|"🌊 m04d · RAFT optical flow"| tgt["🎯 motion target<br/>🏷️ K-cls label + 📏 D-vec"]
    enc["🧬 encoder pooled feats"] --> mh["🧠 motion_aux head"]
    mh --> ce["🏷️ CE loss"]
    mh --> mse["📏 MSE loss"]
    tgt --> ce
    tgt --> mse
    ce --> L["⚖️ α · CE + β · MSE"]
    mse --> L
    classDef node fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b,stroke-width:1.5px
    class raw,tgt,enc,mh,ce,mse,L node
```

---

## § 9 — 🏗️ Training-loop schematic (m09a / m09c × encoder / head)

```mermaid
flowchart TB
    subgraph A1["📦 m09a1 · 🏋️🔓 pretrain_encoder"]
        a1a["🦣 ViT-G · 🔓 full backward"] --> a1b["🧠 motion_aux head"]
    end
    subgraph A2["📦 m09a2 · 🏋️🧊 pretrain_head"]
        a2a["🦣 ViT-G · 🧊 frozen"] --> a2b["🧠 motion_aux head ✅ trains"]
    end
    subgraph C1["📦 m09c1 · 🔧🔓 surgery_encoder"]
        c1a["🦣 ViT-G · ✂️ stage-gated backward<br/>(subset 4 / 8 / 8)"] --> c1b["🧠 motion_aux head"]
        c1a -->|"🎯 saliency-weighted JEPA"| c1t["🧊 SALT teacher"]
    end
    subgraph C2["📦 m09c2 · 🔧🧊 surgery_head"]
        c2a["🦣 ViT-G · 🧊 frozen"] --> c2b["🧠 motion_aux head ✅ trains"]
    end
    classDef node fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b,stroke-width:1.5px
    classDef sg fill:#e0e7ff,stroke:#4338ca,color:#1e1b4b
    class a1a,a1b,a2a,a2b,c1a,c1b,c1t,c2a,c2b node
    class A1,A2,C1,C2 sg
```

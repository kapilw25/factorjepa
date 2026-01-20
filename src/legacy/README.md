
## Quick Start

```bash
# Setup
chmod +x setup_env.sh
./setup_env.sh
source venv_3Denv/bin/activate
```

---

## Commands

### Phase 0: Algorithmic Judge + LLM (M1 Mac / CPU)

| Command | Description |
|---------|-------------|
| `python src/m01_shortest_path.py --test` | Test BFS, Dijkstra, A* algorithms |
| `python src/m01_shortest_path.py --demo` | Demo with path visualization |
| `python src/m02_hybrid_judge.py --test` | Test algorithmic judge |
| `python src/m02_hybrid_judge.py --compare` | Compare algo vs expected |
| `python src/m02_hybrid_judge.py --demo` | Demo evaluation (algo only) |
| `python src/m02_hybrid_judge.py --demo --llm` | Demo with LLM explanations |

### Phase 1: VLM + AI2-THOR

**GPU Server (AI2-THOR requires GPU + X11 display)**
| Command | Description |
|---------|-------------|
| `python src/m03_ai2thor_capture.py --list` | List available AI2-THOR scenes |
| `python src/m03_ai2thor_capture.py --scenes 5` | Capture 5 POC scenes |
| `python src/m03_ai2thor_capture.py --scene FloorPlan1` | Capture single scene |

**M1 Mac / CPU (API-based, no GPU required)**
| Command | Description |
|---------|-------------|
| `python src/m04_vlm_agent.py --test` | Test VLM agent with sample image |
| `python src/m04_vlm_agent.py --image <path> --task "bed to lamp"` | Generate path from image |
| `python src/m05_vlm_pipeline.py --demo` | End-to-end VLM demo |
| `python src/m05_vlm_pipeline.py --scene FloorPlan1 --task "bed to lamp"` | Evaluate single scene |
| `python src/m02_hybrid_judge.py --vlm --image data/images/ai2thor/FloorPlan1/` | VLM evaluation mode |

---

## References

- [Current Plan](iter/iter1/plan1.md)
- [Proposal](Literature/proposal/Proposal_LLMs_as_Agents.pdf)
- [Experimental Framework](Literature/proposal/experimental_framework.pdf)

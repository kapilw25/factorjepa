# FactorJEPA profile for /audit-judge-pattern

Read by `.claude/skills/audit-judge-pattern/judge_pattern_check.py` (the ```json block) and by the skill (the table).

| Key | FactorJEPA value |
|---|---|
| Workshop folder | `overleaf/v4_AAAI_workshop` |
| Judge files | `feedback/output/<draft or round>/by_<judge>/step2.md` (rounds: v1_Human_Centric, v2_Cog2Soc, v3_open_venues, ...) |
| Run the check | from the workshop folder: `python3 ../../.claude/skills/audit-judge-pattern/judge_pattern_check.py v4_Data2AgentEx/0_main.pdf feedback/output/*/by_*/step2.md` (activate `venv_walkindia` first: a hook refuses bare python3) |
| Build after a fix | `python -I build_all.py` from the workshop folder (all unfrozen drafts, every gate) |
| Decision rows | the workshop `TODOs.md` |
| Hard limits a rewrite keeps | numbers only as `common/numbers.tex` macros (drift gate), 4 body pages, anonymity, no em-dashes, no semicolons in new prose |
| Gate in the build | not yet: run by hand (build_all.py does not call it) |

```json
{
  "work_names": ["FactorJEPA", "DENSEWORLD-115k"],
  "anon_banner": null,
  "abstract_end": "\\n\\s*\\n"
}
```

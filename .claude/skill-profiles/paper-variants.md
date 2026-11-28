# FactorJEPA profile for /paper-variants

| Key | FactorJEPA value |
|---|---|
| 📁 Workshop folder | `overleaf/v4_AAAI_workshop` |
| 📋 Venues | `overleaf/v4_AAAI_workshop/venues.md` (deadlines, limits, review mode) |
| 🟡 Venue drafts | `v1_Human_Centric`, `v3_Cog2Soc_fixes`, `v4_Data2AgentEx`, `v5_Continual_Causality` |
| 🔒 Frozen | `v2_Cog2Soc` (has `FROZEN`) |
| 🟢 Shared files | `common/`: preamble, numbers, disclosure, links, sec_data, sec_method, sec_eval, sec_scope, fig_teaser, tab_composition, figs, results, factorjepa.bib |
| 🔧 Build | `source venv_walkindia/bin/activate && python -I overleaf/v4_AAAI_workshop/build_all.py > $TMPDIR/b.log 2>&1` (from the repo root) |
| ✅ Gates in the build | build errors and undefined refs, page limit from `venue.tex`, `??` and em-dash glyphs in the PDF, anonymity, disclosure footnote, drift (stale copy, pasted 40-word paragraph, typed number), AI signatures, one title per draft |
| 🔢 Numbers | `common/numbers.tex` macros only; results JSON in `common/results/` |
| 🔀 Switches | `\input{venue}` per draft; `\disclosure` text switched per venue |
| 🧾 Cross-reference style | `Section~4` written out (house style) |
| 🚀 Push | only after a clean build: `echo 1 \| bash git_push.sh "<plain message>"` |
| ❌ Not run by the build | /audit-judge-pattern and /audit-robotic-tone (run by hand) |

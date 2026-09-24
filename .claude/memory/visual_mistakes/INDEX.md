---
name: visual-mistakes-index
description: One-line index of every logged figure/plot/demo mistake; load the individual VM<n>.md only when its row matches the task
type: project
---

# Visual mistakes KB — index

Append-only. Each row is one file `VM<n>.md`. The PreToolUse hook `surface-visual-mistakes.sh` injects the matching rows when you edit a figure/plot; open the full file only when relevant.

| id | category | mistake |
|----|----------|---------|
| VM30 | demo-readability | a probe win on a TECHNICAL label ≠ a layman-verifiable demo (GT anti-correlates with perce |
| VM29 | demo-readability | the eval metric wins are REAL but SUB-PERCEPTUAL: a naked-eye layman demo is not honestly  |
| VM27 | demo-readability | audit agent FALSE-APPROVED a layman claim it pattern-matched from the captions (2026-07-14 |
| VM28 | latex-figure-placement | forced a LATENT-space metric into a PIXEL demo it cannot show (2026-07-14) |
| VM26 | plot-authoring | title-card legend line orphaned by a design swap of the element it describes (2026-07-14) |
| VM25 | demo-readability | crop-scale evidence is unusable when generative output is foggy; compare at GESTALT scale  |
| VM24 | demo-readability | numerically-decisive ROI is visually MISLEADING when fog hue coincides with the object (20 |
| VM21 | demo-readability | verdict/summary rows ship placeholder glyphs instead of values (2026-07-14) |
| VM22 | demo-readability | hero clip chosen without checking the visual evidence is decisive (2026-07-14) |
| VM23 | demo-readability | fog-vs-fog: a colour-temperature difference is NOT a layman-visible win (2026-07-14) |
| VM1 | latex-figure-placement | winner-arm annotations struck through the metric labels (2026-07-12) |
| VM2 | plot-authoring | suptitle wider than the plot → wasted flank space (2026-07-12) |
| VM3 | latex-figure-placement | multi-panel comparison rendered side-by-side landscape (2026-07-12) |
| VM4 | latex-figure-placement | value label at an edge-hugging marker ran into the RHS column (2026-07-12) |
| VM5 | plot-authoring | compressed vertical panels made per-row annotations collide (2026-07-12) |
| VM7 | demo-readability | metric name in scene title contradicts the on-frame unit label (2026-07-12) |
| VM8 | plot-authoring | chart y-axis label drawn over the neighbouring image panel (2026-07-12) |
| VM9 | plot-authoring | sequential dark cmap + black bold cell text = illegible endpoint cells (2026-07-12) |
| VM10 | demo-readability | identical duplicate row/col labels for distinct entities (2026-07-12) |
| VM11 | plot-authoring | plotted element with no legend entry (dashed fit line) (2026-07-12) |
| VM12 | demo-readability | outro/verdict references entities never introduced in the demo (2026-07-12) |
| VM13 | latex-figure-placement | single-model rendering of a multi-model layout leaves a dead half-frame (2026-07-12) |
| VM19 | demo-readability | mask caption describes a render style the renderer does not produce (2026-07-14) |
| VM20 | latex-figure-placement | video frame layout leaves a dead horizontal band between content strip and pinned footer ( |
| VM15 | plot-authoring | CI whisker strikes the value label; a struck minus reads as PLUS (2026-07-13) |
| VM16 | latex-figure-placement | in-axes annotation overflows and is struck by the axes spine (2026-07-13) |
| VM17 | plot-authoring | same-titled sibling panels computed on DIFFERENT arm rosters (2026-07-13) |
| VM18 | plot-authoring | blanket suptitle claim false in one instantiation (2026-07-13) |
| VM14 | plot-authoring | same entity titled DIFFERENTLY across sibling figures (2026-07-13) |
| VM6 | plot-authoring | hardcoded arm name in a figure title (2026-07-12) |
| VM31 | latex-figure-placement | color-emoji tofu in PIL/DejaVu video renders |
| VM32 | demo-readability | overlay/text detector misses text baked over MOVING footage → human-eyeball gate is mandat |
| VM33 | latex-figure-placement | full-width figure* caterpillar rendered near-SQUARE → y-tick arm NAMES fall below the 9pt  |
| VM34 | plot-authoring | a "readability" fix that DROPS / REFORMATS / GROUPS-AWAY spec'd content is a spec VIOLATIO |
| VM35 | latex-figure-placement | non-BOLD value/tick numbers + wasteful WHITE space slipped through because the audit had n |
| VM36 | latex-figure-placement | figsize HEIGHT shrunk but suptitle y / top margin not recomputed -> suptitle 2nd line over |
| VM37 | latex-figure-placement | a C5 font-bump that ignores the AUTHORING-width vs PLACEMENT-width downscale under-deliver |
| VM38 | latex-figure-placement | audit run ONCE not per-edit gate -> placement defects recurred; render+read every changed page before presenting |
| VM39 | latex-figure-placement | raster-cropping one panel out of a multi-panel figure with PDF TEXT-BOX coordinates bled neighbour-panel ink + left a dead lead-in; derive crop edges from ink scans, assert 0 px diff vs source |
| VM40 | latex-figure-placement | illustration rasters pasted at a fraction of authored width carry baked-in text far below 9 pt while the compositor's own 10-pt labels mask it; measure the smallest baked cap per source panel |

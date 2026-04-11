# Mermaid Rules for This Project

## Line Breaks
- **NEVER use `\n`** — renders as literal text
- **ALWAYS use `<br>`** for line breaks inside node labels
- Example: `A["Step 1:<br>Load Data"]`

## Node Style Template
```
style X fill:#HEX,color:#fff,font-weight:bold,font-size:28px
```

## Color Palette (white-text safe)

| Color | Hex | Usage |
|-------|-----|-------|
| Blue | `#1e88e5` | Input / Load data |
| Purple | `#8e24aa` | Loop / Iteration |
| Teal | `#00897b` | Compute / Process |
| Red | `#e53935` | Measure / Evaluate |
| Deep Orange | `#f4511e` | Decision / Branch |
| Deep Purple | `#5e35b1` | Aggregate / Collect |
| Cyan | `#00acc1` | Rank / Sort |
| Green | `#43a047` | Select / Pick |
| Pink | `#d81b60` | Final Output / Result |
| Dark Red | `#b71c1c` | Reject / Exclude |
| Blue Grey | `#546e7a` | Optional / Fallback |
| Brown | `#6d4c41` | External / Reference |

## Supported Types
- USE: `flowchart`, `graph`, `sequenceDiagram`, `classDiagram`, `stateDiagram`, `gantt`, `mindmap`
- AVOID: `quadrantChart`, `sankey`, `xychart` (require newer mermaid versions)

## VS Code Extension
- KEEP: `bierner.markdown-mermaid` only
- UNINSTALL: `mermaidchart.vscode-mermaid-chart`, `vstirbu.vscode-mermaid-preview` (conflict)

## Direction
- Default: `flowchart LR` (left-to-right) for pipelines
- Use `flowchart TD` (top-down) for hierarchies

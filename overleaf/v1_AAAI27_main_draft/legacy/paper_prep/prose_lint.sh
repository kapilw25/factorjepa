#!/usr/bin/env bash
# prose_lint.sh — the TEXT + FIGURE auditor for the AAAI paper (the visual-audit agent reads rendered
# PNGs; this fills the .tex-prose gap AND scans text baked into matplotlib figure PDFs). Run before every
# compile: bash paper_prep/prose_lint.sh
# Flags LLM writing-tells reviewers notice. Exit non-zero if any RENDERED em-dash is found (hard fail).
set -o pipefail
cd "$(dirname "$0")/.." || exit 2          # → overleaf/2026___FactorJEPA_AAAI/
BODY=(1_introduction.tex 2_data.tex 2_factor_jepa.tex 8_conclusion.tex)
APPX=(11_appendix.tex)

echo "════════ PROSE LINT ════════"
fail=0

# 1) EM-DASHES — the #1 AI tell. '---' (LaTeX) and '—' (unicode). LaTeX COMMENTS (% ... to EOL) are
#    STRIPPED first — a '% --------' separator line never renders, so it is not an em-dash the reviewer
#    sees. En-dash '--' for numeric ranges is allowed. BOTH body and appendix must be 0 (rendered).
strip_comments() { perl -pe 's/(?<!\\)%.*$//' "$@"; }
for scope in "BODY:${BODY[*]}" "APPENDIX:${APPX[*]}"; do
  name="${scope%%:*}"; files="${scope#*:}"
  n=$(strip_comments $files 2>/dev/null | grep -oE "\-\-\-|—" | wc -l | tr -d ' ')
  printf "  em-dash rendered (--- / —)  %-9s : %s\n" "$name" "$n"
  [ "$n" -gt 0 ] && fail=1
done
echo "  ── rendered em-dash occurrences, comments excluded (must be 0) ──"
perl -ne 's/(?<!\\)%.*$//; print "$ARGV:$.: $_" if /---|—/; close ARGV if eof' "${BODY[@]}" "${APPX[@]}" 2>/dev/null | head -40

# 2) FIGURE-PDF EM-DASHES — text baked into matplotlib figure PDFs is invisible to a .tex grep (this is
#    how Fig 7's suptitle em-dashes hid for a while). Scan every figure actually \includegraphics'd.
echo "  ── figure-PDF em-dashes (rendered inside plots; must be 0) ──"
USED=$(cat "${BODY[@]}" "${APPX[@]}" 2>/dev/null | tr '\n' ' ' | grep -oE '\{figures/[^}]*\}' | sed -E 's/\{figures\///; s/\}//' | sort -u)
figfound=0
for stem in $USED; do
  f="figures/$stem"
  [ -f "$f" ] || continue
  case "$f" in
    *.pdf)
      m=$(pdftotext "$f" - 2>/dev/null | grep -oE '—' | wc -l | tr -d ' ')
      if [ "$m" -gt 0 ]; then printf "     %-46s %s em-dash(es)\n" "$stem" "$m"; figfound=1; fail=1; fi ;;
  esac
done
[ "$figfound" -eq 0 ] && echo "     (none)"

# 3) AI-TELL WORDS/PHRASES (case-insensitive) across body — report counts, don't hard-fail.
echo "  ── AI-tell words in BODY (review, soften) ──"
telfound=0
for w in "delve" "moreover" "furthermore" "it'?s worth noting" "in conclusion" "a testament" \
         "notably" "importantly" "crucially" "seamless" "realm" "tapestry" "underscore" \
         "leverage" "utilize" "plethora" "meticulous" "showcase" "pivotal"; do
  c=$(grep -oiE "\b$w" "${BODY[@]}" 2>/dev/null | wc -l | tr -d ' ')
  [ "$c" -gt 0 ] && { printf "     %-18s %s\n" "$w" "$c"; telfound=1; }
done
[ "$telfound" -eq 0 ] && echo "     (none)"

echo "════════════════════════════"
[ "$fail" -eq 0 ] && echo "PROSE LINT: PASS (0 rendered em-dashes in prose or figures)" \
                  || echo "PROSE LINT: FAIL (rendered em-dashes present — replace with , : ( ) or regenerate the figure)"
exit "$fail"

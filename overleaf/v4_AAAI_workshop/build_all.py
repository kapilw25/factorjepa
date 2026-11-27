"""Rebuild every unfrozen FactorJEPA workshop draft (paper-variants skill) and run the per-draft gates; one table at the end.
    source venv_walkindia/bin/activate; python -I overleaf/v4_AAAI_workshop/build_all.py 2>&1 | tee logs/build_all_drafts_$(date +%Y%m%d_%H%M%S).log
Gates per draft: build (errors, undefined refs), pages (main text ends by the venue limit from venue.tex), PDF text
('??', em-dash glyph), anonymity (anonymous author line, no identifying names or links in sources or PDF, no PDF dates),
disclosure footnote (used once; names the other three venues, never its own), drift (stale copy of a common/ file, pasted
40-word paragraph, measured number typed in a venue file), AI signatures (.claude/skills/audit-ai-signatures).
Across drafts: every unfrozen draft has its own title (owner pick 2026-11-27).
"""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMMON = HERE / "common"
AI_SIG = ROOT / ".claude/skills/audit-ai-signatures/check_ai_signatures.py"
TEMPLATE_FILES = {"aaai2027.sty", "aaai2027.bst", "0_main.tex", "venue.tex", "FROZEN", "TODOs.md"}
MIN_WORDS = 40  # a venue paragraph this long, identical to one in common/, was pasted instead of \input
# a measured number typed in a venue file instead of a common/numbers.tex macro: decimals, thousands, 3+ digit integers
TYPED = re.compile(r"(?<![\w.\\-])(\d+\.\d+|\d{1,3}(?:,\d{3})+|\d{3,})(?![\w.])")
TYPED_OK = {"2026", "2027"}
IDENTIFYING = re.compile(r"kapilw|wanaskar|github\.com/|arxiv\.org/abs|2608\.01049|factorjepa_anon", re.I)
VENUE_NAMES = {"venueCog": "Cog2Soc", "venueData": "Data2AgentEx", "venueCausal": "Continual Causality",
               "venueAgms": "Human-Centric"}


def run(cmd, cwd=None):
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return p.returncode, p.stdout + p.stderr


def live_lines(path: Path):
    """Source lines with comments removed (an escaped \\% stays)."""
    for line in path.read_text().splitlines():
        yield re.split(r"(?<!\\)%", line, maxsplit=1)[0]


def paragraphs(text: str) -> set:
    body = "\n".join(re.split(r"(?<!\\)%", ln, maxsplit=1)[0] for ln in text.splitlines())
    return {" ".join(p.split()) for p in re.split(r"\n\s*\n", body) if len(p.split()) >= MIN_WORDS}


def drift(venue: Path) -> list:
    common_names = {p.name for p in COMMON.rglob("*") if p.is_file()}
    hits = [f"stale copy {p.relative_to(venue)}" for p in venue.rglob("*") if p.is_file() and p.name in common_names
            and p.name not in TEMPLATE_FILES]
    shared = set().union(*(paragraphs(p.read_text()) for p in COMMON.glob("*.tex")))
    pasted = paragraphs((venue / "0_main.tex").read_text()) & shared
    typed = [m.group(1) for line in live_lines(venue / "0_main.tex")
             if not line.lstrip().startswith(("\\title", "/TemplateVersion"))
             for m in TYPED.finditer(re.sub(r"\\(?:cite|ref|label|input|bibliography)\{[^}]*\}|V-JEPA 2\.1", " ", line))
             if m.group(1) not in TYPED_OK]
    return (hits + [f"pasted paragraph: {p[:60]}..." for p in pasted]
            + [f"typed number {t} (use a common/numbers.tex macro)" for t in typed])


def venue_info(venue: Path) -> dict:
    v = (venue / "venue.tex").read_text()  # FileNotFoundError is the intended crash: an unfrozen draft needs its switches
    m = re.search(r"^% pages: (\d+)", v, re.M)
    if not m:
        raise RuntimeError(f"{venue.name}/venue.tex has no '% pages: N' line")
    switches = [s for s in VENUE_NAMES if f"\\{s}true" in v]
    if len(switches) != 1:
        raise RuntimeError(f"{venue.name}/venue.tex must set exactly one venue switch, found {switches}")
    return {"pages": int(m.group(1)), "anon": "\\anontrue" in v, "venue": VENUE_NAMES[switches[0]]}


def pdf_text(pdf: Path) -> str:
    _, t = run(["pdftotext", str(pdf), "-"])
    return re.sub(r"-\n", "", t)  # join words hyphenated across lines


def gates(d: Path, info: dict) -> list:
    """(gate, ok, detail) rows for one built draft."""
    R = []
    tex, pdf, log = d / "0_main.tex", d / "0_main.pdf", d / "0_main.log"
    L = log.read_text(errors="ignore")
    errs = len(re.findall(r"^! ", L, flags=re.M))
    undef = sorted(set(re.findall(r"(?:Citation|Reference) `([^']+)' on page \d+ undefined", L)))
    R.append(("build", errs == 0 and not undef, f"{errs} errors; undefined: {', '.join(undef) or 'none'}"))
    rc, inf = run(["pdfinfo", str(pdf)])
    pages = int(re.search(r"Pages:\s+(\d+)", inf).group(1))
    ref_page = None
    for p in range(1, pages + 1):
        _, pg = run(["pdftotext", "-f", str(p), "-l", str(p), str(pdf), "-"])
        lines = [ln.strip() for ln in pg.split("\n") if ln.strip()]
        if "References" in lines:
            ref_page = p - 1 if lines[0] == "References" else p
            break
    R.append(("pages", ref_page is not None and ref_page <= info["pages"],
              f"main text ends on page {ref_page} of {pages} (limit {info['pages']})"))
    text = pdf_text(pdf)
    # the AAAI-27 kit's links block prints "Dataset \u2014 <url>" itself; any other em dash is ours and fails
    prose_em = sum(ln.count("\u2014") for ln in text.splitlines() if not re.match(r"^\w[\w ]* \u2014 https?://", ln))
    R.append(("pdf text", "??" not in text and prose_em == 0,
              f"{text.count('??')} '??', {prose_em} em-dash glyphs outside the kit's links block"))
    if info["anon"]:
        srcs = [tex] + sorted(COMMON.glob("*.tex"))
        leaks = sorted({m.group(0) for s in srcs for ln in live_lines(s) for m in IDENTIFYING.finditer(ln)}
                       | {m.group(0) for m in IDENTIFYING.finditer(text)})
        author = re.search(r"\\author\{([^}]*)\}", tex.read_text())
        dated = [k for k in ("CreationDate", "ModDate", "Author") if re.search(rf"^{k}:", inf, re.M)]
        ok = not leaks and author is not None and "anonymous" in author.group(1).lower() and not dated
        R.append(("anonymity", ok, f"leaks: {', '.join(leaks) or 'none'}; author line: "
                  f"{author.group(1) if author else 'missing'}; PDF metadata: {', '.join(dated) or 'no dates or author'}"))
    uses = sum(ln.count("\\footnote{\\disclosure}") for ln in live_lines(tex))
    m = re.search(r"Concurrently under review, all non-archival:(.{0,260}?)An earlier version", text, re.S)
    flat = lambda s: re.sub(r"[\s-]", "", s)  # pdftotext drops or keeps a hyphen at a line break; compare without both
    named = [n for n in VENUE_NAMES.values() if m and flat(n) in flat(m.group(1))]
    ok = uses == 1 and m is not None and info["venue"] not in named and len(named) == 3
    R.append(("disclosure", ok, f"footnote used {uses}x; names {', '.join(named) or 'nothing'}"))
    dr = drift(d)
    R.append(("drift", not dr, "; ".join(dr) or "no stale copy, pasted paragraph or typed number"))
    rc, out = run([sys.executable, str(AI_SIG), str(tex)])
    R.append(("ai-signatures", rc == 0, out.strip().splitlines()[-1] if out.strip() else "no output"))
    return R


def title(d: Path) -> str:
    m = re.search(r"^\\title\{(.*)\}", (d / "0_main.tex").read_text(), re.M)
    return m.group(1) if m else ""


def main():
    drafts = sorted(d for d in HERE.glob("v*_*") if (d / "0_main.tex").exists())
    rows, failed, titles = [], 0, {}
    for d in drafts:
        if (d / "FROZEN").exists():
            pdf = d / "0_main.pdf"
            sha = hashlib.sha256(pdf.read_bytes()).hexdigest()[:12] if pdf.exists() else "no pdf"
            rows.append((d.name, "frozen", f"sha256 {sha}"))
            continue
        info = venue_info(d)
        titles.setdefault(title(d), []).append(d.name)
        print(f"== {d.name}: building ({info['venue']}, limit {info['pages']} pages)", flush=True)
        rc, out = run(["latexmk", "-pdf", "-interaction=nonstopmode", "-silent", "0_main.tex"], cwd=d)
        if rc != 0:
            print(out[-2000:], flush=True)
        R = gates(d, info)
        for g, ok, detail in R:
            print(f"   {'PASS' if ok else 'FAIL'}  {g:13s} {detail}", flush=True)
        bad = [g for g, ok, _ in R if not ok]
        failed += bool(bad)
        rows.append((d.name, "FAIL: " + ", ".join(bad) if bad else "PASS", R[1][2]))
    dup = {t: ds for t, ds in titles.items() if len(ds) > 1}
    for t, ds in dup.items():
        print(f"FAIL  title shared by {', '.join(ds)}: {t}")
    w = [max(len(str(r[i])) for r in rows + [("draft", "result", "detail")]) for i in range(3)]
    for r in [("draft", "result", "detail")] + rows:
        print(" | ".join(str(x).ljust(w[i]) for i, x in enumerate(r)))
    if failed or dup:
        sys.exit(f"{failed} unfrozen draft(s) failed{', titles repeat' if dup else ''}")


if __name__ == "__main__":
    main()

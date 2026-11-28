"""judge_pattern_check.py: fail a paper PDF on the tone pattern AI judges (Claude, ChatGPT, Gemini) repeat about abstracts, and list
every sentence a judge quoted as "Before" that is still word for word in the PDF.

Checks on the ABSTRACT (J1-J5) and the judges' own top asks (J6):
  J1 contribution early: a sentence among the first 3 presents the work (we present / introduce / propose / release, or one of the
     profile's work_names followed by a verb).
  J2 not a number list: no 3 sentences in a row with 3+ numbers each, and at most 10 numbers per 100 words.
  J3 no definitions: the abstract defines no term ("X is one model", "means that", "is defined as", "we call").
  J4 implication close: the last sentence says what follows (should / point(s) to / suggest / guide / motivate / call for / enable).
  J5 no negative close: the last sentence holds no confession or tie (not tested, unverifiable, cannot, no clear, not logged, tie).
  J6 judges' "Before" quotes: every quoted "Before" sentence and Part D table quote still in the PDF is a WARN (rewrite it, or a
     decision row with its reason in the draft's TODOs.md).
Usage: python3 judge_pattern_check.py PAPER.pdf [JUDGE_step2.md ...] [--profile P]    exit 1 on any J1-J5 FAIL.
       python3 judge_pattern_check.py --selftest
Project words live in .claude/skill-profiles/audit-judge-pattern.md (```json block, found by walking up from the PDF and the working
directory): work_names (the paper's method/dataset names), anon_banner (regex of a header line to drop before finding the abstract),
abstract_end (regex where the abstract stops, e.g. the first blank line, when the next column follows it in pdftotext order).
History: dataset-bench-lab audit-cvpr-reviewer/judge_pattern_check.py (Nov 27 2026), made project-independent Sat Nov 28 2026.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

SKILL = "audit-judge-pattern"
NUM = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?")
PRESENT = re.compile(r"\b(we (present|introduce|propose|release|contribute|build)|this paper (presents|introduces|proposes))\b", re.IGNORECASE)
DEFINE = re.compile(r"\b(is one model|means that|is defined as|we call|refers to)\b", re.IGNORECASE)
IMPLY = re.compile(r"\b(should|points? to|suggests?|guides?|motivates?|calls? for|enables?|opens? the way|so that)\b", re.IGNORECASE)
NEGCLOSE = re.compile(r"\b(not tested|unverifiable|unverified|cannot|no clear|not logged|tie|ties|inconclusive|remains? open)\b", re.IGNORECASE)
NAME_VERB = r"\b(is|tests|grades|measures|turns|trains|adapts|holds|learns|predicts)\b"
DEFAULT_NAMES = r"[A-Z][A-Za-z\-]+(Bench|Gym|PO|Net|JEPA)"


def find_profile(starts):
    for start in starts:
        for d in [start.resolve(), *start.resolve().parents]:
            p = d / ".claude" / "skill-profiles" / f"{SKILL}.md"
            if p.exists():
                return p
    return None


def load_profile(path):
    if path is None:
        return {}
    m = re.search(r"```json\s*\n(.*?)\n```", path.read_text(), re.DOTALL)
    if not m:
        sys.exit(f"FATAL: {path} has no ```json block")
    return json.loads(m.group(1))


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", t) if len(s.strip()) > 3]


def tex_anchor(pdf):
    """First 5 words of the abstract in the .tex next to the PDF, as a whitespace-tolerant regex (None without a .tex)."""
    tex = Path(pdf).with_suffix(".tex")
    if not tex.exists():
        return None
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex.read_text(), re.DOTALL)
    if not m:
        return None
    words = re.findall(r"[A-Za-z][A-Za-z,'-]*", re.sub(r"\\\w+(\{[^}]*\})?", " ", m.group(1)))[:5]
    return r"\s+".join(re.escape(w) for w in words) if len(words) == 5 else None


def abstract_of(text, banner=None, end=None, anchor=None):
    """end: the profile's abstract_end regex, for layouts where the next column's text follows the abstract in pdftotext order.
    anchor: the abstract's first words from the .tex, used when the PDF text order puts other text right after "Abstract"."""
    t = re.sub(banner, " ", text, flags=re.DOTALL) if banner else text
    k = re.search(anchor, t) if anchor else None
    if k:
        a = t[k.start():k.start() + 4000]
    else:
        m = re.search(r"\bAbstract\b(.*?)(\n\s*(1\s+)?Introduction\b|\n\s*\d?\s*[A-Z][a-z]+ [a-z]+[^\n]{0,40}\n\s*\n)", t, re.DOTALL)
        a = m.group(1) if m else t[:2500]
    if end:
        e = re.search(end, a.strip("\n"))
        a = a.strip("\n")[:e.start()] if e else a
    return re.sub(r"\s+", " ", a.replace("-\n", "")).strip()


def check_abstract(a, names=DEFAULT_NAMES):
    out, S = [], sentences(a)
    if not S:
        return ["J0 FAIL: no abstract found"]
    named = re.compile(r"^(?:" + names + r")\b.{0,40}" + NAME_VERB)
    if not any(PRESENT.search(s) or named.search(s) for s in S[:3]):
        out.append("J1 FAIL: no sentence among the first 3 presents the work (we present / introduce / propose)")
    counts = [len(NUM.findall(s)) for s in S]
    run = best = 0
    for c in counts:
        run = run + 1 if c >= 3 else 0
        best = max(best, run)
    words = len(a.split())
    nums = sum(counts)
    dens = 100 * nums / max(words, 1)
    if best >= 3:
        out.append(f"J2 FAIL: {best} sentences in a row carry 3+ numbers each (a number list, not findings)")
    if dens > 10:
        out.append(f"J2 FAIL: {nums} numbers in {words} words ({dens:.1f} per 100, limit 10)")
    for s in S:
        if DEFINE.search(s):
            out.append(f"J3 FAIL: the abstract defines a term: \"{s[:90]}\"")
            break
    if not IMPLY.search(S[-1]):
        out.append(f"J4 FAIL: the last sentence states no implication: \"{S[-1][:110]}\"")
    if NEGCLOSE.search(S[-1]):
        out.append(f"J5 FAIL: the abstract closes on a confession or tie: \"{S[-1][:110]}\"")
    return out


def judge_befores(md):
    t = Path(md).read_text(encoding="utf-8")
    outs = []
    for m in re.finditer(r"(?im)^\s*\**\s*before\s*\**\s*:?\s*\n?\s*[\"“](.{25,400}?)[\"”]", t):
        outs.append(m.group(1))
    # Part D tables: | # | where | "quote" |. Row ids may carry a letter or a dot (C1, B11, 4.1), not only digits.
    for m in re.finditer(r"\|\s*[A-Z]?\d+(?:\.\d+)?\s*\|[^|\n]*\|\s*[\"“](.{25,300}?)[\"”]\s*\|", t):
        outs.append(m.group(1))
    return outs


def norm(s):
    return re.sub(r"[^a-z0-9% ]", "", re.sub(r"\s+", " ", s.lower())).strip()


def main(pdf, mds, prof):
    text = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True, check=True).stdout
    names = "|".join([DEFAULT_NAMES] + [re.escape(n) for n in prof.get("work_names", [])])
    a = abstract_of(text, prof.get("anon_banner"), prof.get("abstract_end"), tex_anchor(pdf))
    fails = check_abstract(a, names)
    flat = norm(text.replace("-\n", ""))
    print(f"judge_pattern_check: {pdf}\n  abstract: {len(a.split())} words, {len(sentences(a))} sentences")
    for f in fails:
        print("  " + f)
    warn = 0
    for md in mds:
        for q in judge_befores(md):
            k = norm(q)[:70]
            if len(k) > 30 and k in flat:
                warn += 1
                print(f"  J6 WARN NOT ADDRESSED ({md.split('/')[-2]}): \"{q[:110]}\"")
    print(f"  {'FAIL' if fails else 'PASS'} J1-J5 ({len(fails)} finding(s)), J6 {warn} judge quote(s) still in the paper")
    return 1 if fails else 0


def selftest():
    report = ("Agents that share a street with people need models of crowded scenes. Most corpora sample this rarely. The corpus holds "
              "115,687 clips from 714 videos of 22 cities over 276 hours. One configuration wins on 3 of 4 diagnostics at 2B and 4 of 4 at 1B "
              "by 1.1 to 2.4 widths. The rivals lead on 6 of 13 coherence rows at 115k with 23,106 test clips. Coherence is not tested at 2B.")
    story = ("Agents that share a street with people need models of crowded scenes. We present a corpus of public street video and a "
             "curriculum that teaches a video world model the layout, the road users and their interactions in turn. The staged schedule "
             "carries most of the gain, and curated views make prediction degrade more slowly as more of the scene is hidden. This suggests "
             "that structuring public video, not labelling it, is the cheap lever for world models of shared spaces.")
    tie = "We present a method. It reaches 0.56 on A. On B the methods tie and the gap is not tested."
    cases = [("number-list abstract with a confession close", report, True), ("story-order abstract", story, False),
             ("close on a tie", tie, True)]
    ok = 0
    for name, a, want_fail in cases:
        f = check_abstract(a)
        got = bool(f)
        ok += got == want_fail
        print(f"{'ok ' if got == want_fail else 'BAD'} {name}: {'FAIL' if got else 'PASS'} {f[:3]}")
    print(f"selftest {ok} of {len(cases)}")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--selftest"]:
        sys.exit(selftest())
    ppath = None
    if "--profile" in argv:
        i = argv.index("--profile")
        ppath = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if not argv:
        sys.exit(__doc__)
    ppath = ppath or find_profile([Path(argv[0]).parent, Path.cwd()])
    print(f"profile: {ppath if ppath else 'none (generic name pattern)'}")
    sys.exit(main(argv[0], argv[1:], load_profile(ppath)))

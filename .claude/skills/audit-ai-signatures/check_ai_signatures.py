#!/usr/bin/env python3
"""Scan a LaTeX paper (main file + every \\input) for prose that reads as machine-written.

Only text that reaches the PDF is scanned: comment lines and inline %-comments are dropped.
Exit code 1 if any FAIL check fires.

  emdash      '---' or a Unicode em dash in prose                         FAIL
  phrases     stock LLM vocabulary (delve, tapestry, pivotal, seamless,    WARN, listed with file:line
              underscore, showcase, multifaceted, it is worth noting ...)
  connectors  Moreover / Furthermore / Additionally / Notably openers      WARN above 4 per 1k words
  emphasis    \\textbf + \\emph + \\textit commands per 1k words            WARN above 25 per 1k words
  quotes      curly and straight apostrophes mixed in one paper           WARN

usage: python3 check_ai_signatures.py <main.tex> [--max-hits 8] [--allow "phrase" ...]
"""
import argparse
import os
import re
import sys

INPUT_RE = re.compile(r'\\(?:input|include)\{([^}]+)\}')
PHRASES = [
    r'\bdelv(?:e|es|ing)\b', r'\btapestr(?:y|ies)\b', r'\bpivotal\b', r'\bseamless(?:ly)?\b',
    r'\bunderscor(?:e|es|ing)\b', r'\bshowcas(?:e|es|ing)\b', r'\bmultifaceted\b', r'\bholistic\b',
    r'\bmeticulous(?:ly)?\b', r'\bintricate\b', r'\bin the realm of\b', r'\brealm\b', r'\bever[- ]evolving\b',
    r'\bgame[- ]chang(?:er|ing)\b', r'\bparadigm shift\b', r'\bgroundbreaking\b', r'\brevolutioni[sz]e\b',
    r'\bcutting[- ]edge\b', r'\btestament to\b', r'\bembark(?:s|ing)?\b', r'\bnavigat(?:e|ing) the (?:complexities|landscape)\b',
    r'\bunlock(?:s|ing)? the\b', r'\bharness(?:es|ing)? the\b', r'\bit is worth noting\b', r"\bit's worth noting\b",
    r'\bin conclusion\b', r'\bin summary\b', r'\bat its core\b', r'\bplays? a (?:crucial|pivotal|key|vital) role\b',
    r'\bcrucial(?:ly)?\b', r'\bvital\b', r'\bfoster(?:s|ing)?\b', r'\belevat(?:e|es|ing)\b', r'\bnuanced\b',
    r'\bdive (?:deep )?into\b', r'\bstands? as\b', r'\bserves? as a testament\b', r'\bnot only\b[^.]{0,80}\bbut also\b',
    r'\bin today\'s\b', r'\blandscape\b', r'\bensur(?:e|es|ing) that\b', r'\bremarkabl(?:e|y)\b',
]
CONNECTORS = re.compile(r'(?:^|[.!?]\s+)(Moreover|Furthermore|Additionally|Notably|Importantly|Crucially),')
EMPH = re.compile(r'\\(?:textbf|emph|textit)\{')


def files_of(main):
    order, seen = [], set()

    def visit(p):
        rp = os.path.realpath(p)
        if rp in seen:
            return
        seen.add(rp)
        order.append(p)
        with open(p, encoding='utf-8') as fh:
            for line in fh:
                code = re.sub(r'(?<!\\)%.*', '', line)
                for m in INPUT_RE.finditer(code):
                    n = m.group(1)
                    c = os.path.join(os.path.dirname(p), n if n.endswith('.tex') else n + '.tex')
                    if not os.path.exists(c):
                        raise RuntimeError(f'{p}: \\input target not found: {c}')
                    visit(c)
    visit(main)
    return order


def prose_lines(path):
    with open(path, encoding='utf-8') as fh:
        for i, line in enumerate(fh, 1):
            code = re.sub(r'(?<!\\)%.*', '', line).rstrip('\n')
            if code.strip():
                yield i, code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('main')
    ap.add_argument('--max-hits', type=int, default=8, help='locations listed per row')
    ap.add_argument('--allow', action='append', default=[], help='phrase regex to skip (e.g. a defined term)')
    a = ap.parse_args()

    base = os.path.dirname(os.path.abspath(a.main))
    allow = [re.compile(x, re.I) for x in a.allow]
    words = 0
    hits = {'emdash': [], 'connectors': [], 'quotes_curly': [], 'quotes_straight': 0, 'emphasis': 0}
    phrase_hits = {}
    for f in files_of(a.main):
        rel = os.path.relpath(f, base)
        for i, line in prose_lines(f):
            text = re.sub(r'\\[a-zA-Z]+\*?', ' ', line)
            words += len(re.findall(r'[A-Za-z]{2,}', text))
            for m in re.finditer(r'---|\u2014', line):
                hits['emdash'].append(f'{rel}:{i}')
            for m in CONNECTORS.finditer(text):
                hits['connectors'].append(f'{rel}:{i} {m.group(1)}')
            hits['emphasis'] += len(EMPH.findall(line))
            if '\u2019' in line or '\u201c' in line or '\u201d' in line:
                hits['quotes_curly'].append(f'{rel}:{i}')
            hits['quotes_straight'] += len(re.findall(r"[A-Za-z]'[A-Za-z]", line))
            for pat in PHRASES:
                for m in re.finditer(pat, text, flags=re.I):
                    w = m.group(0)
                    if any(al.search(w) for al in allow):
                        continue
                    phrase_hits.setdefault(w.lower(), []).append(f'{rel}:{i}')

    k = max(words, 1) / 1000
    rows, failed = [], False

    def add(check, status, count, detail):
        nonlocal failed
        failed |= status == 'FAIL'
        rows.append((check, status, str(count), detail))

    add('emdash', 'FAIL' if hits['emdash'] else 'PASS', len(hits['emdash']), ', '.join(hits['emdash'][:a.max_hits]) or '-')
    for w, locs in sorted(phrase_hits.items(), key=lambda kv: -len(kv[1])):
        add(f'phrase: {w}', 'WARN', len(locs), ', '.join(locs[:a.max_hits]))
    if not phrase_hits:
        add('phrases', 'PASS', 0, '-')
    nc = len(hits['connectors'])
    add('connectors', 'WARN' if nc / k > 4 else 'PASS', nc, f'{nc / k:.1f} per 1k words; ' + ', '.join(hits['connectors'][:a.max_hits]))
    ne = hits['emphasis']
    add('emphasis', 'WARN' if ne / k > 25 else 'PASS', ne, f'{ne / k:.1f} \\textbf/\\emph/\\textit per 1k words')
    nq = len(hits['quotes_curly'])
    mixed = nq and hits['quotes_straight']
    add('quotes', 'WARN' if mixed else 'PASS', nq, f'{nq} lines with curly quotes vs {hits["quotes_straight"]} straight apostrophes; ' + ', '.join(hits['quotes_curly'][:a.max_hits]))

    w = [max(len(r[i]) for r in rows + [('check', 'status', 'count', 'where')]) for i in range(4)]
    w[0] = min(w[0], 32)
    w[3] = min(w[3], 96)
    line = lambda l, m, r: l + m.join('\u2500' * (x + 2) for x in w) + r
    print(f'{a.main}: {words} words of prose')
    print(line('\u250c', '\u252c', '\u2510'))
    print('\u2502' + '\u2502'.join(f' {h:<{w[i]}} ' for i, h in enumerate(('check', 'status', 'count', 'where'))) + '\u2502')
    print(line('\u251c', '\u253c', '\u2524'))
    for r in rows:
        print('\u2502' + '\u2502'.join(f' {str(c)[:w[i]]:<{w[i]}} ' for i, c in enumerate(r)) + '\u2502')
    print(line('\u2514', '\u2534', '\u2518'))
    print('RESULT:', 'FAIL' if failed else 'PASS')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

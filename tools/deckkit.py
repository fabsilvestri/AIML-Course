"""Slide-level surgery on a reveal.js deck.

A deck is a preamble, a list of top-level <section> slides, and a postamble.
Rebuilding a lecture is mostly selecting, reordering and rewriting slides —
which is a list operation, not a text-editing one.
"""
import re, sys

SECTION = re.compile(r'^<section\b', re.M)


def split(path):
    """-> (preamble, [slide_html, ...], postamble)"""
    src = open(path, encoding='utf-8').read()
    head, _, rest = src.partition('<div class="slides">')
    body, _, tail = rest.rpartition('</div>\n</div>')
    slides, depth, buf = [], 0, []
    for line in body.splitlines(keepends=True):
        stripped = line.lstrip()
        if depth == 0 and stripped.startswith('<section'):
            depth = 1
            buf = [line]
            if '</section>' in line:
                slides.append(''.join(buf)); depth = 0
            continue
        if depth:
            buf.append(line)
            depth += stripped.count('<section') - stripped.count('</section>')
            if depth <= 0:
                slides.append(''.join(buf)); depth = 0
    return head + '<div class="slides">\n', slides, '</div>\n</div>' + tail


def title(slide):
    m = re.search(r'data-menu-title="([^"]*)"', slide)
    if m:
        return m.group(1)
    m = re.search(r'<h[123][^>]*>(.*?)</h[123]>', slide, re.S)
    return re.sub(r'<[^>]+>', '', m.group(1)).strip() if m else '(untitled)'


def join(pre, slides, post):
    out = [pre]
    for i, s in enumerate(slides, 1):
        out.append(f'\n<!-- {i} -->\n' if not s.startswith('\n') else '')
        out.append(s if s.endswith('\n') else s + '\n')
    out.append(post)
    return ''.join(out)


if __name__ == '__main__':
    pre, slides, post = split(sys.argv[1])
    for i, s in enumerate(slides):
        cls = re.search(r'<section class="([^"]*)"', s)
        print(f'{i:3d}  {(cls.group(1) if cls else ""):14s}  {title(s)}')

# Fonts

Vendored, not loaded from a CDN — the decks must render in a lecture hall with no
network. All Open Font License.

| file | used by | why |
|---|---|---|
| `SourceSans3-VF.woff2` | the browser | variable, weight axis 200–900 |
| `SourceSans3-Italic-VF.woff2` | the browser | italic companion |
| `SourceCodePro-VF.woff2` | the browser | code blocks |
| `SourceSans3-{Regular,SemiBold,Bold}.ttf` | matplotlib | static instances |
| `SourceSans3-VF.ttf`, `SourceCodePro-VF.ttf` | — | sources for the instances |

## Why both woff2 and ttf

The reveal `white` theme vendors only **400 and 600** of Source Sans Pro, so
every `font-weight: 700` in the slide theme silently fell back to semibold —
headings and `<strong>` rendered at the same weight, and weight carried no
hierarchy at all. Source Sans 3 is a metric-compatible superset with a real
weight axis, so the theme now has the three weights it always claimed.

matplotlib cannot read woff2, and a *variable* TTF resolves to its default
instance — which for Source Sans 3 is ExtraLight. Hence the three static cuts,
renamed so `font_manager` sees one family with three weights.

## Regenerating the static instances

```python
from fontTools.varLib import instancer
from fontTools.ttLib import TTFont
f = instancer.instantiateVariableFont(TTFont("SourceSans3-VF.ttf"), {"wght": 600})
# then rewrite name IDs 1, 2, 4, 6 to "Source Sans 3" / "SemiBold" / …
```

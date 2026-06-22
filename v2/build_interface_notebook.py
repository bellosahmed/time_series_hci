"""Build v2/notebooks/08_interface.ipynb — documented, imports src/interface.py + src/llm.py."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# v2 — Notebook 08 · Natural-language interface + local LLM

**The user-facing layer.** Ask a question in plain English; the system detects what you mean, the
**query engine** computes the exact number, and the answer is returned **trust-aware**. If a local
LLM (Ollama) is running it may *phrase* the answer more naturally — but it can only rephrase the
computed facts, never change a number.

```
question → detect intent + country/flow/dates → query_engine (exact) → answer
                                                        │
                                                 local LLM (optional) only PHRASES it
```

Trust labels: 🔢 computed (the truth) · 🧠 inferred (labelled context) · 💬 LLM phrasing.
Logic in `src/interface.py` and `src/llm.py`.
> ⚠️ v2 = international trade.""")

md("## Step 0 · Setup")
code(r"""import sys
from pathlib import Path
def find_v2_root():
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "src" / "interface.py").exists():
            return base
    raise FileNotFoundError("v2 root not found")
V2 = find_v2_root(); sys.path.insert(0, str(V2))
DATA = V2.parent / "data" / "eu_trade_monthly.csv"

from src import preprocess as pp
from src import interface as ui
from src import llm

df = pp.load_trade(DATA)
print("loaded:", df.shape)
print("Ollama available:", llm.is_available(), "| models:", llm.list_models())""")

md(r"""## Step 1 · Ask questions in plain English

**What:** run a batch of natural questions through the interface. **How to read:** each 🔢 line is
the exact, code-computed fact — phrased in words but with the engine's numbers.""")
code(r"""questions = [
    "What was France's highest export month?",
    "What was Germany's lowest import month?",
    "Are Germany and France exports correlated?",
    "How much did Poland's exports grow from 2010 to 2024?",
    "What was Spain's worst monthly export drop?",
    "Who had the biggest trade surplus in 2024?",
    "Top exporters in 2024?",
    "Summarise the Netherlands.",
]
for q in questions:
    print("Q:", q)
    print(ui.ask(df, q))
    print("-" * 70)""")

md(r"""## Step 2 · Robustness — synonyms & free phrasing (not just keywords)

**What:** the same question asked different ways. **Why:** the interface keys off intent synonyms,
so it isn't brittle to exact wording. **How to read:** all variants resolve to the same computed
fact.""")
code(r"""for q in ["When did Italy export the most?",
          "Italy's peak exports?",
          "What is the largest IT export month?"]:
    print("Q:", q, "->", ui.ask(df, q).computed)""")

md(r"""## Step 3 · Multi-country reasoning through the interface

**What:** a "shock" question routes to the reasoning layer and returns computed facts **plus** a
clearly-labelled inference. **How to read:** 🔢 is measured; 🧠 is interpretation, never asserted as
fact.""")
code(r"""print(ui.ask(df, "Which countries were abnormal in 2020-04?"))
print("-" * 70)
print(ui.ask(df, "Which countries had a shock in 2009-01?"))""")

md(r"""## Step 4 · The local LLM phrasing (if Ollama is running)

**What:** with `use_llm=True`, the local model rephrases the computed facts. **Why:** nicer prose,
same guarantees — the prompt forbids changing numbers, and if Ollama is down it silently falls back
to the template. **How to read:** 💬 = the LLM's wording of the 🔢 facts.""")
code(r"""q = "How much did Poland's exports grow from 2010 to 2024?"
a = ui.ask(df, q, use_llm=True)
print("computed (truth):", a.computed)
print("LLM phrasing    :", a.phrased if a.phrased else "(Ollama not running — template used)")""")

md(r"""## Step 5 · Graceful errors & suggestions

**What:** the interface asks for missing details instead of guessing, and ships a list of example
questions for a UI. **Why:** never fabricate an answer — a core trust requirement.""")
code(r"""print(ui.ask(df, "What is the highest?"))          # no country -> asks which
print(ui.ask(df, "tell me a joke"))                 # out of scope -> lists capabilities
print("\nSuggested questions:")
for s in ui.SUGGESTIONS: print("  -", s)""")

md(r"""## Step 6 · Summary

| Capability | Routed to | Trust |
| --- | --- | --- |
| highest / lowest / average | `query_engine` | 🔢 exact |
| correlation, growth, worst drop | `query_engine` | 🔢 exact |
| surplus / deficit / balance / ranking | `query_engine` | 🔢 exact |
| country summary | `query_engine` | 🔢 exact |
| shock / which countries | `reasoning` | 🔢 + 🧠 labelled |
| natural phrasing | `llm` (optional) | 💬 rephrase only |

The interface **never computes** — it routes to the truth layer and labels every part. The LLM is a
phrasing convenience, not a source of facts. That is the project's trust principle, end to end.

**→ Next: Notebook 09 · Evaluation & comparison** — bring together detector metrics, the v1↔v2
comparison, and the final results tables; then the web app + write-up.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent / "notebooks" / "08_interface.ipynb"
nbf.write(nb, out)
print("Wrote", out, "with", len(cells), "cells")

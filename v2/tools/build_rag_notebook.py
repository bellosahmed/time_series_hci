"""Build v2/notebooks/09_rag.ipynb — documented, imports src/rag_system.py."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# v2 — Notebook 09 · Hybrid RAG (lexical BM25 + semantic LSA)

**What RAG adds.** The query engine gives exact *numbers*; RAG retrieves *context* — country
descriptions, what the 2008/COVID/2022 events were, what a method means. Combined, an answer can say
both **what happened** (🔢 computed) and **why it might matter** (📚 retrieved · 🧠 inferred).

**Hybrid retrieval** = two retrievers, merged:
- **BM25** (lexical) — great at *exact terms* ("surplus", "Germany").
- **LSA** (semantic) — matches *meaning*, so "pandemic crash" finds the COVID doc even without the
  word "COVID".
- **Hybrid** — normalise both scores and combine, getting the strengths of each.

**Trust rule:** RAG supplies *context only*, never numbers. The KB contains no euro amounts; the
truth layer stays the only source of figures.
> Note: LSA is used for the semantic half because the installed Ollama models can't embed; the
> `search()` interface is unchanged if we later pull a neural embed model.""")

md("## Step 0 · Setup — build the retriever")
code(r"""import sys
from pathlib import Path
def find_v2_root():
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "src" / "rag_system.py").exists():
            return base
    raise FileNotFoundError("v2 root not found")
V2 = find_v2_root(); sys.path.insert(0, str(V2))
DATA = V2.parent / "data" / "eu_trade_monthly.csv"

from src import preprocess as pp
from src import rag_system as rag
from src import interface as ui
from src import llm

kb = rag.knowledge_base()
hybrid = rag.HybridRetriever().fit(kb)
print(f"KB documents: {len(kb)}")
print("doc ids:", [i for i, _ in kb])""")

md(r"""## Step 1 · BM25 (lexical) — matches exact terms

**How to read:** scores are highest for docs literally containing the query words.""")
code(r"""bm = rag.BM25().fit(kb)
for q in ["trade surplus", "isolation forest anomaly"]:
    print("Q:", q)
    for i, s in bm.search(q, k=3):
        print(f"   {s:5.2f}  {i}")
    print()""")

md(r"""## Step 2 · LSA (semantic) — matches meaning, not just words

**What:** query with words that *don't appear* in the target doc. **Why:** semantic retrieval finds
it anyway. **How to read:** "pandemic lockdown" surfaces the COVID doc though it says "COVID-19".""")
code(r"""sem = rag.SemanticLSA().fit(kb)
for q in ["pandemic lockdown halted commerce", "war disrupted energy and trade"]:
    print("Q:", q)
    for i, s in sem.search(q, k=3):
        print(f"   {s:5.2f}  {i}")
    print()""")

md(r"""## Step 3 · Hybrid — lexical + semantic, side by side

**What:** compare BM25-only, semantic-only, and hybrid on the same query. **How to read:** hybrid
blends both rankings, so it is robust whether you use exact terms or paraphrase.""")
code(r"""q = "which country sells the most abroad"
print("Q:", q, "\n")
print("BM25     :", [i for i, _ in bm.search(q, k=3)])
print("Semantic :", [i for i, _ in sem.search(q, k=3)])
print("Hybrid   :", [i for i, _ in hybrid.search(q, k=3)])""")

md(r"""## Step 4 · The full trust-aware answer (the whole stack)

**What:** combine everything — the **query engine** computes the number, **RAG** adds context, the
**LLM** (if up) phrases it. **How to read:** 🔢 = computed truth · 📚 = retrieved context ·
🧠 = labelled inference · 💬 = LLM wording. The number only ever comes from 🔢.""")
code(r"""# 1) exact fact from the query engine (the truth layer, via the interface)
df = pp.load_trade(DATA)
ans = ui.ask(df, "What was Spain's worst monthly export drop?")
# 2) context from RAG
ctx = hybrid.retrieve_texts(ans.computed + " export crash crisis", k=2)
# 3) assemble trust-aware answer
print(rag.explain(ans.computed, retrieved=ctx,
                  inferred="The timing matches a major global shock."))
# 4) optional LLM phrasing of the SAME number
phrased = llm.explain("Spain worst export drop", ans.computed,
                      inferred="timing matches a global shock")
print("\n💬", phrased if phrased else "(Ollama off — template/computed used)")""")

md(r"""## Step 5 · Summary

| Component | Role | Trust |
| --- | --- | --- |
| BM25 | lexical retrieval (exact terms) | 📚 context |
| LSA | semantic retrieval (meaning) | 📚 context |
| Hybrid | normalise + combine + re-rank | 📚 context |
| Query engine | the only source of numbers | 🔢 computed |
| LLM | rephrase computed facts | 💬 wording |

RAG enriches answers with *why/what* context while the **numbers stay exact and code-computed**.
Hybrid retrieval is robust to both keyword and paraphrased questions. Neural embeddings can replace
LSA later behind the same interface (pull `nomic-embed-text` in Ollama).

**→ Next: Notebook 10 · Evaluation & v1↔v2 comparison**, then the web app + write-up.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "09_rag.ipynb"
nbf.write(nb, out)
print("Wrote", out, "with", len(cells), "cells")

"""Build v2/notebooks/13_chunking_bert.ipynb — chunking + neural (BERT) retrieval."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# v2 — Notebook 13 · Chunking + BERT retrieval (modern RAG)

Two upgrades to the retrieval layer:
- **Chunking** — split *long* documents into overlapping, sentence-aware pieces so a query matches
  the relevant part. (Our KB is already atomic; this readies the system for long sources.)
- **BERT** — dense neural embeddings (`sentence-transformers`, a distilled BERT) that capture
  *meaning*, not just words. Drop-in replacement for LSA behind the same `search()` interface.

**Honest framing:** for our 20 tiny, atomic docs, BM25+LSA already retrieve correctly. Chunking and
BERT *pay off when the corpus is large and documents are long* — demonstrated here so the system
scales.
> ⚠️ v2 = international trade. Retrieval returns CONTEXT only; numbers stay with the query engine.""")

md("## Step 0 · Setup")
code(r"""import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
def find_v2_root():
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "src" / "rag_system.py").exists():
            return base
    raise FileNotFoundError("v2 root not found")
V2 = find_v2_root(); sys.path.insert(0, str(V2))
from src import rag_system as rag
kb = rag.knowledge_base()
print("KB docs:", len(kb))""")

md(r"""## Step 1 · Chunking a long document

**What:** split a long passage into overlapping ~40-word chunks. **Why:** retrieval works best on
focused pieces; overlap stops meaning being cut at a boundary. **How to read:** one long doc → several
retrievable chunks.""")
code(r"""long_doc = (
 "The 2008 global financial crisis began in the United States banking sector and spread worldwide. "
 "International trade collapsed sharply in late 2008 as credit froze and demand fell. "
 "Exports from major economies dropped by roughly a quarter within months. "
 "Governments responded with stimulus packages and interest-rate cuts. "
 "Trade began a slow recovery through 2009 and 2010, though unevenly across countries. "
 "The episode exposed how interconnected and fragile global supply chains had become."
)
chunks = rag.chunk_text(long_doc, max_words=30, overlap=8)
print(f"{len(chunks)} chunks:\n")
for i, c in enumerate(chunks):
    print(f"[{i}] ({len(c.split())} words) {c}\n")""")

md(r"""## Step 2 · BERT vs LSA vs BM25 — paraphrase query

**What:** a query that shares *no words* with the target doc. **Why:** to show neural embeddings
match meaning. **How to read:** BERT should rank the COVID doc top even though the query says
"factories shut and shipping stopped".""")
code(r"""q = "factories shut and shipping stopped worldwide"
bm = rag.BM25().fit(kb)
lsa = rag.SemanticLSA().fit(kb)
bert = rag.BertRetriever().fit(kb)
print("Query:", q, "\n")
print("BM25 :", [i for i, _ in bm.search(q, 3)])
print("LSA  :", [i for i, _ in lsa.search(q, 3)])
print("BERT :", [i for i, _ in bert.search(q, 3)])""")

md(r"""## Step 3 · Chunk → embed → retrieve (the full modern flow)

**What:** index the chunks from Step 1 with BERT, then ask a question and retrieve the best chunk.
**How to read:** the query "how did governments react?" should surface the stimulus chunk.""")
code(r"""chunk_docs = [(f"chunk_{i}", c) for i, c in enumerate(chunks)]
chunk_index = rag.BertRetriever().fit(chunk_docs)
for query in ["how did governments react to the crisis?",
              "what happened to exports?"]:
    cid, score = chunk_index.search(query, k=1)[0]
    print(f"Q: {query}\n   -> [{score:.2f}] {dict(chunk_docs)[cid]}\n")""")

md(r"""## Step 4 · Hybrid with the BERT backend

**What:** the same hybrid retriever, now using BERT for the semantic half. **How to read:** BM25
(exact) + BERT (meaning), combined and re-ranked.""")
code(r"""hybrid_bert = rag.HybridRetriever(semantic="bert").fit(kb)
for q in ["which country exports the most", "trade war and sanctions"]:
    print("Q:", q, "->", [i for i, _ in hybrid_bert.search(q, 3)])""")

md(r"""## Step 5 · Summary & when to use what

| Retriever | Captures | Cost | Best when |
| --- | --- | --- | --- |
| BM25 | exact terms | none | keyword queries, small KB |
| LSA | co-occurrence | sklearn | offline semantic, small KB |
| **BERT** | true meaning | model download | **large/varied corpora, paraphrase** |
| **Chunking** | focused pieces | none | **long documents** |

For our 20 atomic docs the three retrievers agree; **BERT + chunking earn their keep as the corpus
grows** (reports, articles). The architecture supports all of them behind one `search()` interface,
and retrieval still returns *context only* — the query engine remains the sole source of numbers.

**This completes the modern RAG stack (lexical + semantic + neural + chunking).**""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent / "notebooks" / "13_chunking_bert.ipynb"
nbf.write(nb, out)
print("Wrote", out, "with", len(cells), "cells")

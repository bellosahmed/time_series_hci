# 4. Retrieval — Information Retrieval metrics on a hand-labelled query set

## What's being evaluated

The RAG **retriever** in `src/rag_system.py` — the component that decides which sensor
description gets surfaced as "📚 Context" alongside a computed answer. Note what this
*isn't* evaluating: the truth layer's numbers are already proven correct
([`01-truth-layer.md`](01-truth-layer.md)); this page is about whether the *right explanatory
text* gets found for a given question, which is a search/ranking problem, not a computation one.

## Method

Because this is fundamentally a search-engine problem — "given a query, is the right document
ranked near the top?" — it's evaluated with the field's own standard metrics rather than an
invented one:

- **Precision@k** — of the top-k results, what fraction are actually relevant?
- **Recall@k** — of all relevant documents, what fraction appear in the top-k?
- **MRR** (Mean Reciprocal Rank) — how high up the ranking is the *first* correct result?
- **MAP** (Mean Average Precision) — precision averaged across every relevant result's position.
- **NDCG** (Normalized Discounted Cumulative Gain) — rewards relevant results ranked *higher*
  more than the same results ranked lower.

Ground truth (`retrieval_eval.GROUND_TRUTH`) is a hand-labelled set of
`(query, {which sensor doc(s) are actually relevant})` pairs — defined by human judgement before
ever running a retriever against it, e.g. `("atmospheric pressure in millibars", {"p (mbar)"})`.

**A harder, held-out generalisation test.** Because the sensor-description text was itself
enriched with colloquial synonyms after an earlier evaluation pass found gaps (the retrieval
system's own history — see `docs/RETRIEVAL_EVALUATION.md`), a fair test needs queries the
enrichment *couldn't* have been tuned to. `retrieval_eval` includes a held-out paraphrase set
whose every descriptive word was verified **absent** from the knowledge base, specifically to
test whether a retriever generalises or has just memorised the exact vocabulary it was tuned on.

## Real evidence (reproduced 15 July 2026, k=3, 10-query standard set)

```bash
python3.12 -c "
from src import retrieval_eval, rag_system
gt = retrieval_eval.GROUND_TRUTH
print('TF-IDF', retrieval_eval.evaluate_retriever(rag_system.Retriever(), gt, k=3))
print('LSA   ', retrieval_eval.evaluate_retriever(rag_system.LSARetriever(), gt, k=3))
"
```

| Retriever | P@3 | R@3 | MRR | MAP | NDCG@3 |
| --- | --- | --- | --- | --- | --- |
| TF-IDF (VSM baseline) | 0.333 | 1.000 | 0.950 | 0.950 | 0.963 |
| LSA (TF-IDF → truncated SVD) | 0.333 | 1.000 | 0.950 | 0.950 | 0.963 |

(P@3 = 0.333 is expected and correct, not a weakness: each query in this ground truth has exactly
1 relevant document, so the best any retriever can score at k=3 is 1 relevant out of 3 slots
returned — R@3 = 1.000 shows every single relevant document was actually found within the top 3.)

`verify_results.py` guards both retrievers with `MRR >= 0.8` — both clear that bar comfortably at
0.95.

## Honest caveats (a real, previously-documented finding, still true)

`docs/RETRIEVAL_EVALUATION.md` documents a genuine before/after finding from when this system was
first evaluated: with the *original* one-line sensor descriptions, keyword-based retrievers
(TF-IDF, BM25, LSA) tied a dense sentence-transformer encoder on standard queries but fell well
behind it on paraphrased queries (e.g. "how muggy does it get" instead of "relative humidity") —
MRR 0.267 vs. 0.500. Enriching the descriptions with everyday synonyms closed that gap for
everyone (paraphrase MRR rose to 1.000 for the keyword methods). But on a genuinely held-out set
— words verified absent from even the enriched descriptions — the keyword methods collapse back
to MRR 0.250 (they have no word to match at all) while the dense encoder still generalises to
0.556. **The honest conclusion, not softened**: document quality dominates on words the KB has
seen; a dense retriever is what still helps on words it's never seen. Neither retriever is simply
"the best one" — which one wins depends entirely on how far outside the KB's vocabulary a real
user's phrasing falls.

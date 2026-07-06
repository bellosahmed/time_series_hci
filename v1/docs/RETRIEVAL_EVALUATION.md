# Retrieval evaluation + LSA retriever — results & process

Built **2026-07-05** from the COP509 Information-Retrieval material (Evaluation Measures + LSI +
RAG lectures). This is the "review-later" record of **what was done, why, and what the numbers were**.

**State after this work (2026-07-05):** 169 tests pass (1 skipped) · `verify_results.py` **29/29** ·
ruff clean · mypy clean · coverage 92%. *(Current gates as of 2026-07-06: **182 tests** / **29/29**.)* Retrievers compared: **TF-IDF, BM25, LSA, and a dense sentence-transformer**;
the sensor knowledge base was then **enriched** and the effect re-measured (see the enrichment section).

---

## Why (the gap this closes)
The project already evaluates **anomaly detection** rigorously (precision/recall/F1, confusion,
bootstrap CIs, McNemar) and verifies the **truth layer** `== pandas`. But the **RAG retriever** — the
component that grounds each explanation in sensor metadata — was never *quantitatively* evaluated; it
just returned documents. The module's **Evaluation Measures** lecture provides exactly the tools to
fix that, and the **LSI** lecture provides a second retriever to compare against the TF-IDF baseline
("compare against a baseline and discuss the difference" — coursework guidance).

## What was built
1. **`src/retrieval_eval.py`** — the IR metrics **Precision@k, Recall@k, MRR, MAP, NDCG@k** (binary
   relevance) plus `evaluate_retriever(retriever, ground_truth, k)`, and a small hand-labelled query
   set `GROUND_TRUTH` (10 queries → the sensor doc each should retrieve). Relevance is defined
   externally by us (the assessor), never by the system — the standard IR principle.
2. **Four retrievers** (all in `rag_system.py`, all sharing the `Retriever` interface):
   - **`Retriever`** — TF-IDF + cosine (the VSM baseline; pre-existing).
   - **`BM25Retriever`** — Okapi BM25 (`rank_bm25`): the probabilistic baseline (TF saturation +
     length normalisation).
   - **`LSARetriever`** — LSI/LSA: TF-IDF → truncated SVD (13 latent dims) → cosine in topic space
     (scikit-learn only, deterministic).
   - **`dense_retriever()`** — a dense **sentence-transformer** (`all-MiniLM-L6-v2`, the module's
     recommended encoder) via `EmbeddingRetriever`, with an automatic TF-IDF fallback when the model
     is unavailable.
3. **Two `verify_results.py` checks** (part of the 29/29) that recompute the metrics and guard the TF-IDF and
   LSA baselines (dense/BM25 excluded to keep the trust gate dependency-light).
4. **Tests** — `tests/test_retrieval_eval.py` (metrics vs hand-computed values + end-to-end +
   paraphrase gap) and retriever cases in `tests/test_rag.py` (BM25, LSA, dense — the last two skip
   gracefully if the optional package/model is absent).

## Process (how the ground truth was made)
Following the taught recipe: (1) write 10 test queries covering every sensor family; (2) manually
judge the single relevant sensor doc for each (binary relevance); (3) run each retriever; (4) compare
ranked output to the judgements; (5) report the metrics in a table. Annotation criterion: *the query
names (a synonym of) exactly this sensor*.

## Results — four retrievers, two query sets

**Standard labelled queries (n = 10):** each names its sensor using knowledge-base vocabulary.

| retriever | P@1 | P@3 | R@3 | MRR | MAP | NDCG@3 |
|---|---|---|---|---|---|---|
| **TF-IDF** (VSM baseline) | 0.900 | 0.300 | 0.900 | 0.900 | 0.900 | 0.900 |
| **BM25** (Okapi) | 0.900 | 0.300 | 0.900 | 0.900 | 0.900 | 0.900 |
| **LSA** (TF-IDF → SVD, 13 dims) | 0.900 | 0.300 | 0.900 | 0.900 | 0.900 | 0.900 |
| **Dense** (sentence-transformers `all-MiniLM-L6-v2`) | 1.000 | 0.333 | 1.000 | **0.933** | **0.933** | **0.956** |

*(P@3 ≈ 0.30 is the ceiling — one relevant doc per query, so at k=3 the best possible is 1/3. R@3
and MRR are the informative columns.)*

**Paraphrase / out-of-vocabulary queries (n = 5):** each names its sensor **without any KB word**
("how hot", "muggy", "breezy") — isolating the semantic gap.

| retriever | R@3 | MRR | MAP |
|---|---|---|---|
| TF-IDF | 0.400 | 0.267 | 0.267 |
| BM25 | 0.200 | 0.067 | 0.067 |
| LSA | 0.400 | 0.267 | 0.267 |
| **Dense** (`all-MiniLM-L6-v2`) | **0.600** | **0.500** | **0.500** |

## Interpretation — the honest finding, now resolved
- **The three keyword methods (TF-IDF, BM25, LSA) tie on the standard set (MRR/MAP = 0.900).** On this
  small, low-redundancy KB, BM25's length-normalisation and LSA's SVD add nothing over plain TF-IDF —
  they all rank by literal term overlap (LSA is built *on* TF-IDF; the SVD is near-lossless here, see
  the next section). The single shared miss is *"how hot does it get"*: the word *"hot"* never appears
  in the KB, so a keyword method gives it zero weight.
- **The dense retriever wins — and the earlier prediction is now measured, not asserted.** On the
  standard set it lifts MRR **0.900 → 0.933** and R@3 → **1.000** (it alone catches "how hot"). On the
  paraphrase set it roughly **doubles** TF-IDF/LSA (MRR **0.267 → 0.500**) and triples BM25 — because a
  sentence encoder maps *meaning*, so "hot" ≈ "temperature" even with no shared words. This is the
  direct, quantified confirmation of the recommendation made when only TF-IDF/LSA existed.
- **Reported honestly, dense is better but not a silver bullet:** on paraphrases it still only reaches
  MRR 0.500 (3 of 5 in the top-3). Even a 384-dim MiniLM misses hard paraphrases on a 14-doc KB;
  a larger encoder or an enriched KB would help.
- **BM25 trails TF-IDF on paraphrases (MRR 0.067 vs 0.267) for a legitimate reason.** BM25 uses the
  *same* stop-word removal as the TF-IDF baseline (added for a fair comparison — otherwise BM25 matched
  the stop-word "it" and returned artefactual results). With stop words gone, a paraphrase leaves at
  most a single *generic* content word matching (e.g. only "air" in "how heavy or thin is the air"),
  and BM25's length-normalised, TF-saturated weighting of that one term ranks the target sensor lower
  than raw TF-IDF does — so it misses `rho` where TF-IDF catches it. When no content word matches at
  all ("how hot"), BM25 now correctly returns nothing. The takeaway is the same: keyword scoring on a
  lone generic term is essentially noise; only the dense encoder matches on *meaning*.

*Numbers for the dense rows are produced with the model loaded locally (`all-MiniLM-L6-v2`, 384-dim);
they are excluded from `verify_results.py` to keep the trust gate dependency-light, and the dense/BM25
tests skip gracefully when the optional packages are absent.*

## Why the two scores look identical — defence with evidence

The tie is **not** a bug or a copy-paste retriever. Four pieces of evidence explain it, and the last
one proves the LSA implementation is genuinely working.

**(a) The metrics only grade the position of the one relevant doc.** Each labelled query has exactly
one relevant sensor doc, so MRR/MAP/R@3/NDCG depend *solely* on where that doc ranks. Both retrievers
place it at **rank 1 for 9/10 queries** (and neither retrieves it for *"how hot does it get"*), so the
graded numbers must be equal. The retrievers are **not identical rankings**, though: they agree on
top-1 **10/10** but their full top-3 differs on **3/10** queries — the differences are all in the
*irrelevant tail* (ranks 2–3), which the metrics never see. "Identical result" precisely means
"identical on the graded position."

**(b) The SVD is near-lossless here.** The term-document matrix is 14 docs × 84 terms, **rank 14**.
LSA keeps **13 of 14** dimensions → retains **94.9 % of variance** (discards 5.1 %). The singular
values are almost flat (1.38 → 0.84) — there is no dominant "topic" structure. So LSA is essentially
a *rotation of the TF-IDF space minus one tiny dimension*, which preserves cosine geometry (hence
rank-1).

**(c) There are no latent topics to exploit.** Mean pairwise TF-IDF cosine between the 14 docs is
**0.055** (max 0.273) — the docs are **near-orthogonal**. LSA's mechanism is merging *co-occurring*
terms into shared topics; with almost no co-occurrence there is nothing to merge, so SVD can only
reproduce its input.

**(d) The component sweep proves LSA is really operating** (not a no-op). Reducing the latent
dimensions makes LSA diverge from TF-IDF and degrades the metrics monotonically, tracking the variance
retained — exactly as theory predicts:

| n_components | variance retained | MRR | MAP | top-1 == TF-IDF |
| ---: | ---: | ---: | ---: | ---: |
| 13 (default) | 94.9 % | 0.900 | 0.900 | 10/10 |
| 8 | 63.9 % | 0.800 | 0.800 | 8/10 |
| 5 | 41.9 % | 0.700 | 0.700 | 7/10 |
| 3 | 24.3 % | 0.600 | 0.600 | 6/10 |
| 2 | 15.7 % | 0.500 | 0.500 | 4/10 |
| 1 | 4.5 % | 0.183 | 0.183 | 2/10 |

**Conclusion for a viva:** *LSA ties TF-IDF because (a) the metrics see only the single relevant doc's
rank, which both give as rank 1 for 9/10 queries, and (b) on this KB the SVD is near-lossless (13/14
dims, 94.9 % variance) over near-orthogonal docs (mean cosine 0.055), so there are no latent topics to
gain from. The sweep confirms the LSA is genuinely working — compressing the space makes it diverge
and monotonically lowers MRR (0.90 → 0.18) in lock-step with retained variance. LSA cannot beat TF-IDF
here because compression can only lose information when there is no redundancy to remove; its benefit
appears on larger corpora with synonymy and term co-occurrence.*

Reproduce this evidence:
```bash
python3.12 - <<'PY'
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src import rag_system as rag, retrieval_eval as re
X = TfidfVectorizer(stop_words='english').fit_transform(rag.SENSOR_DOCS.values())
lsa = rag.LSARetriever()
print('rank', np.linalg.matrix_rank(X.toarray()), '| variance kept', lsa._svd.explained_variance_ratio_.sum())
off = cosine_similarity(X)[~np.eye(14, dtype=bool)]; print('mean inter-doc cosine', off.mean())
tf = rag.Retriever()
for nc in (13, 8, 5, 3, 2, 1):
    m = re.evaluate_retriever(rag.LSARetriever(n_components=nc), k=3); print(nc, round(m['mrr'], 3))
PY
```

## KB enrichment — before/after (acting on the evaluation)

The evaluation above showed the bottleneck was the **knowledge base**, not the retriever: the
original 14 one-line docs used no everyday vocabulary, so paraphrases missed. Acting on that, the
docs were **enriched** with natural synonyms and "how it feels" language (e.g. the humidity doc now
says *"muggy, sticky, damp, humid, clammy"*; wind says *"windy, breezy, gusty, calm"*). A cheap
edit of ~14 strings — no new dependency. Re-measured on the *same* query sets:

**Paraphrase MRR, before → after enrichment:**

| retriever | before | after |
|---|---|---|
| TF-IDF | 0.267 | **1.000** |
| BM25 | 0.067 | **1.000** |
| LSA | 0.267 | **1.000** |
| Dense (MiniLM) | 0.500 | 0.867 |

(Standard-set MRR after enrichment also rose to ~0.95–1.0 for every retriever.)

**The finding — and it flips the earlier conclusion.** Enriching the documents closed the paraphrase
gap for *everyone*, and made cheap literal **TF-IDF perfect (1.000) — now ahead of the dense
encoder (0.867)**. So for a small, curated KB, **document quality dominates the choice of retriever**:
improving the ~14 descriptions beat adding a neural model. Two honest caveats:
- **Overlap with the probe.** The enrichment adds the standard everyday synonyms, and the paraphrase
  set probes exactly that language, so some of the jump reflects the KB now *containing* the vocabulary.
  A fully held-out test would use unseen phrasings; the fair claim is "better docs close the gap on the
  everyday language users actually employ."
- **Dense's residual miss.** Dense still ranks *"how breezy is it"* under the **gust** doc rather than
  **wind speed** (it conflates related meanings), and its longer multi-concept vectors blur slightly —
  which is *why* it now trails the keyword methods it previously beat.

**Practical takeaway for the system:** keep TF-IDF as the default (fast, dependency-free, now perfect
on this KB); the dense retriever remains valuable as the KB grows or for truly unseen phrasings.

## Held-out generalisation test (answering "did you teach to the test?")

The enrichment above adds the same everyday words the paraphrase set probes, so its near-perfect scores
partly reflect the KB now *containing* the vocabulary. To measure genuine generalisation we use a
**held-out set** (`HELDOUT_GROUND_TRUTH`, n=6) whose every descriptive word is **verified absent** from
the KB (`sweltering`, `nippy`, `oppressive`, `sultry`, `blustery`, `gale`, `buffeting`, `weighty`,
`compressed`, …). No retriever has seen these words, so a keyword method has nothing to match.

| held-out (unseen words) | R@3 | MRR | MAP |
|---|---|---|---|
| TF-IDF / BM25 / LSA | 0.333 | **0.250** | 0.250 |
| Dense (`all-MiniLM-L6-v2`) | 0.667 | **0.556** | 0.556 |

**Finding (the fair number):** on truly unseen vocabulary the keyword methods collapse to ~0.25 (they
only score at all via generic words like "air", mostly wrong), while the **dense encoder genuinely
generalises to ~0.56** — matching *sweltering*→temperature, *buffeting gusts*→gust, *weighty/compressed
air*→density from its pretraining, with no shared word. It is honestly **not perfect** (it misses
*sweltering*→T and *oppressive/sultry*→rh, mapping them to nearby humidity/vapour docs). This is the
defensible claim: *enriched docs win on the everyday words users actually use, and the dense retriever
is the one that still helps when a query uses words the KB has never seen.* Guarded by
`tests/test_retrieval_eval.py` (keyword weak on held-out; dense > keyword when the model is available).

## For the report
- Cite the retriever as the module's **VSM baseline** (TF-IDF + cosine) and this evaluation as an
  application of the **IR Evaluation Measures** (P@k/R@k/MRR/MAP/NDCG).
- Report the honest result: on a small curated KB, LSA ≈ TF-IDF; the win from semantic methods appears
  only under vocabulary mismatch, which the labelled set exposes and the dense retriever addresses.
- The project's **faithfulness gate** (`llm.faithfulness`) is the RAG **Faithfulness** evaluation
  measure from the RAG lecture; **retrieval** quality is now evaluated separately here — together they
  cover both halves of a RAG system (retrieve well, generate faithfully).

## Reproduce
```bash
cd code/v1
python3.12 verify_results.py     # 29/29 (includes the two retrieval checks)
python3.12 -m pytest tests/test_retrieval_eval.py tests/test_rag.py
python3.12 -c "from src import rag_system, retrieval_eval as re; \
print(re.evaluate_retriever(rag_system.Retriever(), k=3)); \
print(re.evaluate_retriever(rag_system.LSARetriever(), k=3))"
```

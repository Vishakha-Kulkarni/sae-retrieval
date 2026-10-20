# Report outlines owned by P5 (draft 20 Oct 2026)

Prose for these sections waits for the 3 Nov freeze, because it depends on numbers. Placeholders name the
cell of the `python -m p5.table` output they come from: `[row, column]`, for example `[L, Mis@1k MRR@10]`.
Report freeze 3 Nov, draft 7 Nov, submission 10 Nov.

## Combined results (about 0.75 page, with the ablation table)

1. **Setup, two sentences.** Every row starts from the same frozen SimLM embeddings, SAE checkpoint and CL-SR
   index, and is tuned on the 5,000 validation queries only. Dev and TREC DL ran once. Name the baseline SAE
   (Efficient k = 32 or Max k = 128, team decision) and state our reproduced `[B0, MRR@10]` next to the
   paper's number (0.343 or 0.368). Say plainly that it's our reproduction, not the paper's figure.
2. **The table** (`ablation.tex`): B0, P1–P5 rows, then the combined row. One sentence on reading it: †
   marks a significant difference from B0, and the cost columns are on the right.
3. **Which pieces help, in order of `[row, MRR@10]` gain over B0.** For each row: the gain, whether it's
   significant, and what it costs (`Concepts/q`, `Index GB`, `ms/q`). Pieces that change passages (P2, P3,
   P4) move `Index GB`; P1 and P5 don't.
4. **Mismatch queries.** `[*, Mis@1k MRR@10]` against SimLM's 0.185 (paper Table 5). This is where P5
   should show up, and if it doesn't, say so here.
5. **Do gains add up?** Compare the combined row with the sum of single-piece gains. If P3 also changes
   queries, the combined run applies P3's threshold before P5's expansion (method note, P3 boundary).
6. **Negative results** stay in the table, each with a one-sentence reason.

## Ablation discussion (about 0.3 page)

- P5 internal: `[L, *]` against `[T, *]` and `[N, *]`, with the Holm-corrected p-values from
  `family.dev.json` (6 tests). Only those support "L beats T/N". If L ≈ T, near-miss concepts do most of the
  work and a lower query threshold is the cheaper fix; say that directly.
- Budget trade-off: recall against `ms/q` and postings touched across B ∈ {4, 8, 16} (validation row JSONs,
  or a small figure if space allows).
- One qualitative example: a mismatch query, the concepts L adds and their SAE descriptions, and where the
  relevant passage moves from and to. Pick it from the validation set, never from dev.

## Conclusion (about 0.2 page)

- One sentence on the question: does training any stage of CL-SR to rank, instead of leaving it hand-set,
  close the gap to SimLM?
- The answer in numbers: best combined `[*, MRR@10]` and `[*, R@1k]` against B0 and against SimLM's 0.411
  (paper Table 1, retrieval with the dense vectors).
- What it costs relative to B0 in index size and latency.
- Limitations: one SAE size, MS MARCO only (no BEIR transfer), single-threaded CPU latency on the NSCC
  hardware, three seeds only for the final rows.

## P5's minute of the video (about 130 words spoken)

| Time | Shows on screen | Says |
|---|---|---|
| 0:00–0:15 | A mismatch query and the passage CL-SR misses | The query and the passage share no active concept, so CL-SR scores it zero. |
| 0:15–0:35 | Pool → MLP → top-B diagram | We add a few concepts to the query only; the index stays the same. |
| 0:35–0:50 | `[L, Mis@1k MRR@10]` and `[L, R@1k]` against B0 and T | The result, significant or not, next to the threshold control. |
| 0:50–1:00 | `ms/q` and `Concepts/q` for B0 and L | What the extra postings cost per query. |

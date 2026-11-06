# Shared AAAI report draft

Course requirement supplied by the professor: **at most six content pages, excluding references; AAAI format; LaTeX preferred**. Conference submission limits, copyright transfer, and reproducibility-checklist requirements are not additional course requirements.

This draft uses the official AAAI 2026 author kit, downloaded directly from AAAI. `aaai2026.sty` and `aaai2026.bst` are unchanged; download provenance and hashes are in `author-kit-source.json`. The supported `draft` option retains the author block without claiming AAAI publication copyright. Replace the author placeholder with the agreed team names and order.

## Edit and build

Upload this entire directory as a ZIP to Overleaf and select `main.tex` as the main document, using pdfLaTeX. Locally, run `make` with Tectonic installed, or `latexmk -pdf main.tex` with a standard TeX distribution. Do not change margins, font sizes, column spacing, or the official style to squeeze in content.

`main.pdf` is a working preview, not a submission-ready report. Bold Draft markers identify incomplete sections. The current abstract and conclusion describe completed P1 evidence only; rewrite both after integrating team results. P1 tables come from `experiments/p1/full_corpus_heldout.json` and `experiments/p1/vocabulary_mismatch.json`. Other owners should verify their final experimental scope before adding numbers.

## Ownership and approximate six-page budget

| Section | Owner | Content-page budget |
|---|---|---:|
| Abstract and introduction | P1, team final pass | 0.7 |
| Related work | P2 | 0.5 |
| Base pipeline | P3 | 0.5 |
| Five method subsections | Each person | 1.5 |
| Experimental setup | P4 | 0.6 |
| Results, efficiency and combined ablation | P5 integrates; each owns results | 1.8 |
| Conclusion and limitations | P5 | 0.4 |

References are outside this budget. Budgets guide editing; verify the compiled PDF stays within the professor's limit as sections grow. Avoid treating differing index backends, checkpoints, or sparsity settings as a controlled scoring-only comparison. P1's higher latency, lower TREC recall, and nonsignificant DL19 comparison must remain visible.

Use the shared report branch `report/aaai-draft` for report changes. Keep experimental development on separate branches. This draft PR stays open for collaboration until the team is ready to merge it.

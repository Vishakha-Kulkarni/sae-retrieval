"""P4 Pruning generic concepts

P4: Pruning generic concepts


Problem: some concepts fire on a large share of passages, like stopwords. They create very long posting lists, slow search and add little ranking signal.


Build: a learned concept gate: one trainable weight per concept, trained with the dense-score KL loss plus a sparsity penalty (L1 or hard-concrete L0) that pushes useless concepts to zero. A document-frequency cutoff is the simple baseline. Then profile latency, index size and memory for every variant, including the other four people's outputs.

Why it matters: efficiency is half of the paper's claim, and a learned gate decides what to drop from data rather than from a hand-set rule.

Output: index size and latency against MRR@10 for each sparsity-penalty strength.





"""


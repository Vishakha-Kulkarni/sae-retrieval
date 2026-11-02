"""Generate P1's report contribution and one-minute video script from completed held-out results."""
import argparse
import json
from pathlib import Path
from .train import file_hash


def package(summary_path, validation_path, output):
    summary = json.loads(Path(summary_path).read_text())
    protocol = json.loads(Path(validation_path).read_text())
    if summary['protocol_sha256'] != file_hash(validation_path):
        raise ValueError('Report protocol does not match held-out evaluation')
    if set(summary['splits']) != {'dev', 'dl19', 'dl20'}:
        raise ValueError('Cannot write final results before all held-out splits complete')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    table = []
    pvalues = []
    for split, label, metric in [('dev', 'MS MARCO dev (MRR@10)', 'mrr_at_10'),
                                  ('dl19', 'TREC DL 2019 (NDCG@10)', 'ndcg_at_10'),
                                  ('dl20', 'TREC DL 2020 (NDCG@10)', 'ndcg_at_10')]:
        data = summary['splits'][split]
        m = data['metrics']
        learned = data['learned_three_seed_summary'][metric]
        table.append(f"{label} & {m['released_bm25'][metric]:.4f} & {m['tuned_bm25'][metric]:.4f} & "
                     f"{learned['mean']:.4f} $\\pm$ {learned['sample_std']:.4f} \\\\")
        ps = data['primary_seed_comparison']['paired_randomization_tests']
        pvalues.append(label + ': ' + ', '.join(f"{r['reference']} adjusted $p={r['p_holm']:.4g}$" for r in ps))
    dev = summary['splits']['dev']
    baseline = dev['metrics']['tuned_bm25']
    learned = dev['learned_three_seed_summary']['mrr_at_10']
    mean_latency = sum(dev['metrics']['learned_s' + s]['mean_latency_ms'] for s in ('42', '43', '44')) / 3
    index_gib = baseline['index_bytes'] / (1024 ** 3)
    latex = r'''% P1 contribution fragment; merge into the team's AAAI report.
\section{Introduction}
Passage retrieval is a central component of question answering. Dense retrievers
encode text into vectors that support semantic matching, but their coordinates
are difficult to interpret. Sparse autoencoders offer an alternative: represent
each embedding by a small set of active latent concepts and retrieve passages
through an inverted index. We build on Concept-Level Sparse Retrieval
(CL-SR)~\cite{park-etal-2025-decoding}.

The base SAE objective reconstructs embeddings rather than directly optimizing
the final sparse ranking, and CL-SR applies a fixed BM25-style formula to learned
activation magnitudes. Our project investigates five independent modifications:
learned impact scoring, ranking-aware SAE training, adaptive passage sparsity,
concept pruning, and query expansion. We compare each change against our own
reproduced baseline with shared artifacts and validation-only selection, measuring
ranking quality, index storage, and retrieval latency. This contribution isolates
the scoring rule while freezing the representations and indexed concepts.

\subsection{P1: Learned impact scoring}
We freeze the shared SimLM embeddings and B0 SAE checkpoint. For each concept
shared by a query and passage, a one-hidden-layer MLP predicts a nonnegative
contribution. Its four inputs are the log-transformed query and passage
activations, the concept's corpus IDF, and the log-transformed passage concept
count relative to the corpus mean. The hidden layer has 32 ReLU units; the output
uses softplus. The scorer has 193 trainable parameters.
\begin{equation}
s_\theta(q,d)=\sum_{c\in\mathrm{supp}(q)\cap\mathrm{supp}(d)}f_\theta(\phi(q_c,d_c)).
\end{equation}
The additive form supports evaluation over concept posting lists without a
dense-teacher candidate filter at inference. We train by listwise KL distillation
from SimLM scores over 100 dense training candidates per query, with temperature
1 and Adam learning rate $10^{-3}$. We use the same 5,000 sampled training queries
for seeds 42, 43, and 44, train for ten epochs, and select each checkpoint by
MRR@10 on the shared 5,000-query validation split. Passage representations retain
at most 24 concepts. Default BM25 uses the released constants; tuned BM25 uses
a predeclared 28-setting grid on 500 validation queries, followed by confirmation
of the top three, the released setting, and the previous candidate-tuned setting
on all 5,000 validation queries. No dev or TREC results select settings.

\subsection{P1 results}
Table~\ref{tab:p1} reports full-corpus retrieval with the same frozen concept
index for all methods. P1 values are means and sample standard deviations over
three seeds. MS MARCO dev MRR@10 is primary; TREC NDCG@10 uses linear relevance
gains, matching the base evaluator. The predeclared primary seed 42 is compared
query-wise against the two baselines with 10,000 paired randomizations and Holm
correction within each split. TREC comparisons are secondary.
\begin{table}[t]
\centering
\small
\begin{tabular}{lrrr}
\hline
Dataset / metric & Released & Tuned & P1 \\
\hline
'''
    latex += '\n'.join(table) + '\n'
    latex += r'''\hline
\end{tabular}
\caption{P1 held-out full-corpus retrieval; tuning is confined to validation.}
\label{tab:p1}
\end{table}
'''
    latex += '\n\noindent ' + '; '.join(pvalues) + '.\n'
    latex += (f"The shared serialized index occupies {index_gib:.3f} GiB. P1 adds its small scoring checkpoint; "
              "it does not alter the postings. On " + str(baseline.get('gpu_name')) + ', '
              f"mean dev retrieval latency is {baseline['mean_latency_ms']:.2f} ms for tuned BM25 and "
              f"{mean_latency:.2f} ms for P1 averaged across seeds. These measurements include posting transfers, "
              "scoring, and ranking after warmup, but exclude text encoding, SAE query encoding, and index loading. "
              "They therefore measure retrieval latency, not end-to-end question-answering latency.\n")
    latex += ("A validation text audit found two questions whose case/punctuation-normalized text matched "
              "training questions. Excluding them did not explain the validation gain; we retained the common split "
              "and report the sensitivity analysis separately. Three seeds and held-out evaluation assess stability "
              "and generalization, but do not prove the absence of overfitting. Vocabulary-mismatch subset results "
              "remain dependent on the shared subset definitions.\n")
    (root / 'p1_contribution.tex').write_text(latex)
    (root / 'p1_reference.bib').write_text('''@inproceedings{park-etal-2025-decoding,
  title={Decoding Dense Embeddings: Sparse Autoencoders for Interpreting and Discretizing Dense Retrieval},
  author={Park, Seongwan and Kim, Taeklim and Ko, Youngjoong},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025}, doi={10.18653/v1/2025.emnlp-main.1345},
  url={https://aclanthology.org/2025.emnlp-main.1345/}
}
''')
    video = ("My contribution is P1, learned concept impact scoring. The baseline turns dense embeddings "
             "into sparse concepts and ranks matching passages with BM25. I replace that formula with a small MLP "
             "with 193 trainable parameters. It learns from SimLM scores and assigns a contribution to each shared "
             "concept. Summing these contributions keeps scoring compatible with the concept index. We freeze "
             "the embeddings, SAE, and postings, and compare against released and tuned BM25. "
             f"On the untouched MS MARCO dev set, P1's three-seed mean MRR at ten is {learned['mean']:.4f}, "
             f"compared with {baseline['mrr_at_10']:.4f} for validation-tuned BM25. "
             "We also evaluate TREC, significance, index size, and latency. Settings are selected on validation, "
             "with test sets kept out of tuning. Held-out results test generalization; they cannot prove "
             "overfitting is impossible. The system retrieves passages rather than generating answers.")
    (root / 'p1_video_script.txt').write_text(video + '\n')
    (root / 'pending_dependencies.json').write_text(json.dumps({
        'P5': 'Exact shared validation/dev vocabulary-mismatch query-ID subsets (top 10/100/1000 definitions)',
        'team': 'Merge this contribution with the shared six-page AAAI report and combined ablation table',
        'P1_user': 'Record the approximately one-minute video contribution using the generated script'}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    package(a.summary, a.validation, a.output_dir)


if __name__ == '__main__':
    main()

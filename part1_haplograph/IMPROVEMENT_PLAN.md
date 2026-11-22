# Part 1 improvement plan

Plan produced with a structured brainstorm-and-review workflow, starting from the existing part 1 pipeline
(steps 0-5, see [../DOCUMENTATION.md](../DOCUMENTATION.md)). It is updated after each improvement
iteration; implemented items are marked **[done]**.

## Introduction

Haploblock clusters discretize the genome into a vocabulary of inherited haplotype states, and the
1000 Genomes haplograph records how these states co-occur across 2,548 individuals. Part 1 asks
whether graph embeddings of these states, fed to the DNA foundation model Evo 2 in place of
nucleotide tokens, let Evo 2 learn haplotype structure that generalizes to individuals it has never
seen. The core question is therefore one of held-out generalization, not of fit on the training
half. The first chr21 runs show why this is hard: the median cluster is carried by one person, 97%
of nodes have no edge, 61,892 clusters occur only in the test half, and a simple bigram baseline
already predicts the next cluster 41% of the time. The improvements below make the evaluation
honest before any expensive GPU run: a held-out test of the embeddings themselves, uncertainty
estimates, and an information-theoretic reference for next-cluster prediction. Information theory and
population genetics give the interdisciplinary angle: conditional entropy measures how predictable the next state is from the current one, and replication of linkage (co-occurrence) in independent individuals is the classic test
of a real LD signal. The data is the public haplograph and the BH_26 split; nothing else is needed.
The deliverable is a validation report per fine-tuned model that says, with confidence intervals,
whether it beats simple baselines on the test half. Known risks: the Evo 2 and BioNeMo paths have
not yet run on a GPU, and the rare-cluster vocabulary may need collapsing (iteration 2) before
next-cluster prediction is a well-posed task.

## Hypotheses (scored)

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | Embeddings learned on train individuals predict cluster co-occurrence in the test individuals | Pursue (iteration 1) |
| H2 | Model-vs-baseline claims need paired bootstrap CIs over test individuals | Pursue (iteration 1) |
| H3 | Bigram conditional entropy on train is the reference cross-entropy a model with more context should beat | Pursue (iteration 1) |
| H4 | Accuracy within the true next region separates haplotype knowledge from position knowledge | Pursue (iteration 1) |
| H5 | Collapsing clusters with train support < k into one per-region rare token makes the task well-posed | Pursue (iteration 2) |
| H6 | One comparison table across embedding methods and frozen/fine-tuned models | Dropped in iteration 2 (summary.json already holds it; minimalism) |
| H7 | One config-driven runner for steps 0-5 | Dropped in iteration 2 (duplicates DOCUMENTATION section 11; minimalism) |
| H8 | Test accuracy stratified by population | Dropped for now (ancestry removed from the pipeline by design) |
| H9 | Phenotype prediction from individual embeddings | Dropped (no real phenotypes in 1000G) |
| H10 | Genome-wide (cross-chromosome) sequences | Dropped (memory / context) |
| H11 | Report the share of test targets that are merged rare tokens | Pursue (iteration 2, with H5) |
| H12 | Positive control: the step-4 learning setup reaches at least ~bigram test top-1 when the signal is learnable; otherwise the objective / head is broken | Pursue (iteration 3) |
| H13 | Score only the current and next region's tokens (exact, cheaper) | Dropped in iteration 3 (merging already cut the vocabulary; extra code) |
| H14 | GPU smoke mode for Evo 2 loading | Dropped in iteration 3 (cannot be verified without a GPU) |
| H15 | (number not used) | – |
| H16 | Calibration (ECE) of next-token probabilities | Dropped in iteration 3 (the loss already reflects it) |
| H17 | Positional signal in the mock backbone | Superseded in iteration 3: H12 failed for lack of genomic position in the inputs, fixed for every backbone by `--pos-freqs` |

## Pipeline changes

| Stage | Change | Hypothesis |
|---|---|---|
| 5 `embedding_cooccurrence` **[done, iteration 1]** | New CPU-only task: AUC of embedding cosine for cluster pairs that co-occur (lift >= threshold) among TEST individuals vs distance-matched random pairs, per embedding method; pairs in different regions only | H1 |
| 5 `test_set_performance` **[done, iteration 1]** | Paired cluster bootstrap over test individuals (95% CIs for model, bigram and their difference); region-conditioned top-1; bigram conditional entropy on train as a reference cross-entropy | H2, H3, H4 |
| 4 / 5 `--min-support` **[done, iteration 2]** | Rare-cluster merging; share of merged test targets reported | H5, H11 |
| ~~6~~ | ~~Model comparison table~~ (dropped, iteration 2) | H6 |
| ~~runner~~ | ~~Config-driven `run_pipeline.py`~~ (dropped, iteration 2) | H7 |
| 4 / 5 `--pos-freqs` **[done, iteration 3]** | Genomic-position encoding appended to embeddings (inputs and candidate keys); positive-control tests | H12 |

Review changelog: round 1 restricted H1 to pairs in different regions with a minimum test support and
distance-matched negatives (avoids "nearby = similar" and noisy rare-cluster lift); round 2 made
the bootstrap resample individuals (predictions within one person are correlated) and paired the
model-baseline difference; round 3 had no substantive change, so the review stopped.

## Iteration 1 results (chr21, BH_26, CPU)

- `embedding_cooccurrence`, first version (41,098 pairs): XGBoost AUC 0.713, node2vec 0.546, GNN
  0.534. The code review showed these labels mostly encoded cluster FREQUENCY (one chance co-carrier
  of two rare clusters gives a huge lift). Corrected version (distinct pairs, >= 5 expected test
  co-carriers, frequency-matched AUC, region-block bootstrap; 3,228 pairs, 116 positive): GNN 0.978
  [0.960, 0.989], XGBoost 0.968 [0.957, 0.979], node2vec 0.881 [0.855, 0.906], train-lift reference
  0.991, genomic proximity 0.618 (0.53 distance-matched), cluster frequency 0.354. For common clusters
  all three embeddings carry linkage that replicates in unseen individuals, close to the train-lift
  ceiling; rare clusters remain untestable this way (H5).
- `test_set_performance` (mock backbone): bigram top-1 0.414 [0.406, 0.423]; bigram plug-in
  conditional entropy 1.82 nats on train (a reference, not a floor: models with more context can
  go below it). A real Evo 2 run has to beat the bigram top-1 with a CI excluding 0.

## Iteration 2 (priority: minimal, non-redundant code)

| Item | Change |
|---|---|
| H5 + H11 | `--min-support k` in step 4 (`GenomeData`): clusters with training support < k become one token per haploblock region (embedding = mean of its members); default 2 (the second audit showed that at 1 the merged token never occurs in training, so it cannot be learned); carriers counted on step 4's own training individuals. Stored in the checkpoint and reapplied by step 5, which also reports the share of test targets that are merged |
| M1 | Module docstrings keep purpose, input/output files and usage; option tables and prose duplicated in DOCUMENTATION.md / `--help` are removed |
| M2 | Remove dead or duplicated code across part 1 |

Review changelog: round 1 checked that the collapsed token uses no test information (membership from
`support_train`, embedding = mean of train-only step-3 embeddings) and added a fallback to `support`
with a warning for `--no-split` graphs; round 2 kept the file-format contracts in the docstrings;
round 3 had no substantive change.

## Development plan

- Iteration 1: H1-H4 in step 5 (no GPU needed), tested on synthetic data and chr21.
- Iteration 2: H5/H11 and the minimalism pass M1-M2.
- Iteration 3: H12 positive control (regression test + chr21 CPU experiment); fix what it reveals.

## Iteration 3 (final)

Every model so far had test top-1 near 0 while the bigram reaches 0.44, and nothing showed that the
step-4 learning setup can learn at all. H12 is that positive control: on synthetic data the mock
backbone must reach >= 0.8 x the bigram's top-1 within a few seconds (fixed seed), and on chr21 a
longer CPU run shows how far it gets. Review changelog: round 1 set the pass bar at bigram level
(beating chance proves little); round 2 required a fast, deterministic test; round 3 no change.

Result: the control FAILED at first (synthetic: top-1 0.13 vs bigram 0.49 after 420 steps, while
region-conditioned top-1 was 0.66): the inputs and candidate keys carried no genomic position, so the
model could not tell which region comes next. Fix: `--pos-freqs` (default 16) appends a sin/cos
encoding of the cluster's start to every embedding (standardized; wavelengths 10 kb .. the graph's largest
start). Final control (test settings: 60 epochs, 317 predictions on step 4's validation individuals): top-1 0.48 with the
encoding vs 0.11 without, bigram 0.53 (0.9 x). A regression test asserts >= 0.8 x bigram, another
that the encoding is appended. On chr21, 3 CPU epochs of the small
mock: test top-1 0.041 with vs 0.010 without the encoding, bigram 0.44: a real GPU training budget is
needed to know whether Evo 2 beats the bigram there.

## Testing steps

- Every script: `--test` (synthetic haplograph, CPU, seconds).
- `embedding_cooccurrence`: on the synthetic graph (no positional linkage) genomic proximity must stay
  near AUC 0.5 and the train-lift reference well above it; GNN / XGBoost should approach that
  reference (node2vec is weak there under the tiny self-test walk budget, so it is not asserted). The step-5
  `--test` asserts these properties. On chr21, compare methods.
- `test_set_performance`: the CIs must contain the point estimates; with `--splits none` the
  model's stored validation accuracy must still be reproduced exactly (when step 4 evaluated all
  validation sequences, i.e. at most `--val-examples`).
- End to end: the real chr21 run of DOCUMENTATION.md section 11.2.

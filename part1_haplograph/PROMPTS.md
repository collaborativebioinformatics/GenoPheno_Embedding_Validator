# Prompts for implementing the part 1 improvement plan

Each prompt is self-contained for a fresh agent session working in `part1_haplograph/` of this
repository. Read `../DOCUMENTATION.md` and `IMPROVEMENT_PLAN.md` first. Never modify `part2_proteomics/`.

## Iteration 1 (step 6, no GPU needed)

**H1 implement.** In `6_validate_evo2.py`, add a task `embedding_cooccurrence` that, for every
`node_emb_<method>.npy` in the graph folder, samples cluster pairs from different haploblock regions
on the same chromosome whose co-occurrence lift among the TEST individuals (carriers.npz columns of
the test set) is at least `min_lift` (both clusters with test support >= `min_support`), samples
negatives at matched genomic distance, and reports the ROC AUC of embedding cosine similarity per
method, with a bootstrap CI. Use scipy only.

**H1 test.** Extend the step-6 `--test` so `embedding_cooccurrence` runs on the synthetic graph, and
assert that genomic proximity stays near 0.5, the train-lift reference is clearly above 0.5, and GNN is above 0.5 (node2vec is too weak under the tiny self-test settings to assert).

**H1 wire.** Add the task to `validation_tasks.txt` and the default task list, and document it in
DOCUMENTATION.md section 4 (step 6).

**H2-H4 implement.** In `test_set_performance`, record per-individual correct counts for the model
and the bigram baseline; add a paired bootstrap over test individuals (95% CI of model top-1, bigram
top-1 and their difference); add region-conditioned top-1 (argmax over the clusters of the target's
region); compute the bigram conditional entropy H(next | current) on the training individuals and
report it next to the test cross-entropy.

**H2-H4 test.** Check on the synthetic data that each CI contains its point estimate, that
region-conditioned accuracy >= unconditioned accuracy, and that `--splits none` still reproduces the
stored validation accuracy.

**H2-H4 wire.** Report the new numbers in report.md / summary.json and the per-chromosome CSV.

## Iteration 2 (minimal, non-redundant code)

**H5 implement.** In `evo2_model.GenomeData`, add `min_support` (default 1): nodes whose `support_train`
(else `support`, with a warning) is below it are mapped to one representative per haploblock region,
whose embedding is the mean of the members; `tokens`, `candidates` and the local indices use the
collapsed vocabulary. Add `--min-support` to step 4, store it in the checkpoint, apply it in
`load_finetuned`, and report in `test_set_performance` the share of test targets that are collapsed.
Keep it to a few dozen lines.

**H5 test.** On the synthetic data, check that with a high `--min-support` the number of candidates
per chromosome drops, the model still trains and step 6 reproduces the stored validation accuracy.

**M1-M2 implement.** In every part 1 script, cut the module docstring to purpose, input/output files
and usage (options live in `--help`, the rest in DOCUMENTATION.md); remove dead code and helpers
duplicated across scripts. Every `--test` and `python -m pytest tests` must still pass; report the
line count before and after.

## Iteration 3 (positive control)

**H12 implement / test.** Add a regression test that trains step 4 with `--mock-evo2` on the synthetic
haplograph for enough steps (fixed seed, < 30 s on CPU) and asserts that its validation top-1 is at
least 0.8 x the bigram's top-1 on the same validation individuals. If it fails, find out why (head,
temperature, learning rates, positional information in the mock) and fix the cause in
`evo2_model.py` / step 4, keeping the code minimal. Then run a longer chr21 CPU experiment and record
the result in DOCUMENTATION.md.

## Integration and documentation

Run every `--test`, then the chr21 CPU run of DOCUMENTATION.md section 11.2, and update
DOCUMENTATION.md (script reference, limitations) and `../prompt_history`.

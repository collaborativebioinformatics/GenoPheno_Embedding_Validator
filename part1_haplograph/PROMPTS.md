# Prompts for implementing the part 1 improvement plan

Each prompt is self-contained for a fresh agent session working in `part1_haplograph/` of this
repository. Read `../DOCUMENTATION.md` and `IMPROVEMENT_PLAN.md` first. Never modify `part2_proteomics/`.

## Iteration 1 (step 5, no GPU needed)

**H1 implement.** In `5_validate_evo2.py`, add a task `embedding_cooccurrence` that, for every
`node_emb_<method>.npy` in the graph folder, samples cluster pairs from different haploblock regions
on the same chromosome whose co-occurrence lift among the TEST individuals (carriers.npz columns of
the test set) is at least `min_lift` (both clusters with test support >= `min_support`), samples
negatives at matched genomic distance, and reports the ROC AUC of embedding cosine similarity per
method, with a bootstrap CI. Use scipy only.

**H1 test.** Extend the step-5 `--test` so `embedding_cooccurrence` runs on the synthetic graph, and
assert that genomic proximity stays near 0.5, the train-lift reference is clearly above 0.5, and GNN is above 0.5 (node2vec is too weak under the tiny self-test settings to assert).

**H1 wire.** Add the task to `validation_tasks.txt` and the default task list, and document it in
DOCUMENTATION.md section 4 (step 5).

**H2-H4 implement.** In `test_set_performance`, record per-individual correct counts for the model
and the bigram baseline; add a paired bootstrap over test individuals (95% CI of model top-1, bigram
top-1 and their difference); add region-conditioned top-1 (argmax over the clusters of the target's
region); compute the bigram conditional entropy H(next | current) on the training individuals and
report it next to the test cross-entropy.

**H2-H4 test.** Check on the synthetic data that each CI contains its point estimate, that
region-conditioned accuracy >= unconditioned accuracy, and that `--splits none` still reproduces the
stored validation accuracy.

**H2-H4 wire.** Report the new numbers in report.md / summary.json and the per-chromosome CSV.

## Iteration 2

**H5 implement.** Add `--min-support k` to step 4: clusters with training support < k are replaced
by one pseudo-token per haploblock region whose embedding is the mean of its members; targets and
candidates use the collapsed vocabulary; store the mapping parameters in the checkpoint, and make
`evo2_model.load_finetuned` and step 5 apply the same mapping.

**H6 implement.** Add `6_compare_models.py` that reads several `validation_*/summary.json` files and
writes one comparison table (method, backbone, frozen/fine-tuned, test top-1 with CI, difference to
bigram with CI, co-occurrence AUC).

## Iteration 3

**H7 implement.** Add `run_pipeline.py` that runs steps 0-5 from one YAML/JSON config (chromosomes,
split, methods, backbone, tasks), skipping steps whose outputs are current, with `--test`.

## Integration and documentation

Run every `--test`, then the chr21 CPU run of DOCUMENTATION.md section 11.2, and update
DOCUMENTATION.md (script reference, limitations) and `../prompt_history`.

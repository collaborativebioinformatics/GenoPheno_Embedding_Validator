# proteomics_emb: Generate embeddings of the proteomics data with ProLM for latefusion

We use the code from [ProLM](https://github.com/Qiu-Shizheng/ProLM).

The file "finetune.py" contains the following changes as compared with the main branch of ProLM:

- saving the backbone model itself (`backbone.pt`, at the best epoch);
- saving one embedding per patient (`patient_embeddings.csv`: `eid`, `e0` ... `e<hidden_size - 1>`), computed once
  after training from the selected model, in evaluation mode (no dropout, no gradients), in batches and on the
  model's device;
- when no `--pretrained-checkpoint` is given, all parameters are trained. ProLM's `build_model` always calls
  `freeze_backbone`, which otherwise leaves the encoder, the feature and expression projections and the [CLS]
  token at their random initial values (only the classifier head would train).

**Shape of the embeddings.** The backbone splits the proteins into chunks of `--chunk-size`, runs each chunk with
its own [CLS] token and averages the chunk [CLS] vectors. The result is one vector per patient,
`(n_patients, hidden_size)`, which is what late fusion needs. Per-protein vectors are not returned by ProLM; getting
them would need a change to `model.py`.

**Still to handle for a fair fusion comparison.** `finetune.py` makes its own train/test split (`--test-size`) and
picks the best epoch on that test split. The patient embeddings of the training patients are therefore fitted to
their labels. For late fusion, train ProLM inside each fold of the downstream evaluation and take the embeddings
from that fold's model.

So far this has been tested with the example as provided in the documentation of ProLM.

Next we will test that this works with the data from our project, first of all "synthtan_prolm".

Note: `finetune.py` uses relative imports (`from .data import ...`), so it runs only from inside the ProLM package:
install a fork of ProLM in editable mode (`pip install -e .`) with this file as `src/prolm/finetune.py`, then run
`python -m prolm.finetune`.


## Usage

Needs Python 3 with `torch` and `pandas` .

```bash
python -m prolm.finetune \
  --expression-csv path/to/expression.csv \
  --labels-csv path/to/labels.csv \
  --protein-embeddings path/to/protein_embeddings.npz \
  --output-dir path/to/output/ \
  --epochs 2 \
  --batch-size 4 \
  --hidden-size 32 \
  --num-layers 1 \
  --chunk-size 512
```

## Output

| Folder | Contents |
|---|---|
| `--output-dir` | `best_classifier.pt`, `backbone.pt`, `patient_embeddings.csv`, `metrics.json`, `protein_order.txt` |



## Fold-internal embeddings for late fusion (`cv_embeddings.py`)

`finetune.py` makes one train/test split, picks the best epoch on the test split and embeds every patient with
that model, so its embeddings cannot be used for a fair late-fusion comparison. `cv_embeddings.py` trains ProLM
from scratch inside each cross-validation fold instead:

- one model per fold on all tasks at once (one sigmoid output per `labels_<TASK>.csv`), every parameter trainable;
- the epoch is chosen on a validation part of the fold's training patients (`--val-frac`, early stopping on
  validation loss); the test fold is used only for the final metrics;
- labels are joined to the expression rows by `eid`; inputs must be numeric, complete and finite (ProLM's loader
  would otherwise fill gaps with the mean over all patients); the folds are stratified (`--stratify-task`) or read
  from `--folds-csv` (`eid, fold`), so every arm of the comparison can use the same folds;
- the output folder must be new, empty or this script's run with the same inputs, settings, device, versions and
  code; completed folds are skipped on a rerun (`--overwrite` recomputes), each fold is moved into place only when
  complete, and `summary.csv` is rebuilt from the completed folds after each fold. Nothing is deleted in a folder
  the script did not make.

Per fold it writes `patient_embeddings.csv` (`eid, e0 ...`; all patients, embedded by that fold's model),
`test_predictions.csv` (ProLM's own head), `metrics.json` (with class counts per task), `fold_sets.csv`
(`eid, set`; lists all patients, so never use it as `--train-individuals`) and `<run>_fold<k>_train.txt` /
`<run>_fold<k>_test.txt` (one id per line; `<run>` = `--run-name`, default the output folder's name). These are the
format of `--train-individuals` in `1_build_graph.py`, `3_node_embeddings.py` and `4a_finetune_evo2_embeddings.py`,
and of `--train-individuals` / `--test-individuals` in `5_validate_evo2.py`. For SynthTAN the ids (`S001` ...) are
the individual names of the genome graph, so no `--sample-map` is needed. The full usage is in the script's
docstring; `finetune_on_synthetic_data.ipynb` runs it on one cohort.

```bash
python proteomics_emb/cv_embeddings.py \
  --expression-csv synthtan_prolm_inputs/cohort_n200/expression.csv \
  --labels-dir synthtan_prolm_labels/cohort_n200 \
  --protein-embeddings synthtan_prolm_inputs/protein_embeddings.npz \
  --output-dir prolm_cv/cohort_n200 --device cuda
```

**Using the embeddings.** Join a fold's protein and genome embeddings on `eid`, train on the fold's training
patients and test on its test patients. Never mix folds (each fold's model has its own vector space) or cohorts
(the SynthTAN cohorts reuse the ids `S001` ...). Compare genome only, protein only and genome + protein with the
same downstream learner, training rows and tuning budget; `test_predictions.csv` is a separate baseline. The
training patients' embeddings come from a model trained or selected on their labels, so they may look more
separable than the test patients' (stacking optimism): compare arms on the test folds only.

**Genome side, still to do.** Passing the training file to the genome steps is necessary but not sufficient:

- The SynthTAN builder files (`builder/nodes.csv`, `edges.csv`) are genome-wide CSVs. `1_build_graph.py` expects
  `<local-dir>/<chrom>/nodes.csv.gz` and `edges.csv.gz`, defaults to chr1–chr22 (`--chroms chr101,...,chr122` is
  needed) and downloads from data.haploblocks.org when a local file is missing. A small converter is still needed.
- `1_build_graph.py` reuses cached parts in `<workdir>/parts/<chrom>/` without checking which cohort they came
  from, and names its default output folder after the training file, reusing an existing folder of that name.
  Use a separate `--workdir` per cohort and a separate `--out` per fold, and check the graph folder's
  `train_individuals.txt` against the fold's file.
- No step writes a per-patient genome embedding yet (step 4a saves only the fine-tuned model).
- The SynthTAN builder regions, and the builder's filter on node support (`min_support`), were computed from all
  patients of a cohort, and step 1 keeps only edges within a chromosome. Step 1's train-only mode recomputes edge
  weights but cannot undo that, so a genome arm built on these files is not fully held out with respect to region
  discovery.

Needs ProLM installed as the `prolm` package (`pip install -e .` in a ProLM clone).

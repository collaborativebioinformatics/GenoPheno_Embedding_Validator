# fusion: per-patient genome and protein inputs for late fusion (SynthTAN)

Builds, for each SynthTAN cohort, one folder in which the genome side and the protein side share the same
patients (`eid`), the same cross-validation folds and one table layout, so the two can be joined per patient.
All data are synthetic.

| Script | What it does |
|---|---|
| `synthtan_to_haplograph.py` | splits a cohort's genome-wide builder files (`builder/nodes.csv`, `edges.csv`) into the per-chromosome layout `1_build_graph.py` reads (`<chrom>/nodes.csv.gz`, `edges.csv.gz`, plus `chroms.txt`) |
| `make_fusion_inputs.py` | writes the shared layout: folds, labels, train/test lists, the raw 0/1 genome vector, the ProLM embeddings and the simple per-label protein vectors (see its docstring) |
| `run_genome_folds.sh` | per fold: step 1 (graph from the fold's training patients), edge filter, step 3 (node2vec), per-patient node2vec means; with `EVO2=1` also step 4a (Evo 2) and per-patient Evo 2 embeddings |
| `filter_graph_support.py` | after step 1: removes edges whose nodes have fewer than 3 carriers or non-carriers among the fold's training patients (the builder's rule, applied to the training patients) |
| `genome_patient_embeddings.py` | one vector per patient from a fold's node embeddings (mean) or from a step-4a Evo 2 model |
| `tests/mock_evo2/` | a tiny fake `evo2` package for testing the Evo 2 plumbing on a CPU; its outputs are meaningless |

## Layout (one folder per cohort)

```
folds.csv                               eid, fold   (the folds of proteomics_emb/cv_embeddings.py)
labels/labels_<TASK>.csv                eid, label
genome_raw_columns.txt                  node id of each column of genome_embeddings_raw.csv
fold<k>/<run>_fold<k>_train.txt         training patients, one id per line (--train-individuals of steps 1-4a)
fold<k>/<run>_fold<k>_test.txt          test patients
fold<k>/genome_embeddings_raw.csv       eid, g0 ...   0/1 state per builder node, no model (same in every fold)
fold<k>/genome_embeddings_node2vec.csv  eid, g0 ...   mean of the patient's node2vec node embeddings (64)
fold<k>/genome_embeddings_evo2.csv      eid, g0 ...   Evo 2 read-out states averaged over the patient's nodes
fold<k>/protein_embeddings.csv          eid, p0 ...   ProLM embedding of the fold (32; shared across labels)
fold<k>/protein_<TASK>.csv              eid, p0 ...   simple protein vector for one label: the 20 proteins with
                                                      the largest |Welch t| among the fold's training patients
fold<k>/protein_<TASK>_columns.txt      protein id of each column of protein_<TASK>.csv
```

Every table has one row per patient (training and test). For fold k and a label: join a genome table and a
protein table on `eid`, fit everything (scaling, PCA, classifier) on the rows of the training list and score on
the test list. Genome tables are shared across labels; `protein_<TASK>.csv` belongs to one label. Never mix folds
(each fold has its own models) or cohorts (the cohorts reuse the ids `S001` ...). Compare genome only, protein
only and joined inputs with the same downstream model and training rows.

## Reproduce here (ProLM on a GPU; genome side on a CPU, node2vec, no Evo 2)

```bash
REPO=$PWD/GenoPheno_Embedding_Validator; D=$PWD/demo; B=$PWD/fusion_run; c=cohort_n200
python $REPO/proteomics_emb/cv_embeddings.py --expression-csv $D/synthtan_prolm_inputs/$c/expression.csv \
  --labels-dir $D/synthtan_prolm_labels/$c --protein-embeddings $D/synthtan_prolm_inputs/protein_embeddings.npz \
  --output-dir $B/prolm_cv/$c --device cuda
python $REPO/fusion/synthtan_to_haplograph.py --builder-dir $D/synthetic_v0_1/$c/builder --out $B/haplograph_input/$c
python $REPO/fusion/make_fusion_inputs.py --prolm-run $B/prolm_cv/$c --expression-csv $D/synthtan_prolm_inputs/$c/expression.csv \
  --labels-dir $D/synthtan_prolm_labels/$c --builder-nodes $D/synthetic_v0_1/$c/builder/nodes.csv --out $B/fusion_inputs/$c
FUSION_DIR=$B/fusion_inputs/$c HAPLO_INPUT=$B/haplograph_input/$c WORK=$B/work/$c bash $REPO/fusion/run_genome_folds.sh
```

Per fold, `run_genome_folds.sh` builds the graph from the training patients (step 1), removes edges whose nodes are
not eligible among the training patients (`filter_graph_support.py`), trains node2vec (step 3) and writes the
per-patient means. Every product gets a `.prov` record (its inputs, settings and its own checksum); reruns skip
matching products and stop on any mismatch.

## Running Evo 2 (step 4a) on a GPU server

Evo 2 has not been run for real with these scripts yet; the Evo 2 path was tested only with the fake package.
Treat the first real fold as a test.

1. **Hardware and install.** Evo 2 7B needs a large GPU (step 4a recommends 80 GB). `install_evo2.sh` supports an
   RTX PRO 6000 Blackwell (`rtx`) or an L40S (`l40s`) and needs CUDA 12.8 and Python 3.12. It creates a virtual
   environment `~/evo2-env` (with numpy, pandas and scipy) and ends with an Evo 2 generation test. Use that
   environment's interpreter explicitly below (`PY=$HOME/evo2-env/bin/python`), and make sure `PYTHONPATH` does
   not point at `fusion/tests/mock_evo2`.
2. **Code and data.** Use exactly the code you were given (the git bundle or commit named in the package README),
   and copy the cohort folders `fusion_inputs/<cohort>`, `haplograph_input/<cohort>` and `work/<cohort>`. Use the
   copied `work/` folders: they hold the fold graphs and the node2vec embeddings that
   `genome_embeddings_node2vec.csv` was made from, so Evo 2 starts from the same node embeddings.
3. **Plumbing test with the fake package** (CPU is enough; outputs go to `*_MOCK` files), on scratch copies:
   ```bash
   set -o pipefail
   REPO=/abs/path/GenoPheno_Embedding_Validator; PKG=/abs/path/synthtan_fusion_package; PY=$HOME/evo2-env/bin/python
   T=$(mktemp -d); cp -r $PKG/fusion_inputs/cohort_n100 $T/fi; cp -r $PKG/work/cohort_n100 $T/work
   MOCK_EVO2=1 EVO2=1 FOLDS=0 EVO2_ARGS="--epochs 1" PY=$PY \
     FUSION_DIR=$T/fi HAPLO_INPUT=$PKG/haplograph_input/cohort_n100 WORK=$T/work bash $REPO/fusion/run_genome_folds.sh
   ```
   It should print the mock module path, `evo2: 100 patients x 16 dims` and `done`. Then `rm -rf $T`.
4. **Real test on one fold, one chromosome** (scratch copies again, so nothing real is overwritten):
   ```bash
   T=$(mktemp -d); cp -r $PKG/fusion_inputs/cohort_n100 $T/fi; cp -r $PKG/work/cohort_n100 $T/work
   set -o pipefail
   env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 MOCK_EVO2=0 EVO2=1 FOLDS=0 \
     EVO2_ARGS="--chroms chr105 --epochs 1 --log-every 1" PY=$PY \
     FUSION_DIR=$T/fi HAPLO_INPUT=$PKG/haplograph_input/cohort_n100 WORK=$T/work \
     bash $REPO/fusion/run_genome_folds.sh 2>&1 | tee $T/test.log
   ```
   Check that it prints the real evo2 module path (mock=0), that step 4a logs optimizer steps and a finite
   validation loss, and that `$T/fi/fold0/genome_embeddings_evo2.csv` has 100 rows of finite values (4,096
   columns for Evo 2 7B). One epoch on one chromosome only tests that it runs, not that it learns.
5. **Full run**, one cohort at a time, on the real folders:
   ```bash
   set -o pipefail
   env -u PYTHONPATH MOCK_EVO2=0 EVO2=1 EVO2_ARGS="--epochs 1 --train-blocks 4" PY=$PY \
     FUSION_DIR=$PKG/fusion_inputs/cohort_n200 HAPLO_INPUT=$PKG/haplograph_input/cohort_n200 \
     WORK=$PKG/work/cohort_n200 bash $REPO/fusion/run_genome_folds.sh 2>&1 | tee evo2_cohort_n200.log
   ```
   Per fold this fine-tunes Evo 2 on the fold's training patients only (objective `next`: predict the next node
   along each chromosome; no labels are used), saves `work/<cohort>/graph_fold<k>/evo2_next_node2vec.pt` and
   writes `fusion_inputs/<cohort>/fold<k>/genome_embeddings_evo2.csv` (+ `.json`, `.prov`). With
   `--train-blocks 0` Evo 2 stays frozen and only the new input layers and the prediction head are trained (less
   memory). A rerun skips completed folds; changing `EVO2_ARGS` afterwards stops with an error until the old
   model is removed.
6. **Send back** the `fold*/genome_embeddings_evo2.csv`, `.json` and `.prov` files and the log.

The Evo 2 embeddings have the size of the read-out layer (the model's hidden size, 4,096 for Evo 2 7B): reduce
them within each fold (e.g. PCA fitted on the training rows) before joining them to a 20-column protein table.

## Safety checks built in

- `run_genome_folds.sh` checks that `HAPLO_INPUT` was converted from the same builder files as `FUSION_DIR`
  (sha256), stamps `WORK` with the conversion's checksum and refuses an unstamped non-empty `WORK`, refuses a fold
  graph whose `train_individuals.txt` differs from the fold's training list, checks that `evo2` is the fake
  package exactly when `MOCK_EVO2=1` (the fake one sets `IS_MOCK`), refuses `EVO2_ARGS` that set the options it
  owns, and records the inputs and output checksums of every product (`.prov`).
- `genome_patient_embeddings.py` checks the graph's patients against `folds.csv`, the training-set ids of the node
  embeddings and of the Evo 2 model against the fold's training list, and that the trained Evo 2 blocks loaded.
- `make_fusion_inputs.py` checks that the ProLM run used the same expression and label files, that its folds and
  fold folders belong to that run, that each fold's `patient_embeddings.csv` and training list are the ones that
  fold wrote (checksums in its `metrics.json`), and that no train or test list repeats a patient.

## Limitations

- **Edges between chromosomes are dropped.** Step 1 keeps edges within a chromosome only; 90% of the SynthTAN
  builder edges (including the co-loss motifs, whose regions are on different chromosomes) are not in the graph.
  They reach the models only through which patients carry which nodes.
- **The genome regions are not fully held out.** The SynthTAN builder defined the regions (graph nodes) and the
  list of candidate edges from all patients of a cohort. Step 1's train-only mode recomputes edge weights and lift
  from the training patients, and `filter_graph_support.py` removes edges whose nodes are not eligible among the
  training patients, but candidate edges the builder did not list cannot be added.
- **The builder's chrM ALL region is left out** (a bookkeeping node carried by every patient). Patients without
  any event node get a zero genome vector (7-8 per cohort here); the count is in each `.json`.
- **Patients' own nodes.** A patient's vector averages the nodes it carries; the graph and embeddings come from
  the training patients, but every patient's carrier pattern is used to place it.

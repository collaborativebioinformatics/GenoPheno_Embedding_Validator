#!/usr/bin/env python3
"""Convert the synthetic SynthTAN v0.1 cohorts to the input format of ProLM.

ProLM (https://github.com/Qiu-Shizheng/ProLM) takes three inputs: an expression
CSV (``eid`` plus one column per protein), a labels CSV (``eid,label``) and an
``.npz`` with one embedding vector per protein.

This script writes two folders into ``--out-dir``:

  synthtan_prolm_inputs/   README.md, protein_embeddings.npz,
                           cohort_*/expression.csv
  synthtan_prolm_labels/   README.md, cohort_*/labels_<TASK>.csv

The embeddings are random (fixed seed) because the proteins are synthetic and
have no sequences. Labels are the generator's true flags from the answer key.

Usage:
    python make_prolm_inputs.py --synth-root DIR --answer-key-root DIR \\
        --out-dir DIR [--make-tarballs] [--check]

Needs: Python 3, numpy, pandas (and tar/gzip for --make-tarballs).
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

N_PROTEINS = 7000
EMBEDDING_SEED = 20261110
EMBEDDING_DIM = 64
INPUTS_DIR = "synthtan_prolm_inputs"
LABELS_DIR = "synthtan_prolm_labels"

# Task name -> (column in patient_truth.tsv, value that counts as label 1).
TASKS = {
    "MOT1_A": ("MOT1_A", "TRUE"),
    "MOT1_B": ("MOT1_B", "TRUE"),
    "MOT2_A": ("MOT2_A", "TRUE"),
    "DOM": ("dominant_loss", "TRUE"),
    "SUB2": ("subgroup", "SUB2"),
    "NULL_INDEP_A": ("NULL_INDEP_A", "TRUE"),
}

# ----------------------------------------------------------------------------
# README templates (written verbatim into the two folders)
# ----------------------------------------------------------------------------

INPUTS_README = """# SynthTAN v0.1 protein data in ProLM input format

**Synthetic data. No patient data.** Three synthetic paired tumor/normal cohorts (41, 100 and 200 patients).
For each patient: the log2 tumor-minus-normal (T-N) value of 7,000 fake proteins (`PX00001`-`PX07000`).

## Files

| File | Contents |
|---|---|
| `cohort_n041/expression.csv` (and `n100`, `n200`) | `eid`, then one column per protein: one row per patient, T-N log2 values, no missing values, not rescaled |
| `protein_embeddings.npz` | one random float32 vector of length 64 per protein (fixed seed), the same for all cohorts. The proteins are synthetic and have no sequences, so these stand in for ProteinBERT embeddings |

Labels are in the separate `synthtan_prolm_labels` package: `<cohort>/labels_<TASK>.csv` with `eid`, `label`.

These are ProLM's three inputs: `--expression-csv`, `--labels-csv`, `--protein-embeddings`.
They load with ProLM's `load_prolm_data()` (all 7,000 proteins and all patients kept).

## Three things to know

1. **Training from scratch:** ProLM's `finetune.py` always calls `freeze_backbone`. Without a pretrained
   checkpoint and with the default `--unfreeze-last-n-layers 0`, only the classifier head trains, on a frozen
   random encoder. Make every backbone parameter trainable (`requires_grad = True`) and check that they change.
2. **Chunk size:** the command-line default is `--chunk-size 64` (110 chunks for 7,000 proteins). Choose it
   explicitly; 512 gives 14 chunks. Attention works only within a chunk.
3. **Protein order and coverage:** keep the columns as they are (they are in genome order). ProLM silently drops
   any protein that has no entry in the `.npz`; here every protein has one.

ProLM's `finetune.py` also makes its own train/test split and picks the best epoch on that test set, so use your
own cross-validation instead. Do any scaling (for example z-scoring each protein) within training folds.
"""

LABELS_README = """# Labels for the SynthTAN v0.1 ProLM inputs

**Synthetic data. Answer-key information: keep separate from the inputs if you want to develop blind.**
`<cohort>/labels_<TASK>.csv`: `eid`, `label` (0/1), one row per patient, the same patients as `expression.csv`.
Labels are the generator's true status of planted genomic signals.

| Task | Label = 1 | Protein signal | n = 41 | n = 100 | n = 200 |
|---|---|---|---:|---:|---:|
| MOT1_A | planted loss of region A of co-loss motif 1 | a strong cis tracker (on the region) | 15 | 21 | 57 |
| MOT1_B | planted loss of region B of co-loss motif 1 | a weak trans tracker (elsewhere) | 16 | 21 | 51 |
| MOT2_A | planted loss of region A of co-loss motif 2 | a weak cis tracker | 11 | 23 | 59 |
| DOM | loss of the dominant arm | many dosage proteins on the arm | 18 | 61 | 126 |
| SUB2 | subgroup-only loss within the dominant arm | a strong trans tracker | 2 | 8 | 24 |
| NULL_INDEP_A | planted loss of a null region | none (no protein responds) | 18 | 28 | 52 |

Counts are positives per cohort. Notes:
- **SUB2:** use only at n = 200 (2 and 8 positives in the smaller cohorts).
- **NULL_INDEP_A:** a negative control for proteins. A protein-only model should score near chance
  (AUROC about 0.5).
- **The genome predicts these labels almost perfectly.** They are defined by genomic events, so they test
  whether proteins track them. They don't test whether combining proteins with genome data helps.
"""


# ----------------------------------------------------------------------------
# Reading the SynthTAN files
# ----------------------------------------------------------------------------

def read_table(path, sep):
    """Read a table with every cell kept as text, so values are never rounded."""
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


def find_cohorts(synth_root):
    """Return the sorted names of the ``cohort_*`` folders under the root."""
    cohorts = sorted(p.name for p in Path(synth_root).glob("cohort_*") if p.is_dir())
    if not cohorts:
        sys.exit(f"No cohort_* folders found under {synth_root}")
    return cohorts


def read_patients(synth_root, cohort):
    """Return the patient ids in the order of graph/nodes_patient.csv."""
    nodes = read_table(Path(synth_root) / cohort / "graph" / "nodes_patient.csv", ",")
    return nodes["id"].tolist()


def read_matrix(synth_root, cohort):
    """Read the protein matrix (rows proteins, columns patients) as text."""
    path = Path(synth_root) / cohort / "proteins" / "protein_tn_matrix.tsv.gz"
    return read_table(path, "\t").set_index("protein_id")


def read_truth(answer_key_root, cohort, patients):
    """Read patient_truth.tsv with rows in the patient order of the cohort."""
    truth = read_table(Path(answer_key_root) / cohort / "patient_truth.tsv", "\t")
    truth = truth.set_index("patient")
    assert set(truth.index) == set(patients), f"{cohort}: answer key patients differ"
    return truth.loc[patients]


# ----------------------------------------------------------------------------
# Building the outputs
# ----------------------------------------------------------------------------

def make_expression(matrix, patients):
    """Return the patient x protein table (``eid`` first), values copied as text."""
    assert set(matrix.columns) == set(patients), "matrix and nodes_patient differ"
    expression = matrix[patients].T
    expression.index.name = "eid"
    assert (expression != "").all().all(), "missing expression value"
    assert np.isfinite(expression.astype(float).to_numpy()).all(), "non-finite value"
    return expression


def make_embeddings(panel):
    """Draw one random float32 N(0,1) vector of length 64 per protein (fixed seed)."""
    rng = np.random.default_rng(EMBEDDING_SEED)
    draw = rng.standard_normal((len(panel), EMBEDDING_DIM)).astype(np.float32)
    return {protein: draw[i] for i, protein in enumerate(panel)}


def write_npz(path, arrays):
    """Write an .npz that np.load reads, with fixed zip timestamps and modes.

    np.savez stamps the current time into the zip, so the bytes would change on
    every run. Writing the .npy members by hand makes the file reproducible.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for key, array in arrays.items():
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            with archive.open(info, "w") as handle:
                np.lib.format.write_array(handle, array, allow_pickle=False)


def make_labels(truth):
    """Return one ``eid,label`` table per task: 1 where the truth flag is positive."""
    labels = {}
    for task, (column, positive_value) in TASKS.items():
        flag = (truth[column] == positive_value).astype(int)
        labels[task] = pd.DataFrame({"eid": truth.index, "label": flag.to_numpy()})
    return labels


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------

def build(synth_root, answer_key_root, out_dir):
    """Write the inputs and labels folders into ``out_dir``."""
    inputs_dir, labels_dir = out_dir / INPUTS_DIR, out_dir / LABELS_DIR
    cohorts = find_cohorts(synth_root)

    # The protein panel must be the same in every cohort; embeddings are drawn once.
    panel = read_matrix(synth_root, cohorts[0]).index.tolist()
    inputs_dir.mkdir(parents=True, exist_ok=True)
    write_npz(inputs_dir / "protein_embeddings.npz", make_embeddings(panel))
    (inputs_dir / "README.md").write_text(INPUTS_README)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "README.md").write_text(LABELS_README)

    for cohort in cohorts:
        (inputs_dir / cohort).mkdir(exist_ok=True)
        (labels_dir / cohort).mkdir(exist_ok=True)
        patients = read_patients(synth_root, cohort)
        matrix = read_matrix(synth_root, cohort)
        assert matrix.index.tolist() == panel, f"{cohort}: protein panel differs"

        # Expression: one row per patient, one column per protein.
        make_expression(matrix, patients).to_csv(inputs_dir / cohort / "expression.csv")

        # Labels: one file per task, from the answer key.
        truth = read_truth(answer_key_root, cohort, patients)
        for task, table in make_labels(truth).items():
            table.to_csv(labels_dir / cohort / f"labels_{task}.csv", index=False)
    print(f"Wrote {inputs_dir} and {labels_dir}")


def make_tarball(folder):
    """Write ``<folder>.tar.gz`` next to the folder (members owned by root)."""
    tarball = folder.parent / f"{folder.name}.tar.gz"
    subprocess.run(["tar", "--owner=0", "--group=0", "-C", str(folder.parent),
                    "-czf", str(tarball), folder.name], check=True)
    print(f"Wrote {tarball}")


# ----------------------------------------------------------------------------
# Check
# ----------------------------------------------------------------------------

def check(synth_root, answer_key_root, out_dir):
    """Re-run the assertions on the written files; stop with an error on failure."""
    inputs_dir, labels_dir = out_dir / INPUTS_DIR, out_dir / LABELS_DIR
    cohorts = find_cohorts(synth_root)
    panel = read_matrix(synth_root, cohorts[0]).index.tolist()
    assert len(panel) == N_PROTEINS, f"expected {N_PROTEINS} proteins, found {len(panel)}"

    # The READMEs are the templates above.
    assert (inputs_dir / "README.md").read_text() == INPUTS_README
    assert (labels_dir / "README.md").read_text() == LABELS_README

    # Embeddings: one float32 vector of length 64 for every protein.
    with np.load(inputs_dir / "protein_embeddings.npz", allow_pickle=False) as npz:
        assert set(npz.files) == set(panel), "a protein has no npz key (or extra keys)"
        for key in npz.files:
            assert npz[key].dtype == np.float32 and npz[key].shape == (EMBEDDING_DIM,)
        assert npz.files == panel, "npz keys are not in panel order"

    for cohort in cohorts:
        patients = read_patients(synth_root, cohort)
        assert len(patients) == len(set(patients)), f"{cohort}: duplicate patients"
        assert read_matrix(synth_root, cohort).index.tolist() == panel, f"{cohort}: panel differs"

        # Expression: same proteins, patient order as nodes_patient.csv, no NA.
        written = pd.read_csv(inputs_dir / cohort / "expression.csv", dtype=str, keep_default_na=False)
        assert written.columns.tolist() == ["eid"] + panel, f"{cohort}: expression header"
        assert written["eid"].tolist() == patients, f"{cohort}: patient order"
        assert (written != "").all().all(), f"{cohort}: missing value"
        assert not written.isin(["NA", "nan", "NaN"]).any().any(), f"{cohort}: NA value"
        source = make_expression(read_matrix(synth_root, cohort), patients).reset_index()
        assert written.equals(source), f"{cohort}: expression differs from the source matrix"

        # Labels: equal to the answer-key flags, in the same patient order.
        truth = read_truth(answer_key_root, cohort, patients)
        for task, expected in make_labels(truth).items():
            path = labels_dir / cohort / f"labels_{task}.csv"
            labels = pd.read_csv(path, dtype={"eid": str})
            assert labels.equals(expected), f"{cohort}/{task}: labels differ from the answer key"
    print("All checks passed.")


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

def main():
    """Parse the command line, build the folders, optionally tar and check them."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synth-root", required=True, help="folder with cohort_*/ (SynthTAN v0.1)")
    parser.add_argument("--answer-key-root", required=True, help="folder with cohort_*/patient_truth.tsv")
    parser.add_argument("--out-dir", required=True, help="folder to write the two output folders into")
    parser.add_argument("--make-tarballs", action="store_true", help="also write the two .tar.gz files")
    parser.add_argument("--check", action="store_true", help="re-run the assertions on the written files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    build(args.synth_root, args.answer_key_root, out_dir)
    if args.make_tarballs:
        make_tarball(out_dir / INPUTS_DIR)
        make_tarball(out_dir / LABELS_DIR)
    if args.check:
        check(args.synth_root, args.answer_key_root, out_dir)


if __name__ == "__main__":
    main()

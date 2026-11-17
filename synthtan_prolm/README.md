# synthtan_prolm: SynthTAN to ProLM input format

`make_prolm_inputs.py` converts the synthetic SynthTAN v0.1 cohorts (T-N values of 7,000 fake proteins, three
cohorts) to the three inputs of [ProLM](https://github.com/Qiu-Shizheng/ProLM): an expression CSV, a labels CSV and
a protein-embedding `.npz`. The embeddings are random (fixed seed) because the proteins have no sequences. No
patient data.

## Usage

Needs Python 3 with `numpy` and `pandas` (and `tar`/`gzip` for `--make-tarballs`).

```bash
python make_prolm_inputs.py \
  --synth-root path/to/synthetic_v0_1 \
  --answer-key-root path/to/synthetic_v0_1_answer_key \
  --out-dir path/to/output \
  --make-tarballs --check      # both optional
```

Inputs, per cohort: `<synth-root>/cohort_*/proteins/protein_tn_matrix.tsv.gz`,
`<synth-root>/cohort_*/graph/nodes_patient.csv` and `<answer-key-root>/cohort_*/patient_truth.tsv`.

## Output

| Folder | Contents |
|---|---|
| `synthtan_prolm_inputs/` | `README.md`, `protein_embeddings.npz`, `cohort_*/expression.csv` |
| `synthtan_prolm_labels/` | `README.md`, `cohort_*/labels_<TASK>.csv` for `MOT1_A`, `MOT1_B`, `MOT2_A`, `DOM`, `SUB2`, `NULL_INDEP_A` |

The labels are kept in a separate folder so the inputs can be used blind. Both folders' READMEs are embedded in the
script. `--make-tarballs` also writes `<folder>.tar.gz` next to each folder. Rerunning gives byte-identical files
(the `.tar.gz` files differ only in timestamps).

`--check` reruns assertions on the written files: 7,000 proteins, the same panel in every cohort, patient order as in
`nodes_patient.csv`, no missing values, every protein has an `.npz` key, and labels equal the answer-key flags.

"""Build the per-patient fusion inputs for one SynthTAN cohort, in the layout shared by the genome and protein sides.

LAYOUT  (--out, one folder per cohort; <run> = the ProLM run's name, e.g. cohort_n200)
  README.md, manifest.json            what is here, sources and checksums
  folds.csv                           eid, fold: the ProLM run's folds, used by every arm
  labels/labels_<TASK>.csv            eid, label (copied from --labels-dir)
  genome_raw_columns.txt              node id of each column g0, g1 ... of genome_embeddings_raw.csv
  fold<k>/<run>_fold<k>_train.txt     training patients (one id per line): --train-individuals of the genome steps
  fold<k>/<run>_fold<k>_test.txt      test patients
  fold<k>/genome_embeddings_raw.csv   eid, g0 ...: the patient's 0/1 genome state per graph node (builder
                                      nodes.csv, without the builder's chrM ALL bookkeeping region), no model;
                                      the same in every fold (nothing is fitted)
  fold<k>/protein_embeddings.csv      eid, p0 ...: the ProLM embedding of that fold (shared across labels)
  fold<k>/protein_<TASK>.csv          eid, p0 ... p<K-1>: the simple protein vector for one label: the values of
                                      the K proteins with the largest |Welch t| between label 1 and 0 among the
                                      fold's TRAINING patients (raw T-N values; scale them within the fold)
  fold<k>/protein_<TASK>_columns.txt  protein id of each column p0 ... of protein_<TASK>.csv
  Genome embeddings from the genome pipeline are added later as fold<k>/genome_embeddings_<method>.csv
  (fusion/genome_patient_embeddings.py; e.g. node2vec, evo2).

Every table has one row per patient of the cohort (training and test). For fold k and a label: join the genome
and protein tables on eid, fit on the rows of <run>_fold<k>_train.txt, score on <run>_fold<k>_test.txt. Never mix
folds or cohorts (the cohorts reuse the ids S001 ...).

USAGE
  python fusion/make_fusion_inputs.py --prolm-run runs/prolm_cv/cohort_n200 \\
      --expression-csv synthtan_prolm_inputs/cohort_n200/expression.csv \\
      --labels-dir synthtan_prolm_labels/cohort_n200 \\
      --builder-nodes synthetic_v0_1/cohort_n200/builder/nodes.csv --out fusion_inputs/cohort_n200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TASKS = "MOT1_A,MOT1_B,MOT2_A,DOM,SUB2,NULL_INDEP_A"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_ids(path) -> list[str]:
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]


def welch_abs_t(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Welch t| per column of X for groups y == 1 vs y == 0. A column constant within both groups but different
    between them separates them perfectly (+inf, ranked first); a column constant overall is NaN (ranked last)."""
    a, b = X[y == 1], X[y == 0]
    if len(a) < 2 or len(b) < 2:
        raise SystemExit("a label has fewer than 2 training patients in a class; cannot select proteins")
    va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (a.mean(0) - b.mean(0)) / np.sqrt(va + vb)
    t = np.abs(t)
    t[np.isnan(t)] = -np.inf
    return t


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prolm-run", required=True, help="cohort output folder of proteomics_emb/cv_embeddings.py")
    ap.add_argument("--expression-csv", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--builder-nodes", required=True, help="SynthTAN builder/nodes.csv of the same cohort")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--top-k", type=int, default=20, help="proteins per label in the simple protein vector")
    args = ap.parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks or len(set(tasks)) != len(tasks):
        raise SystemExit("--tasks must be a non-empty list without duplicates")
    prolm, out = Path(args.prolm_run), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new folder")

    # ProLM run: same cohort and labels?
    cfg = json.loads((prolm / "config.json").read_text())
    fp = cfg["fingerprint"]
    if fp["sha256_expression"] != sha256(args.expression_csv):
        raise SystemExit(f"{prolm} was run on another expression file than {args.expression_csv} (other cohort?)")
    for t in tasks:
        if fp["sha256_labels"].get(t) != sha256(Path(args.labels_dir) / f"labels_{t}.csv"):
            raise SystemExit(f"{prolm} was run with other labels for {t}")
    run, n_folds = fp["run_name"], cfg["n_folds"]
    folds = pd.read_csv(prolm / "folds.csv", dtype={"eid": str}).set_index("eid")["fold"]
    if folds.index.duplicated().any() or hashlib.sha256(folds.sort_index().to_csv().encode()).hexdigest() != \
            fp["sha256_fold_assignment"]:
        raise SystemExit(f"{prolm}/folds.csv does not match the run's fold assignment")

    expr = pd.read_csv(args.expression_csv, dtype={"eid": str}).set_index("eid")
    labels = pd.DataFrame({t: pd.read_csv(Path(args.labels_dir) / f"labels_{t}.csv", dtype={"eid": str})
                           .set_index("eid")["label"] for t in tasks})
    nodes = pd.read_csv(args.builder_nodes, dtype={"id": str})
    nodes = nodes[nodes["high_dim_edge"] != "chrM_1-2"]                     # builder bookkeeping region
    genome = nodes.set_index("id").drop(columns="high_dim_edge").T          # patients x nodes
    genome.index = genome.index.astype(str)
    ids = sorted(folds.index)
    for name, idx in [("expression", expr.index), ("labels", labels.index), ("builder nodes", genome.index)]:
        if sorted(idx) != ids or pd.Index(idx).duplicated().any():
            raise SystemExit(f"{name} does not list exactly the patients of {prolm}/folds.csv")
    if not genome.isin([0, 1]).all().all():
        raise SystemExit(f"{args.builder_nodes}: patient columns must be 0/1")
    if not 1 <= args.top_k <= expr.shape[1]:
        raise SystemExit(f"--top-k must be between 1 and {expr.shape[1]}")

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(prolm / "folds.csv", out / "folds.csv")
    (out / "labels").mkdir()
    for t in tasks:
        shutil.copy(Path(args.labels_dir) / f"labels_{t}.csv", out / "labels" / f"labels_{t}.csv")
    (out / "genome_raw_columns.txt").write_text("\n".join(genome.columns) + "\n")
    g_raw = genome.loc[ids].astype(np.int8)
    g_raw.columns = [f"g{j}" for j in range(g_raw.shape[1])]
    g_raw.index.name = "eid"

    selected = {}
    for k in range(n_folds):
        src, dst = prolm / f"fold{k}", out / f"fold{k}"
        m = json.loads((src / "metrics.json").read_text())
        if m.get("fold") != k or m.get("fingerprint") != fp:
            raise SystemExit(f"{src} is not fold {k} of this ProLM run (moved or stale folder?)")
        for name, key in (("patient_embeddings.csv", "sha256_patient_embeddings"), (f"{run}_fold{k}_train.txt", "sha256_train_list")):
            if m.get(key) != sha256(src / name):
                raise SystemExit(f"{src}/{name} is not the file fold {k} wrote (checksum differs)")
        dst.mkdir()
        train_ids, test_ids = read_ids(src / f"{run}_fold{k}_train.txt"), read_ids(src / f"{run}_fold{k}_test.txt")
        if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
            raise SystemExit(f"{src}: a patient is listed twice in the train or test list")
        if set(test_ids) != set(folds.index[folds == k]) or set(train_ids) != set(folds.index[folds != k]):
            raise SystemExit(f"{src}: train/test lists do not match folds.csv")
        for f in (f"{run}_fold{k}_train.txt", f"{run}_fold{k}_test.txt"):
            shutil.copy(src / f, dst / f)
        g_raw.to_csv(dst / "genome_embeddings_raw.csv")
        pe = pd.read_csv(src / "patient_embeddings.csv", dtype={"eid": str}).set_index("eid")
        if sorted(pe.index) != ids:
            raise SystemExit(f"{src}/patient_embeddings.csv does not list every patient once")
        pe = pe.loc[ids]
        pe.columns = [f"p{j}" for j in range(pe.shape[1])]
        pe.to_csv(dst / "protein_embeddings.csv")
        X_tr, X_all = expr.loc[train_ids].to_numpy(np.float64), expr.loc[ids]
        for t in tasks:
            score = welch_abs_t(X_tr, labels.loc[train_ids, t].to_numpy())
            top = np.lexsort((np.arange(len(score)), -score))[:args.top_k]     # ties: protein order
            cols = list(expr.columns[top])
            v = X_all[cols].copy()
            v.columns = [f"p{j}" for j in range(len(cols))]
            v.to_csv(dst / f"protein_{t}.csv")
            (dst / f"protein_{t}_columns.txt").write_text("\n".join(cols) + "\n")
            selected.setdefault(t, []).append(cols)

    stability = {t: {"proteins_in_every_fold": sorted(set.intersection(*map(set, c))),
                     "distinct_proteins": len(set().union(*c))} for t, c in selected.items()}
    manifest = {"run_name": run, "n_patients": len(ids), "n_folds": n_folds, "tasks": tasks, "top_k": args.top_k,
                "prolm_run": str(prolm.resolve()), "prolm_fingerprint": fp,
                "expression_csv": str(Path(args.expression_csv).resolve()), "sha256_expression": sha256(args.expression_csv),
                "builder_nodes": str(Path(args.builder_nodes).resolve()), "sha256_builder_nodes": sha256(args.builder_nodes),
                "genome_raw_dim": int(g_raw.shape[1]), "protein_embedding_dim": int(pe.shape[1]),
                "selection_stability": stability}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "README.md").write_text(__doc__.split("USAGE")[0].replace("Build the per-patient", "Per-patient", 1))
    print(f"{run}: {len(ids)} patients, {n_folds} folds, genome raw {g_raw.shape[1]} columns, "
          f"ProLM {pe.shape[1]} dims, simple protein vectors of {args.top_k} for {len(tasks)} labels -> {out}")


if __name__ == "__main__":
    main()

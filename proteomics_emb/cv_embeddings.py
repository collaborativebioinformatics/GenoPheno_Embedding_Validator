"""Fold-internal ProLM patient embeddings for late fusion with the Evo 2 genome embeddings.

Trains ProLM from scratch inside each cross-validation fold and saves, per fold, one protein embedding per
patient plus the fold's train/test individual lists in the format the genome steps of this repository read.
This replaces the single 50/50 split of `finetune_on_synthetic_data.ipynb`, which selected its epoch on the test
half and embedded every patient with that one model.

WHAT IS TRAINED
  One ProLM model per fold on all --tasks at once (multi-label: one sigmoid output per task, mean BCE loss), from
  random weights with every parameter trainable. The fold's training patients are split into a fit part and a
  validation part (--val-frac, stratified on --stratify-task); the epoch with the lowest validation loss is kept
  (early stopping, --patience). The test fold is never used for training or model selection. The mean BCE weights
  the tasks equally, but a rare task such as SUB2 has few positive terms, so the selected epoch need not suit it.

INPUT
  --expression-csv     eid + one numeric column per protein (ProLM format, e.g.
                       synthtan_prolm_inputs/<cohort>/expression.csv). Must be complete and finite: ProLM's loader
                       would fill gaps with the mean over ALL patients, test patients included.
  --labels-dir         folder with labels_<TASK>.csv files (eid, label 0/1), one per task in --tasks
  --protein-embeddings .npz, one finite vector per protein name (e.g. synthtan_prolm_inputs/protein_embeddings.npz)
  --folds-csv          (optional) CSV eid, fold (integers 0..k-1) to reuse folds made elsewhere; otherwise
                       stratified folds are made (--n-folds, --seed, on the eids in sorted order) and written out

OUTPUT  (in --output-dir; <run> = --run-name, default the output folder's name, e.g. cohort_n200)
  folds.csv                        eid, fold: the folds used; give the same file to every arm of the comparison
  config.json                      settings, tasks, input checksums and package versions
  protein_order.txt                protein order seen by the model
  summary.csv                      fold, task, class counts (fit / val / test), test_auroc, test_ap, best_epoch;
                                   rebuilt after each fold from the completed fold0 ... fold<k-1> of this run
  fold<k>/<run>_fold<k>_train.txt  the fold's training patients (fit + validation), one id per line, no header:
                                   the format of --train-individuals in 1_build_graph.py, 2_ancestry_labels.py,
                                   3_node_embeddings.py, 4a_finetune_evo2_embeddings.py and 5_validate_evo2.py.
                                   The file name is unique per run and fold because 1_build_graph.py names its
                                   default output folder after it (and reuses an existing folder of that name).
  fold<k>/<run>_fold<k>_test.txt   the fold's test patients (same format; --test-individuals of 5_validate_evo2.py)
  fold<k>/fold_sets.csv            eid, set (fit / val / test). Lists ALL patients: never pass it as
                                   --train-individuals (the genome steps would read every id in it as training)
  fold<k>/patient_embeddings.csv   eid, e0 ... e<hidden_size-1>: every patient embedded by the fold's model, in
                                   evaluation mode. Join to the same fold's genome embedding on eid (= the
                                   individual names of the genome graph, e.g. S001)
  fold<k>/test_predictions.csv     eid, one predicted probability per task, for the test patients
  fold<k>/metrics.json             per-epoch fit/validation loss, the selected epoch, class counts and metrics per
                                   task (undefined metrics are null)
  fold<k>/classifier.pt, backbone.pt   state_dicts of the selected model

USING THE EMBEDDINGS FOR LATE FUSION
  - For fold k, train the downstream model on the rows of <run>_fold<k>_train.txt and evaluate on the rows of
    <run>_fold<k>_test.txt, with the genome embedding built from the same training file. Never pool embeddings
    across folds (each fold's model has its own vector space) or cohorts (eids repeat across cohorts).
  - Compare genome only, protein only and genome + protein with the SAME downstream learner, training rows and
    tuning budget. test_predictions.csv (ProLM's own head, trained on the fit part only) is a separate baseline.
  - The training patients' embeddings come from a model trained on their labels (fit part) or selected on them
    (validation part), so a downstream model may see them as more separable than the test patients' (stacking
    optimism). Compare arms on the test folds only.

EXAMPLE
  python proteomics_emb/cv_embeddings.py \\
    --expression-csv synthtan_prolm_inputs/cohort_n200/expression.csv \\
    --labels-dir synthtan_prolm_labels/cohort_n200 \\
    --protein-embeddings synthtan_prolm_inputs/protein_embeddings.npz \\
    --output-dir prolm_cv/cohort_n200 --device cuda

OUTPUT FOLDER RULES  The folder must be new, empty or a folder made by this script with the same inputs, settings,
  device, package versions and script/ProLM code (fingerprint in config.json). Completed folds are kept and skipped
  on a rerun (resume); --overwrite recomputes them. Anything else in the folder makes the script stop before it
  changes anything. A fold is written to a temporary folder and moved into place only when it is complete. Keep
  other files (e.g. genome embeddings) outside fold<k>/. Do not run two processes on one folder at the same time.

REPRODUCIBILITY  Splits, initialization and shuffling are seeded (--seed, + fold number). GPU kernels are not
  guaranteed to be bitwise deterministic, so reruns on a GPU can differ slightly.

REQUIRES  numpy, pandas, scikit-learn, torch and ProLM (https://github.com/Qiu-Shizheng/ProLM) installed as the
          `prolm` package (`pip install -e .` in a ProLM clone; it needs transformers < 5).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import prolm
from prolm.data import load_prolm_data
from prolm.model import ProLMBackbone, ProLMClassifier

DEFAULT_TASKS = "MOT1_A,MOT1_B,MOT2_A,DOM,SUB2,NULL_INDEP_A"
# settings that change the results; a rerun into the same output folder must match them
RUN_KEYS = ("expression_csv", "labels_dir", "protein_embeddings", "folds_csv", "n_folds", "stratify_task",
            "val_frac", "hidden_size", "num_layers", "chunk_size", "batch_size", "max_epochs", "patience",
            "learning_rate", "weight_decay", "seed", "run_name", "device")
TOOL = "proteomics_emb/cv_embeddings.py"
BAD_ID = re.compile(r"[\s#,]")       # the genome steps' text readers split on whitespace and drop text after '#'
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
ROOT_FILES = {"folds.csv", "config.json", "protein_order.txt", "summary.csv",
              "config.json.tmp", "summary.csv.tmp", "folds.csv.tmp", "protein_order.txt.tmp"}
FOLD_FILES = ("backbone.pt", "classifier.pt", "metrics.json", "patient_embeddings.csv", "test_predictions.csv",
              "fold_sets.csv")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--expression-csv", required=True)
    ap.add_argument("--labels-dir", required=True, help="folder with labels_<TASK>.csv (eid, label)")
    ap.add_argument("--protein-embeddings", required=True, help=".npz with one vector per protein")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--run-name", help="prefix of the id files (default: the output folder's name)")
    ap.add_argument("--tasks", default=DEFAULT_TASKS, help=f"comma-separated (default {DEFAULT_TASKS})")
    ap.add_argument("--folds-csv", help="reuse folds: CSV with eid, fold")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--stratify-task", help="task to stratify folds and the validation split on (default: first task)")
    ap.add_argument("--val-frac", type=float, default=0.2, help="share of each fold's training patients for validation")
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=64, help="proteins per chunk (7,000 / 64 = 110 chunks)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15, help="epochs without a better validation loss before stopping")
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=314159)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--folds", help="run only these folds, e.g. 0,1 (default: all); others can be run later")
    ap.add_argument("--overwrite", action="store_true", help="recompute completed folds; replace a run with another fingerprint (only in a folder this script made)")
    args = ap.parse_args(argv)
    args.run_name = args.run_name or Path(args.output_dir).resolve().name
    if not SAFE_NAME.fullmatch(args.run_name):
        ap.error("--run-name must be a plain name: letters, digits, '.', '_' or '-' (no path)")
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1:
        ap.error("--max-epochs, --patience and --batch-size must be >= 1")
    if not 0 < args.val_frac < 1:
        ap.error("--val-frac must be between 0 and 1")
    if args.n_folds < 2:
        ap.error("--n-folds must be >= 2")
    return args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(x):
    """NaN / inf -> None, numpy scalars -> Python, so the files are strict JSON."""
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(clean_json(obj), indent=2, allow_nan=False))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def check_ids(ids, what: str) -> None:
    ids = pd.Series(ids)
    if ids.isna().any():
        raise SystemExit(f"{what}: missing eids")
    ids = ids.astype(str)
    if ids.duplicated().any():
        raise SystemExit(f"{what}: duplicated eids, e.g. {ids[ids.duplicated()].iloc[0]}")
    bad = [i for i in ids if not i or BAD_ID.search(i)]
    if bad:
        raise SystemExit(f"{what}: eids must be non-empty without spaces, commas or '#', e.g. {bad[0]!r}")


def read_labels(labels_dir: Path, tasks: list[str]) -> pd.DataFrame:
    """eid x task table of 0/1 labels; every task file must list the same patients exactly once."""
    cols = {}
    for task in tasks:
        f = labels_dir / f"labels_{task}.csv"
        if not f.exists():
            raise SystemExit(f"missing labels file {f}")
        t = pd.read_csv(f, dtype={"eid": str})
        if list(t.columns) != ["eid", "label"]:
            raise SystemExit(f"{f}: expected columns eid,label, got {list(t.columns)}")
        check_ids(t["eid"], str(f))
        if not t["label"].isin([0, 1]).all():
            raise SystemExit(f"{f}: labels must be 0/1")
        cols[task] = t.set_index("eid")["label"].astype(int)
    eids = [set(s.index) for s in cols.values()]
    if any(e != eids[0] for e in eids):
        raise SystemExit("the label files do not list the same patients")
    return pd.DataFrame(cols)


def load_inputs(args, tasks):
    labels = read_labels(Path(args.labels_dir), tasks)
    raw = pd.read_csv(args.expression_csv, dtype={"eid": str})
    if "eid" not in raw.columns:
        raise SystemExit(f"{args.expression_csv}: no eid column")
    check_ids(raw["eid"], args.expression_csv)
    values = raw.drop(columns="eid")
    non_numeric = [c for c in values.columns if not pd.api.types.is_numeric_dtype(values[c])]
    if non_numeric:
        raise SystemExit(f"{args.expression_csv}: non-numeric values in {len(non_numeric)} columns, e.g. {non_numeric[0]}")
    with np.errstate(over="ignore"):
        as32 = values.to_numpy(dtype=np.float64).astype(np.float32)   # the model's dtype
    if not np.isfinite(as32).all():
        # load_prolm_data would fill gaps with the mean over ALL patients, test patients included
        raise SystemExit(f"{args.expression_csv}: missing, infinite or out-of-range values; impute or fix them first")
    with np.load(args.protein_embeddings, allow_pickle=False) as z:
        shapes = {z[k].shape for k in z.files}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1 or not all(np.issubdtype(z[k].dtype, np.number) for k in z.files):
            raise SystemExit(f"{args.protein_embeddings}: expected one numeric 1-D vector of the same length per protein")
        with np.errstate(over="ignore"):
            bad = [k for k in z.files if not np.isfinite(z[k].astype(np.float32)).all()]
    if bad:
        raise SystemExit(f"{args.protein_embeddings}: non-finite vectors (as float32), e.g. {bad[0]}")
    if set(raw["eid"]) != set(labels.index):
        only_e = sorted(set(raw["eid"]) - set(labels.index))[:5]
        only_l = sorted(set(labels.index) - set(raw["eid"]))[:5]
        raise SystemExit(f"expression and labels list different patients (e.g. {only_e} / {only_l})")
    # ProLM's loader: protein order, protein features and the attention bias matrix. Its labels are not used.
    data = load_prolm_data(args.expression_csv, Path(args.labels_dir) / f"labels_{tasks[0]}.csv",
                           args.protein_embeddings, None, eid_col="eid", label_col="label")
    if len(data.protein_names) != values.shape[1]:
        raise SystemExit(f"only {len(data.protein_names)} of {values.shape[1]} proteins have an embedding")
    labels = labels.loc[data.expression.index]           # align by eid, never by row position
    assert list(labels.index) == list(data.expression.index)
    return data, labels


def make_folds(args, labels: pd.DataFrame, strat_task: str) -> pd.Series:
    if args.folds_csv:
        t = pd.read_csv(args.folds_csv, dtype={"eid": str})
        if list(t.columns[:2]) != ["eid", "fold"]:
            raise SystemExit(f"{args.folds_csv}: expected columns eid, fold")
        check_ids(t["eid"], args.folds_csv)
        if set(t["eid"]) != set(labels.index):
            raise SystemExit(f"{args.folds_csv} must list exactly the patients of the expression file")
        v = pd.to_numeric(t["fold"], errors="coerce")
        if v.isna().any() or (v != v.round()).any():
            raise SystemExit(f"{args.folds_csv}: fold values must be integers")
        f = pd.Series(v.astype(int).to_numpy(), index=t["eid"], name="fold")
        k = sorted(f.unique())
        if k != list(range(len(k))) or len(k) < 2:
            raise SystemExit(f"{args.folds_csv}: folds must be 0..k-1 with k >= 2, got {k}")
        return f.loc[labels.index]
    order = sorted(labels.index)                        # fold assignment does not depend on the file's row order
    y = labels.loc[order, strat_task].to_numpy()
    if np.bincount(y, minlength=2).min() < args.n_folds:
        raise SystemExit(f"{strat_task} has fewer than {args.n_folds} patients in a class; choose another --stratify-task")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold = np.empty(len(y), dtype=int)
    for k, (_, test) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold[test] = k
    return pd.Series(fold, index=order, name="fold").loc[labels.index]


def fold_file_names(run_name: str, k: int) -> set[str]:
    return set(FOLD_FILES) | {f"{run_name}_fold{k}_train.txt", f"{run_name}_fold{k}_test.txt"}


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def prepare_output(args, out_dir: Path, config: dict, all_folds: list[int]) -> bool:
    """Check the folder before changing anything: new, empty, or this script's run.

    Returns True when config.json must be (re)written. Every entry, and every file inside the fold folders, must be
    a name this script writes and not a symbolic link; otherwise nothing is touched."""
    if out_dir.is_symlink() or (out_dir.exists() and not out_dir.is_dir()):
        raise SystemExit(f"{out_dir} is not a plain folder")
    entries = list(out_dir.iterdir()) if out_dir.exists() else []
    if not entries:
        out_dir.mkdir(parents=True, exist_ok=True)
        return True
    cfg, cfg_tmp = out_dir / "config.json", out_dir / "config.json.tmp"
    if not cfg.exists() and cfg_tmp.is_file() and not cfg_tmp.is_symlink():
        cfg = cfg_tmp                                  # a first run stopped while publishing its config
    try:
        old = json.loads(cfg.read_text())
    except (OSError, ValueError):
        old = None
    if not isinstance(old, dict) or old.get("tool") != TOOL or not isinstance(old.get("fingerprint"), dict):
        raise SystemExit(f"{out_dir} is not empty and was not made by {TOOL}; use another --output-dir")
    old_fp = old["fingerprint"]
    n_old = old.get("n_folds") if isinstance(old.get("n_folds"), int) else 0
    ks = range(n_old)                                  # ownership comes from the OLD run's manifest only
    run_names = {old_fp.get("run_name")}
    allowed = {f"fold{k}": k for k in ks} | {f".fold{k}.partial": k for k in ks} | {f".fold{k}.old": k for k in ks}
    problems = []
    for e in entries:
        if e.is_symlink():
            problems.append(f"{e.name} (link)")
        elif e.name in ROOT_FILES:
            if not e.is_file():
                problems.append(e.name)
        elif e.name in allowed and e.is_dir():
            ok = set().union(*(fold_file_names(r, allowed[e.name]) for r in run_names if r))
            ok |= {n + ".tmp" for n in ok}
            problems += [f"{e.name}/{c.name}" for c in e.iterdir()
                         if c.is_symlink() or not c.is_file() or c.name not in ok]
        else:
            problems.append(e.name)
    if problems:
        raise SystemExit(f"{out_dir} holds entries this script did not write ({', '.join(sorted(problems)[:5])}); "
                         "move them or use another --output-dir. Nothing was changed.")
    diff = sorted(k for k in set(config["fingerprint"]) | set(old_fp)
                  if old_fp.get(k) != config["fingerprint"].get(k))
    if diff and not args.overwrite:
        raise SystemExit(f"{out_dir} holds a run with other settings, inputs or code ({', '.join(diff)}); "
                         "use another --output-dir, or --overwrite to replace it")
    if diff:                                           # --overwrite: every entry is ours; replace the old run
        markers = [e for e in entries if e.name in ("config.json", "config.json.tmp")]
        for e in [e for e in entries if e not in markers] + markers:     # the ownership marker goes last
            shutil.rmtree(e) if e.is_dir() else e.unlink()
        return True
    if cfg == cfg_tmp:
        cfg_tmp.replace(out_dir / "config.json")
    for k in ks:                                       # same run: finish an interrupted fold replacement
        final, old_dir = out_dir / f"fold{k}", out_dir / f".fold{k}.old"
        if not final.exists() and old_dir.is_dir():
            old_dir.rename(final)
    for e in out_dir.iterdir():                        # then clear leftovers of interrupted steps
        if e.name.startswith(".fold") or e.name.endswith(".tmp"):
            shutil.rmtree(e) if e.is_dir() else e.unlink()
    return False


def fold_done(out_dir: Path, k: int, fingerprint: dict) -> bool:
    """A fold counts as complete only with all its files present and its fingerprint matching this run."""
    fdir = out_dir / f"fold{k}"
    names = fold_file_names(fingerprint["run_name"], k)
    if not all((fdir / n).is_file() and (fdir / n).stat().st_size > 0 for n in names):
        return False
    try:
        m = json.loads((fdir / "metrics.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(m, dict) and m.get("fold") == k and m.get("fingerprint") == fingerprint


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(args, data, n_tasks: int, device) -> ProLMClassifier:
    """ProLM from scratch with every parameter trainable (ProLM's own build_model freezes the backbone)."""
    backbone = ProLMBackbone(
        feature_dim=int(data.features_tensor.shape[1]),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_proteins=len(data.protein_names),
        features_tensor=data.features_tensor,
        bias_matrix=data.bias_matrix,
        chunk_size=args.chunk_size,
    )
    model = ProLMClassifier(backbone, num_classes=n_tasks).to(device)
    for p in model.parameters():
        p.requires_grad = True
    return model


@torch.no_grad()
def predict(model, X: torch.Tensor, device, batch_size: int):
    """(logits, backbone embeddings) for the rows of X, in evaluation mode (no dropout)."""
    was_training = model.training
    model.eval()
    logits, emb = [], []
    for s in range(0, len(X), batch_size):
        e = model.backbone(X[s:s + batch_size].to(device))
        emb.append(e.cpu())
        logits.append(model.classifier(e).cpu())
    model.train(was_training)
    return torch.cat(logits), torch.cat(emb)


def task_metrics(y: np.ndarray, p: np.ndarray, tasks: list[str]) -> dict:
    out = {}
    for j, task in enumerate(tasks):
        yj = y[:, j]
        both = len(np.unique(yj)) == 2
        out[task] = {
            "n": int(len(yj)), "n_pos": int(yj.sum()),
            "auroc": float(roc_auc_score(yj, p[:, j])) if both else float("nan"),
            "ap": float(average_precision_score(yj, p[:, j])) if both else float("nan"),
        }
    return out


def inner_split(args, k, train, y_strat):
    """Fit / validation split of a fold's training patients, stratified when every class allows it."""
    n_val = int(math.ceil(args.val_frac * len(train)))
    if n_val < 1 or len(train) - n_val < 1:
        raise SystemExit(f"fold {k}: {len(train)} training patients cannot be split with --val-frac {args.val_frac}")
    counts = np.bincount(y_strat, minlength=2)
    stratified = counts.min() >= 2 and min(n_val, len(train) - n_val) >= 2
    try:
        fit, val = train_test_split(train, test_size=n_val, random_state=args.seed + k,
                                    stratify=y_strat if stratified else None)
    except ValueError:
        fit, val = train_test_split(train, test_size=n_val, random_state=args.seed + k)
        stratified = False
    return fit, val, stratified


def plan_folds(args, labels, folds, strat_task, run):
    """All splits of the requested folds, checked before anything is trained or written."""
    plan = {}
    for k in run:
        test = np.where(folds.to_numpy() == k)[0]
        train = np.where(folds.to_numpy() != k)[0]
        if len(test) == 0 or len(train) == 0:
            raise SystemExit(f"fold {k} has no test or no training patients")
        fit, val, stratified = inner_split(args, k, train, labels[strat_task].to_numpy()[train])
        y_val = labels[strat_task].to_numpy()[val]
        if not stratified or len(np.unique(y_val)) < 2:
            print(f"warning: fold {k}: validation part not stratified or single-class for {strat_task}")
        plan[k] = (fit, val, test, stratified)
    return plan


def run_fold(args, k: int, split, data, labels, tasks, device, out_dir: Path, fingerprint: dict) -> None:
    set_seed(args.seed + k)
    X_all = torch.tensor(data.expression.to_numpy(), dtype=torch.float32)
    Y_all = torch.tensor(labels.to_numpy(), dtype=torch.float32)
    Y_np = labels.to_numpy()
    eids = np.asarray(labels.index)
    fit, val, test, stratified = split
    train = np.sort(np.concatenate([fit, val]))

    fdir = out_dir / f".fold{k}.partial"          # moved to fold<k> only when complete
    shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir(parents=True)
    stem = f"{args.run_name}_fold{k}"
    (fdir / f"{stem}_train.txt").write_text("\n".join(sorted(eids[train])) + "\n")
    (fdir / f"{stem}_test.txt").write_text("\n".join(sorted(eids[test])) + "\n")
    sets = pd.Series("fit", index=eids)
    sets.iloc[val] = "val"
    sets.iloc[test] = "test"
    pd.DataFrame({"eid": eids, "set": sets.to_numpy()}).to_csv(fdir / "fold_sets.csv", index=False)

    model = build_model(args, data, len(tasks), device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(args.seed + k)
    loader = DataLoader(TensorDataset(X_all[fit], Y_all[fit]), batch_size=args.batch_size, shuffle=True, generator=gen)

    history, best_loss, best_state, best_epoch, since_best = [], float("inf"), None, 0, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for x, y in loader:                      # x and y come from the same shuffled rows
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_logits, _ = predict(model, X_all[val], device, args.batch_size)
        val_loss = float(loss_fn(val_logits, Y_all[val]).item())
        history.append({"epoch": epoch, "fit_loss": float(np.mean(losses)), "val_loss": val_loss})
        if math.isfinite(val_loss) and val_loss < best_loss - 1e-6:
            best_loss, best_state, best_epoch, since_best = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            since_best += 1
            if since_best >= args.patience:
                break
    if best_state is None:
        raise SystemExit(f"fold {k}: no finite validation loss; check the inputs and the learning rate")
    model.load_state_dict(best_state)

    logits, emb = predict(model, X_all, device, args.batch_size)
    prob = torch.sigmoid(logits).numpy()
    pd.DataFrame(emb.numpy(), index=pd.Index(eids, name="eid"),
                 columns=[f"e{j}" for j in range(emb.shape[1])]).to_csv(fdir / "patient_embeddings.csv")
    pd.DataFrame(prob[test], index=pd.Index(eids[test], name="eid"), columns=tasks).to_csv(fdir / "test_predictions.csv")
    test_m = task_metrics(Y_np[test], prob[test], tasks)
    val_m = task_metrics(Y_np[val], prob[val], tasks)
    counts = {t: {"fit_pos": int(Y_np[fit, j].sum()), "fit_n": len(fit), "val_pos": int(Y_np[val, j].sum()),
                  "val_n": len(val)} for j, t in enumerate(tasks)}
    torch.save(model.state_dict(), fdir / "classifier.pt")
    torch.save(model.backbone.state_dict(), fdir / "backbone.pt")
    write_json(fdir / "metrics.json", {
        "fold": k, "run_name": args.run_name, "fingerprint": fingerprint, "n_fit": len(fit), "n_val": len(val), "n_test": len(test),
        "val_stratified": stratified, "best_epoch": best_epoch, "best_val_loss": best_loss,
        "epochs_run": len(history), "history": history, "counts": counts, "val": val_m, "test": test_m})
    final, old = out_dir / f"fold{k}", out_dir / f".fold{k}.old"
    if final.exists():                           # keep the old fold until the new one is in place
        shutil.rmtree(old, ignore_errors=True)
        final.rename(old)
    fdir.rename(final)
    shutil.rmtree(old, ignore_errors=True)
    print(f"fold {k}: best epoch {best_epoch} of {len(history)} (val loss {best_loss:.4f}); test AUROC "
          + ", ".join(f"{t} {m['auroc']:.2f}" for t, m in test_m.items()), flush=True)


def write_summary(out_dir: Path, tasks: list[str], all_folds: list[int], fingerprint: dict,
                  show: bool = True) -> pd.DataFrame:
    rows = []
    for k in all_folds:
        if not fold_done(out_dir, k, fingerprint):
            continue
        m = json.loads((out_dir / f"fold{k}" / "metrics.json").read_text())
        for t in tasks:
            c, te = m["counts"][t], m["test"][t]
            rows.append({"fold": m["fold"], "task": t, "n_fit": c["fit_n"], "n_pos_fit": c["fit_pos"],
                         "n_val": c["val_n"], "n_pos_val": c["val_pos"], "n_test": te["n"], "n_pos_test": te["n_pos"],
                         "test_auroc": te["auroc"], "test_ap": te["ap"], "best_epoch": m["best_epoch"]})
    summary = pd.DataFrame(rows)
    tmp = out_dir / "summary.csv.tmp"
    summary.to_csv(tmp, index=False)
    tmp.replace(out_dir / "summary.csv")
    if show and len(summary):
        done = summary["fold"].nunique()
        agg = summary.groupby("task", sort=False).agg(
            mean_auroc=("test_auroc", "mean"), sd_auroc=("test_auroc", "std"), mean_ap=("test_ap", "mean"),
            folds_with_auroc=("test_auroc", "count"))
        print(f"\n{done} of {len(all_folds)} folds complete; means over folds with a defined AUROC:")
        print(agg.round(3).to_string())
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks or len(set(tasks)) != len(tasks):
        raise SystemExit("--tasks must be a non-empty list without duplicates")
    strat_task = args.stratify_task or tasks[0]
    if strat_task not in tasks:
        raise SystemExit(f"--stratify-task {strat_task} is not in --tasks")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but no GPU is available; use --device cpu")
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)

    data, labels = load_inputs(args, tasks)
    folds = make_folds(args, labels, strat_task)
    all_folds = [int(k) for k in sorted(folds.unique())]
    if args.folds:
        try:
            run = [int(x) for x in args.folds.split(",") if x.strip()]
        except ValueError:
            raise SystemExit("--folds must be comma-separated integers")
        if not run or len(set(run)) != len(run) or not set(run) <= set(all_folds):
            raise SystemExit(f"--folds must be distinct values from {all_folds}")
    else:
        run = all_folds

    plan = plan_folds(args, labels, folds, strat_task, run)

    settings = {k: getattr(args, k) for k in RUN_KEYS}
    prolm_dir = Path(prolm.__file__).parent
    fingerprint = settings | {
        "tasks": tasks, "stratify_task": strat_task,
        "sha256_expression": sha256(args.expression_csv),
        "sha256_protein_embeddings": sha256(args.protein_embeddings),
        "sha256_labels": {t: sha256(Path(args.labels_dir) / f"labels_{t}.csv") for t in tasks},
        "sha256_fold_assignment": hashlib.sha256(folds.sort_index().to_csv().encode()).hexdigest(),
        "sha256_script": sha256(__file__),
        "sha256_prolm": {f: sha256(prolm_dir / f) for f in ("data.py", "model.py")},
        "versions": {p: metadata.version(p) for p in ("torch", "numpy", "pandas", "scikit-learn", "transformers")},
    }
    config = {"tool": TOOL, "fingerprint": fingerprint, "n_patients": len(labels),
              "n_proteins": len(data.protein_names), "n_folds": len(all_folds)}
    if prepare_output(args, out_dir, config, all_folds):   # new run: the ownership marker goes first
        write_atomic(out_dir / "config.json", json.dumps(clean_json(config), indent=2, allow_nan=False))
    for name, text in (("folds.csv", folds.rename_axis("eid").rename("fold").to_csv()),
                       ("protein_order.txt", "\n".join(data.protein_names) + "\n")):
        if not (out_dir / name).is_file():           # same fingerprint, so the content is the same
            write_atomic(out_dir / name, text)
    for t in tasks:
        n_pos, n_neg = int(labels[t].sum()), int((1 - labels[t]).sum())
        if min(n_pos, n_neg) < 2 * len(all_folds):
            print(f"warning: {t} has {n_pos} positives and {n_neg} negatives; per-fold metrics will be unstable "
                  "or undefined (do not report them as a cross-validated estimate)")

    for k in run:
        if fold_done(out_dir, k, fingerprint) and not args.overwrite:
            print(f"fold {k}: already complete, skipped (use --overwrite to recompute)")
            continue
        run_fold(args, k, plan[k], data, labels, tasks, device, out_dir, fingerprint)
        write_summary(out_dir, tasks, all_folds, fingerprint, show=False)
    write_summary(out_dir, tasks, all_folds, fingerprint)


if __name__ == "__main__":
    main()

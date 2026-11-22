"""
Step 6 - validate a fine-tuned Evo 2 model (step 4) on the held-out TEST individuals.

A small task runner: the checks to run are listed one per line in a tasks file (validation_tasks.txt;
`--list-tasks` shows every task and its parameters); new checks go in a --plugins file using the
`task` decorator (example and `ctx` fields: DOCUMENTATION.md section 7).
  model_info              model type, stored metrics, train/test leakage
  test_set_performance    queries the model on the test individuals: next-cluster loss, top-1/top-k,
                          region-conditioned top-1, vs training individuals and a bigram baseline,
                          with paired bootstrap CIs over test individuals (needs step 4's backbone)
  embedding_cooccurrence  do the step-3 embeddings predict cluster co-occurrence in the test individuals?
  cluster_consistency     do train and test carry the same clusters per haploblock? (permutation null)
  gwas_enrichment         haploblocks with more independent GWAS loci than expected (hits clumped by
                          distance), chance levels from randomly placed loci (empirical FDR) and from
                          rotations that keep the hits' spacing, and whether nearby hits share a haploblock (same genome build needed; columns
                          auto-detected, see GWAS_COLS; hit files from 5_download_gwas.py)

Split: by default the graph's train_individuals.txt / test_individuals.txt (steps 1-2); --splits
DIR_OR_URL (half_A.txt / half_B.txt), --train/--test-individuals, or --splits none (the model's own
held-out individuals). Split files: one sample ID per line, or a CSV with an "individual" column.

OUTPUT (--out-dir, default <graph-dir>/validation_<model name>/)
  report.md, summary.json, split.csv (individual, set)
  test_set_performance_chroms.csv     chrom, test_sequences, test_predictions, test_loss, test_top1,
                                      test_top<k>, test_region_top1, bigram_top1
  embedding_cooccurrence.csv          name, kind, auc, auc_distance_matched, auc_frequency_matched, auc_95ci
  cluster_consistency_regions.csv.gz  region, chrom, start, end, n_clusters_total/_train/_test,
                                      test_expected, test_ci_low/high, outside_ci, n_test_only,
                                      test_only_carriers, tvd, tvd_null_mean, tvd_p, chi2, chi2_df, chi2_p, chi2_q
  gwas_enrichment_<label>_regions.csv.gz   region, chrom, start, end, n_snps, n_hits, n_loci,
                                      expected_loci, fold, p, q, lead_snp, lead_p
  nemotron_summary.md                 --nemotron-summary: an NVIDIA Nemotron reading of the aggregate
                                      numbers ($NVIDIA_API_KEY; IDs and paths removed, https only)
  (a task listed twice writes its files with a _2, _3 ... suffix)

USAGE
  python 6_validate_evo2.py --model $G/evo2_next_gnn.pt --graph-dir $G --tasks validation_tasks.txt
  python 6_validate_evo2.py --model $G/evo2_next_gnn.pt --graph-dir $G --gwas haplograph/gwas/height.tsv
  python 6_validate_evo2.py --test                       # synthetic data, mock backbone, CPU
"""
import argparse
import glob
import importlib
import inspect
import json
import os
import re
import shlex
import time
import traceback
import zlib

import sys

if "--cudf" in sys.argv:
    # GPU dataframes: cuDF's pandas accelerator must be switched on BEFORE pandas is imported.
    # pandas code then runs on the GPU where cuDF supports it and falls back to the CPU elsewhere.
    try:
        import cudf.pandas
        cudf.pandas.install()
    except ImportError:
        sys.exit("--cudf needs RAPIDS cuDF (pip install cudf-cu12); run without --cudf to use plain pandas")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from haplo_common import (BH26_SPLITS, die, is_url, load_sample_map, log, read_graph_individuals,
                          read_individual_list, selftest_graph_dir, test_workdir)


GRAPH_SPLIT = "graph"


# ===========================================================================
# Task registry
# ===========================================================================

TASKS = {}


def task(name, description=""):
    """Register a validation task: fn(ctx, **params) -> {"summary": {...}, "notes": [...], "files": [...]}."""
    def wrap(fn):
        TASKS[name] = {"fn": fn, "description": description or (fn.__doc__ or "").strip().split("\n")[0]}
        return fn
    return wrap


def task_params(fn):
    sig = inspect.signature(fn)
    return [p for p in list(sig.parameters.values())[1:]
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]


def convert_params(name, fn, raw):
    """Check names against the function signature and convert values using its annotations."""
    params = {p.name: p for p in task_params(fn)}
    unknown = set(raw) - set(params)
    if unknown:
        raise ValueError(f"task {name}: unknown parameter(s) {sorted(unknown)}; "
                         f"accepted: {', '.join(params) or 'none'}")
    out = {}
    for k, v in raw.items():
        ann = params[k].annotation
        if ann is bool:
            out[k] = v.lower() in ("1", "true", "yes", "y", "on")
        elif ann in (int, float):
            out[k] = ann(float(v)) if ann is int else float(v)
        else:
            out[k] = v
    missing = [p.name for p in params.values() if p.default is p.empty and p.name not in out]
    if missing:
        raise ValueError(f"task {name}: missing required parameter(s) {missing}")
    return out


def parse_task_line(line, where=""):
    parts = shlex.split(line, comments=True)
    if not parts:
        return None
    raw = {}
    for kv in parts[1:]:
        if "=" not in kv:
            raise ValueError(f"{where}: expected key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        raw[k.strip().replace("-", "_")] = v.strip()
    return parts[0], raw


def read_tasks_file(path):
    specs = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            spec = parse_task_line(line, f"{path}:{i}")
            if spec:
                specs.append(spec)
    return specs


def load_plugin(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"validation_plugin_{len(TASKS)}", path)
    if spec is None:
        raise SystemExit(f"cannot load plugin {path}")
    module = importlib.util.module_from_spec(spec)
    module.task, module.log = task, log          # the plugin can use @task and log() directly
    before = set(TASKS)
    spec.loader.exec_module(module)
    log(f"plugin {path}: registered {sorted(set(TASKS) - before) or 'no tasks'}")


# ===========================================================================
# Data: graph, split, model
# ===========================================================================

class Graph:
    """Lazy access to the files written by 2_build_graph.py."""

    def __init__(self, graph_dir):
        self.dir = graph_dir
        self._nodes = self._carriers = self._individuals = self._regions = None

    def _f(self, name):
        p = os.path.join(self.dir, name)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} - run 2_build_graph.py first")
        return p

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = pd.read_csv(self._f("nodes.csv.gz"))
        return self._nodes

    @property
    def carriers(self):
        if self._carriers is None:
            self._carriers = sp.load_npz(self._f("carriers.npz")).tocsr()
        return self._carriers

    @property
    def individuals(self):
        if self._individuals is None:
            self._individuals = read_graph_individuals(self.dir)
        return self._individuals

    def _build_regions(self):
        codes, uniq = pd.factorize(self.nodes["region"])
        first = pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()
        reg = self.nodes.iloc[first][["region", "chrom", "start", "end"]].reset_index(drop=True)
        reg["n_clusters_total"] = np.bincount(codes, minlength=len(uniq))
        self._regions, self._codes = reg, codes

    @property
    def regions(self):
        """One row per haploblock region: region, chrom, start, end, n_clusters_total."""
        if self._regions is None:
            self._build_regions()
        return self._regions

    @property
    def region_codes(self):
        """Per-node index into `regions`."""
        if self._regions is None:
            self._build_regions()
        return self._codes


def model_kind(meta):
    """Recognise which step-4 script produced a checkpoint. Add a rule here for a new model type."""
    if not isinstance(meta, dict):
        return "unknown"
    if meta.get("mode") == "hash":
        return "4b hash-phenotype"
    if meta.get("mode") == "embeddings" or "proj_in" in meta:
        backend = meta.get("backend", "evo2")
        return "4a embeddings" + {"bionemo": " (BioNeMo)", "mock": " (MOCK backbone, not Evo 2)"}.get(backend, "")
    return "unknown"


class Model:
    """One fine-tuned model file, loaded on first use."""

    def __init__(self, path):
        self.path = path
        self._meta = None

    @property
    def meta(self):
        if self._meta is None:
            try:
                import torch
            except ImportError:
                raise SystemExit("reading the model file needs torch")
            try:       # step-4 checkpoints are plain dicts of tensors: never run a file's pickled code
                self._meta = torch.load(self.path, map_location="cpu", weights_only=True)
            except Exception as e:
                die(f"{self.path}: not a plain step-4 checkpoint ({type(e).__name__}: {e}); refusing to unpickle it")
        return self._meta

    @property
    def kind(self):
        return model_kind(self.meta)

    def _names(self, key):
        v = self.meta.get(key) if isinstance(self.meta, dict) else None
        return list(v) if v else None

    def held_out_individuals(self):
        return self._names("val_individuals")

    def trained_individuals(self):
        return self._names("train_individuals")


def join_path(base, name):
    return base.rstrip("/") + "/" + name if is_url(base) else os.path.join(base, name)


class Context:
    def __init__(self, args, model=None):
        self.args = args
        self.graph = Graph(args.graph_dir)
        self.model = model or Model(args.model)
        self.rng = np.random.default_rng(args.seed)
        self.out_dir = args.out_dir
        self._used = {}
        self.train_idx = self.test_idx = None
        self.test_source = None
        self.split_info = {}

    def out(self, name):
        """Path for an output file; repeated names get _2, _3 ... before the extension."""
        n = self._used.get(name, 0) + 1
        self._used[name] = n
        if n > 1:
            m = re.match(r"^(.*?)(\.[a-z]+(?:\.gz)?)?$", name)
            name = f"{m.group(1)}_{n}{m.group(2) or ''}"
        return os.path.join(self.out_dir, name)

    def setup_split(self):
        a = self.args
        inds = self.graph.individuals
        pos = {x: i for i, x in enumerate(inds)}
        to_graph = load_sample_map(a.sample_map) if a.sample_map else {}
        cache = os.path.join(self.out_dir, "inputs")

        # where the split comes from: explicit files > split folder (default BH_26) > model held-out
        if a.test_individuals or a.train_individuals:
            if not a.test_individuals:
                die("--train-individuals given without --test-individuals")
            test_src, train_src = a.test_individuals, a.train_individuals
        elif a.splits == GRAPH_SPLIT:
            g_test = os.path.join(self.graph.dir, "test_individuals.txt")
            g_train = os.path.join(self.graph.dir, "train_individuals.txt")
            if os.path.exists(g_test):
                test_src, train_src = g_test, (g_train if os.path.exists(g_train) else None)
            elif os.path.exists(g_train):
                die(f"{self.graph.dir} has a training set but no test_individuals.txt (step 2 ran with "
                    "--train-individuals but no test file). Pass the matching --test-individuals (and "
                    "--train-individuals) or --splits; falling back to another split could overlap the "
                    "training set")
            else:
                log(f"note: {g_test} not found; using the BH_26 split ({BH26_SPLITS})")
                test_src, train_src = join_path(BH26_SPLITS, a.test_file), join_path(BH26_SPLITS, a.train_file)
        elif a.splits and a.splits.lower() != "none":
            test_src, train_src = join_path(a.splits, a.test_file), join_path(a.splits, a.train_file)
        else:
            test_src = train_src = None

        def load(src, what):
            raw = read_individual_list(src, cache)
            names = [to_graph.get(x, x) for x in raw]
            found = [x for x in names if x in pos]
            if raw and not found:
                raise SystemExit(
                    f"none of the {len(raw):,} {what} IDs in {src} (e.g. {raw[:2]}) match the graph's "
                    f"individuals (e.g. {inds[:2]}). If the graph names individuals differently, pass "
                    "--sample-map (CSV: graph name, sample ID).")
            if len(found) < len(raw):
                log(f"warning: {len(raw) - len(found):,} of {len(raw):,} {what} IDs are not in the graph "
                    f"(e.g. {[x for x in names if x not in pos][:3]}), ignored")
            return found, len(raw)

        if test_src:
            test, n_test_file = load(test_src, "test")
            self.test_source = test_src
        else:
            test = self.model.held_out_individuals()
            if not test:
                raise SystemExit("--splits none and the model file has no held-out individuals "
                                 "(val_individuals); pass --test-individuals or --splits")
            test, n_test_file = [x for x in test if x in pos], len(test)
            self.test_source = "held-out individuals stored in the model"
        test_set = set(test)

        if train_src:
            train, n_train_file = load(train_src, "train")
            overlap = test_set & set(train)
            if overlap:
                log(f"warning: {len(overlap):,} individuals are in both the train and the test file "
                    f"(e.g. {sorted(overlap)[:3]}); removed from train")
            train_set = set(train) - test_set
            train_source = train_src
        elif not test_src and self.model.trained_individuals():     # --splits none: the model's own split
            train_set = {x for x in self.model.trained_individuals() if x in pos} - test_set
            n_train_file, train_source = None, "individuals the model was trained on"
        else:
            train_set, n_train_file = set(inds) - test_set, None
            train_source = "all other graph individuals"

        # the graph's own training set (2_build_graph.py --train-individuals) must not contain test individuals
        graph_train_file = os.path.join(self.graph.dir, "train_individuals.txt")
        if os.path.exists(graph_train_file):
            graph_train = set(read_individual_list(graph_train_file))
            in_both = test_set & graph_train
            self.split_info["graph_built_on"] = f"{len(graph_train):,} training individuals (train_individuals.txt)"
            self.split_info["test_individuals_in_graph_training_set"] = len(in_both)
            if in_both:
                log(f"WARNING: {len(in_both):,} test individuals are in the graph's training set "
                    f"(e.g. {sorted(in_both)[:3]}): steps 2-4 saw them. Rebuild the graph from the "
                    "train half (1_download_haplograph.py, then 2_build_graph.py)")
            elif graph_train != train_set:
                log(f"note: the graph was built on {len(graph_train):,} training individuals, the train set "
                    f"here has {len(train_set):,} ({len(graph_train - train_set):,} / {len(train_set - graph_train):,} "
                    "differ); test individuals are not in it")
        else:
            self.split_info["graph_built_on"] = "all individuals (no train_individuals.txt)"
            log("note: the graph was built from ALL individuals (2_build_graph.py --no-split), "
                "so its edge weights include the test individuals")

        excluded = len(inds) - len(train_set) - len(test_set)
        self.test_idx = np.array(sorted(pos[x] for x in test_set), dtype=np.int64)
        self.train_idx = np.array(sorted(pos[x] for x in train_set), dtype=np.int64)
        if len(self.test_idx) == 0 or len(self.train_idx) == 0:
            raise SystemExit(f"empty split: {len(self.train_idx)} train / {len(self.test_idx)} test individuals")
        self.split_info = {**self.split_info, "train_source": train_source, "test_source": self.test_source,
                           "train_in_file": n_train_file, "test_in_file": n_test_file,
                           "train": int(len(self.train_idx)), "test": int(len(self.test_idx)),
                           "graph_individuals_in_neither_set": int(excluded)}
        neither = sorted(set(inds) - train_set - test_set)
        pd.DataFrame({"individual": [inds[i] for i in self.train_idx] + [inds[i] for i in self.test_idx] + neither,
                      "set": ["train"] * len(self.train_idx) + ["test"] * len(self.test_idx)
                             + ["excluded"] * len(neither)}
                     ).to_csv(os.path.join(self.out_dir, "split.csv"), index=False)
        log(f"split: {len(self.train_idx):,} train ({train_source}) / {len(self.test_idx):,} test "
            f"({self.test_source})" + (f"; {excluded:,} graph individuals in neither set, left out" if excluded else ""))


# ===========================================================================
# Helpers
# ===========================================================================

def bh(p):
    """Benjamini-Hochberg q-values (NaN-safe)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        pv = p[ok]
        order = np.argsort(pv)
        ranked = pv[order] * ok.sum() / np.arange(1, ok.sum() + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty_like(pv)
        out[order] = np.minimum(ranked, 1.0)
        q[ok] = out
    return q


def group_counts(C, idx_matrix):
    """Carriers per node for several groups at once: C [N, I] @ G [I, k] -> [N, k] float32."""
    return np.asarray(C @ idx_matrix, dtype=np.float32)


def indicator(n, idx):
    g = np.zeros(n, np.float32)
    g[idx] = 1
    return g


# ===========================================================================
# Built-in tasks
# ===========================================================================

@task("model_info", "Model type, training metadata and stored metrics; checks for train/test leakage")
def model_info(ctx):
    meta = ctx.model.meta
    kind = model_kind(meta)
    s = {"model_file": ctx.model.path, "kind": kind}
    notes = []
    if isinstance(meta, dict):
        for k in ("objective", "task", "phenotype", "graph_method", "evo2_model", "model_size",
                  "layer", "layer_index", "backend"):
            if meta.get(k) is not None:
                s[k] = meta[k]
        if meta.get("chroms"):
            s["trained_chromosomes"] = ",".join(meta["chroms"])
            missing = set(meta["chroms"]) - set(ctx.graph.nodes["chrom"].unique())
            if missing:
                notes.append(f"WARNING: model was trained on chromosomes not in this graph: {sorted(missing)}")
        tb = meta.get("trained_blocks") or {}
        s["fine_tuned_evo2_params"] = int(sum(getattr(t, "numel", lambda: 0)() for t in tb.values()))
        for phase in ("before", "after"):
            for k, v in ((meta.get("metrics") or {}).get(phase) or {}).items():
                if isinstance(v, (int, float)):
                    s[f"stored_val_{k}_{phase}"] = round(float(v), 4)
        held, trained = ctx.model.held_out_individuals(), ctx.model.trained_individuals()
        test_names = [ctx.graph.individuals[i] for i in ctx.test_idx] if ctx.test_idx is not None else []
        if trained is not None and test_names:
            # the checkpoint lists who it was trained and validated on; both must exclude the test set
            # with --splits none the test set IS the validation set (used for reporting only)
            own_split = ctx.split_info.get("test_source") == "held-out individuals stored in the model"
            trained = set(trained) | (set() if own_split else set(held or []))
            leaked = [x for x in test_names if x in trained]
            s["test_individuals_seen_in_training"] = len(leaked)
            notes.append(f"WARNING: {len(leaked):,} of {len(test_names):,} test individuals are in the model's "
                         f"training or validation set (e.g. {leaked[:3]})" if leaked else
                         "no leakage: no test individual is in the model's training or validation set")
        elif kind.startswith("4a"):
            notes.append("the checkpoint does not record its training individuals: leakage not checked")
        elif kind.startswith("4b"):
            notes.append("4b models are trained on hash-phenotype rows, not individuals; their held-out rows "
                         "are stored as val_rows (indices into the training TSV)")
    else:
        notes.append("the model file is not a dict; only its type is reported")
    return {"summary": s, "notes": notes}


@task("cluster_consistency", "Train vs test: distinct clusters per haploblock, unseen clusters, frequency shifts")
def cluster_consistency(ctx, min_carriers: int = 1, permutations: int = 200, alpha: float = 0.05,
                        batch: int = 32):
    """A cluster counts as present in a set when at least `min_carriers` individuals of it carry it."""
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    C = ctx.graph.carriers.astype(np.float32)
    codes, regions = ctx.graph.region_codes, ctx.graph.regions.copy()
    n_reg, n_ind = len(regions), C.shape[1]
    tr, te = ctx.train_idx, ctx.test_idx

    Rm = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))),
                       shape=(n_reg, len(codes)))            # region x node membership

    def stats_for(a, b):
        """a, b: [N, k] carrier counts of train-like / test-like groups -> per-region stats [R, k]."""
        n_a = Rm @ (a >= min_carriers).astype(np.float32)
        n_b = Rm @ (b >= min_carriers).astype(np.float32)
        tot_a, tot_b = Rm @ a, Rm @ b
        fa = a / np.maximum(tot_a[codes], 1e-12); fb = b / np.maximum(tot_b[codes], 1e-12)
        return n_a, n_b, 0.5 * (Rm @ np.abs(fa - fb))

    a = group_counts(C, indicator(n_ind, tr)[:, None])
    b = group_counts(C, indicator(n_ind, te)[:, None])
    n_tr, n_te, tvd = (x[:, 0] for x in stats_for(a, b))
    a, b = a[:, 0], b[:, 0]

    test_only = (b >= min_carriers) & (a < min_carriers)
    regions["n_clusters_train"] = n_tr.astype(int)
    regions["n_clusters_test"] = n_te.astype(int)
    regions["n_test_only"] = np.bincount(codes, weights=test_only, minlength=n_reg).astype(int)
    regions["test_only_carriers"] = np.bincount(codes, weights=np.where(test_only, b, 0), minlength=n_reg).astype(int)
    regions["tvd"] = tvd

    # chi-square test of homogeneity per region (2 x k table of carrier counts)
    col = a + b
    A = np.bincount(codes, weights=a, minlength=n_reg); B = np.bincount(codes, weights=b, minlength=n_reg)
    T = np.maximum(A + B, 1e-12)
    ea, eb = col * A[codes] / T[codes], col * B[codes] / T[codes]
    contrib = np.where(ea > 0, (a - ea) ** 2 / np.maximum(ea, 1e-12), 0) + \
              np.where(eb > 0, (b - eb) ** 2 / np.maximum(eb, 1e-12), 0)
    chi2 = np.bincount(codes, weights=contrib, minlength=n_reg)
    df = np.bincount(codes, weights=col > 0, minlength=n_reg) - 1
    ok = (df > 0) & (A > 0) & (B > 0)
    p = np.full(n_reg, np.nan)
    p[ok] = stats.chi2.sf(chi2[ok], df[ok])
    regions["chi2"], regions["chi2_df"], regions["chi2_p"] = chi2, df.astype(int), p
    regions["chi2_q"] = bh(p)

    # permutation null: random splits of the same individuals with the same sizes
    pooled = np.concatenate([tr, te])
    null_n, null_tvd = [], []
    log(f"  {n_reg:,} regions; {permutations} random splits of {len(pooled):,} individuals")
    for s in range(0, permutations, batch):
        k = min(batch, permutations - s)
        Ga = np.zeros((n_ind, k), np.float32); Gb = np.zeros((n_ind, k), np.float32)
        for j in range(k):
            perm = ctx.rng.permutation(pooled)
            Gb[perm[:len(te)], j] = 1; Ga[perm[len(te):], j] = 1
        _, nb, tv = stats_for(group_counts(C, Ga), group_counts(C, Gb))
        null_n.append(nb.astype(np.float32)); null_tvd.append(tv.astype(np.float32))
    null_n, null_tvd = np.hstack(null_n), np.hstack(null_tvd)
    regions["test_expected"] = null_n.mean(1)
    regions["test_ci_low"] = np.percentile(null_n, 100 * alpha / 2, axis=1)
    regions["test_ci_high"] = np.percentile(null_n, 100 * (1 - alpha / 2), axis=1)
    regions["outside_ci"] = (n_te < regions["test_ci_low"]) | (n_te > regions["test_ci_high"])
    regions["tvd_null_mean"] = null_tvd.mean(1)
    regions["tvd_p"] = (1 + (null_tvd >= tvd[:, None] - 1e-12).sum(1)) / (1 + permutations)

    path = ctx.out("cluster_consistency_regions.csv.gz")
    cols = ["region", "chrom", "start", "end", "n_clusters_total", "n_clusters_train", "n_clusters_test",
            "test_expected", "test_ci_low", "test_ci_high", "outside_ci", "n_test_only",
            "test_only_carriers", "tvd", "tvd_null_mean", "tvd_p", "chi2", "chi2_df", "chi2_p", "chi2_q"]
    regions[cols].to_csv(path, index=False, float_format="%.6g")

    frac_out = float(regions["outside_ci"].mean())
    frac_tvd = float((regions["tvd_p"] < alpha).mean())
    n_chi = int((regions["chi2_q"] < alpha).sum())
    consistent = frac_out <= 2 * alpha and frac_tvd <= 2 * alpha
    s = {
        "regions": n_reg, "train_individuals": len(tr), "test_individuals": len(te),
        "mean_clusters_per_region_train": round(float(n_tr.mean()), 3),
        "mean_clusters_per_region_test": round(float(n_te.mean()), 3),
        "mean_clusters_per_region_test_expected": round(float(regions["test_expected"].mean()), 3),
        "regions_outside_95pct_interval": f"{frac_out:.3%} (expected ~{alpha:.0%})",
        "regions_tvd_p_below_alpha": f"{frac_tvd:.3%} (expected ~{alpha:.0%})",
        "regions_chi2_q_below_alpha": n_chi,
        "test_only_clusters": int(test_only.sum()),
        "regions_with_test_only_clusters": int((regions["n_test_only"] > 0).sum()),
        "verdict": "consistent" if consistent else "inconsistent - see per-region file",
    }
    per_chrom = regions.groupby("chrom", sort=False)["outside_ci"].mean()
    worst = per_chrom.sort_values(ascending=False).head(3)
    notes = ["chromosomes with most regions outside the interval: "
             + ", ".join(f"{c} {v:.1%}" for c, v in worst.items())]
    if test_only.sum():
        notes.append(f"{int(test_only.sum()):,} clusters are carried in test but not in train: the model never saw "
                     "them, so predictions involving them are extrapolation")
    notes.append("chi-square p-values treat carrier counts as independent (an individual can carry 2 clusters "
                 "of a region); the permutation-based columns do not need that assumption")
    return {"summary": s, "notes": notes, "files": [path]}


# ---- GWAS ------------------------------------------------------------------

GWAS_COLS = {
    "chrom": ["chr", "chrom", "#chrom", "chr_id", "chromosome", "#chr"],
    "pos": ["pos", "bp", "chr_pos", "position", "base_pair_location", "pos_b37", "pos_b38"],
    "p": ["p", "pval", "p_value", "p-value", "pvalue", "p_bolt_lmm", "p.value"],
    "snp": ["rsid", "snp", "snps", "snpid", "markername", "variant_id", "id", "snp_id_current"],
}


def read_gwas(path, chrom_col="", pos_col="", p_col="", snp_col="", sep="auto"):
    import io
    import zipfile
    if path.endswith(".zip"):
        z = zipfile.ZipFile(path)
        members = [n for n in z.namelist() if not n.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"{path}: zip must contain exactly one file, found {members[:5]}")
        member = members[0]
        opener = lambda: io.TextIOWrapper(z.open(member), encoding="utf-8", errors="replace")
    else:
        import gzip
        opener = (lambda: gzip.open(path, "rt", errors="replace")) if path.endswith(".gz") else \
                 (lambda: open(path, errors="replace"))
    with opener() as fh:
        header = fh.readline().rstrip("\n\r")
    if sep == "auto":
        sep = "\t" if "\t" in header else ("," if "," in header else r"\s+")
    elif sep in ("tab", "\\t"):
        sep = "\t"
    names = [h.strip() for h in re.split(sep, header.strip())]
    lower = {n.lower(): n for n in names}

    def pick(kind, override, required=True):
        if override:
            if override not in names:
                raise ValueError(f"{path}: column {override!r} not found; columns: {names}")
            return override
        for cand in GWAS_COLS[kind]:
            if cand in lower:
                return lower[cand]
        if required:
            raise ValueError(f"{path}: no {kind} column found (tried {GWAS_COLS[kind]}); columns: {names}. "
                             f"Set {kind}_col=")
        return None

    cols = {"chrom": pick("chrom", chrom_col), "pos": pick("pos", pos_col), "p": pick("p", p_col),
            "snp": pick("snp", snp_col, required=False)}
    use = [c for c in cols.values() if c]
    with opener() as fh:
        df = pd.read_csv(fh, sep=sep, usecols=use, dtype=str, engine="c" if sep != r"\s+" else "python")
    out = pd.DataFrame({
        "chrom": df[cols["chrom"]].astype(str).str.strip().str.replace(r"^chr", "", regex=True, case=False),
        "pos": pd.to_numeric(df[cols["pos"]], errors="coerce"),
        "p": pd.to_numeric(df[cols["p"]], errors="coerce"),
        "snp": df[cols["snp"]].astype(str) if cols["snp"] else "",
    })
    out["chrom"] = "chr" + out["chrom"].replace({"23": "X", "24": "Y"})
    n0 = len(out)
    out = out.dropna(subset=["pos", "p"])
    out = out[(out["p"] >= 0) & (out["p"] <= 1)]
    out["pos"] = out["pos"].astype(np.int64)
    return out.reset_index(drop=True), cols, n0 - len(out)


def region_pairs(reg_chrom, starts, ends, snp_chrom, snp_pos):
    """(SNP index, region index) for every region whose [start, end] contains the SNP, so a SNP in
    two overlapping (window-extended) regions counts for both."""
    out_s, out_r = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    for c in pd.unique(reg_chrom):
        ri, si = np.where(reg_chrom == c)[0], np.where(snp_chrom == c)[0]
        if len(si) == 0:
            continue
        order = ri[np.argsort(starts[ri], kind="stable")]
        pos = snp_pos[si]
        hi = np.searchsorted(starts[order], pos, side="right")                     # start <= pos: [0, hi)
        lo = np.searchsorted(np.maximum.accumulate(ends[order]), pos, side="left")  # all ends < pos before lo
        n = np.maximum(hi - lo, 0)
        rep = np.repeat(np.arange(len(si)), n)
        k = np.repeat(lo, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        keep = ends[order[k]] >= pos[rep]                                          # nested regions
        out_s.append(si[rep[keep]]); out_r.append(order[k[keep]])
    return np.concatenate(out_s), np.concatenate(out_r)


def clump_hits(chrom, pos, p, hit, dist):
    """Greedy distance clumping of the hits, lowest p first: each hit joins the first lead within
    `dist` bp. Returns the lead's index for every hit (a lead points to itself), -1 for non-hits."""
    group = np.full(len(pos), -1, dtype=np.int64)
    for c in pd.unique(chrom[hit]):
        idx = np.where(hit & (chrom == c))[0]
        idx = idx[np.argsort(pos[idx], kind="stable")]
        ps = pos[idx]
        for k in np.lexsort((ps, p[idx])):
            if group[idx[k]] < 0:
                w = idx[np.searchsorted(ps, ps[k] - dist):np.searchsorted(ps, ps[k] + dist, side="right")]
                group[w[group[w] < 0]] = idx[k]
    return group


@task("gwas_enrichment", "Are haploblocks enriched in GWAS loci, and do haploblocks keep each locus together?")
def gwas_enrichment(ctx, gwas: str, pvalue: float = 5e-8, clump: str = "auto", window: int = 0,
                    permutations: int = 1000, label: str = "", chrom_col: str = "", pos_col: str = "",
                    p_col: str = "", snp_col: str = "", sep: str = "auto", mode: str = "auto"):
    """Hits (p < pvalue) are clumped into independent loci (lowest p first; hits within `clump` bp
    of a lead join its locus; auto = 25 kb for hit lists, 250 kb for summary statistics), and each
    haploblock is tested for more loci than expected: vs its length for a hit list (mode 'hits', e.g.
    the GWAS Catalog), vs its tested SNPs for full summary statistics (mode 'summary'); 'auto' =
    summary if most rows have p > 1e-5. Two nulls, `permutations` draws each: (1) every locus, with
    its hits, moved to a random place on its chromosome: haploblocks reaching q < 0.05 by chance and
    the empirical FDR; (2) all hits rotated along the chromosome with their spacing kept: calls that
    the clustering of hits alone produces, hit pairs sharing a haploblock (more than expected = the
    haploblock boundaries keep GWAS loci together) and hits inside haploblocks. window = bp added on
    each side of every haploblock (a SNP within reach of two counts for both)."""
    if mode not in ("auto", "summary", "hits"):
        raise ValueError(f"mode must be auto, summary or hits, not {mode!r}")
    for name, v in (("permutations", permutations), ("window", window)):
        if v < 0:
            raise ValueError(f"{name} must be >= 0")
    if not 0 < pvalue <= 1:
        raise ValueError(f"pvalue must be in (0, 1], not {pvalue}")
    if clump != "auto" and not (re.fullmatch(r"\d+", str(clump))):
        raise ValueError(f"clump must be 'auto' or a number of bp >= 0, not {clump!r}")
    if not os.path.exists(gwas):
        raise ValueError(f"GWAS file {gwas} not found (5_download_gwas.py writes haplograph/gwas/<label>.tsv)")
    sidecar = re.sub(r"\.tsv$", ".json", gwas)                 # written by 5_download_gwas.py
    if sidecar != gwas and os.path.exists(sidecar):
        build = json.load(open(sidecar)).get("genome_build")
        if build and build != "GRCh38":
            raise ValueError(f"{gwas} is {build}; the haplograph regions are GRCh38 - lift the positions over first")
    label = label or re.sub(r"(\.gz|\.zip|\.tsv|\.txt|\.csv)+$", "", os.path.basename(gwas))
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
    snps, cols, n_bad = read_gwas(gwas, chrom_col, pos_col, p_col, snp_col, sep)
    reg = ctx.graph.regions.dropna(subset=["start", "end"]).reset_index(drop=True).copy()
    reg_chrom = reg["chrom"].to_numpy()
    starts = reg["start"].to_numpy(np.int64) - window
    ends = reg["end"].to_numpy(np.int64) + window
    if window == 0:                      # adjacent haploblocks share a coordinate: it belongs to the next one
        ends = ends - (pd.MultiIndex.from_arrays([reg_chrom, ends]).isin(
            pd.MultiIndex.from_arrays([reg_chrom, starts])) & (ends > starts)).astype(np.int64)

    on = snps["chrom"].isin(set(reg_chrom))
    snps = snps[on].reset_index(drop=True)
    if len(snps) == 0:
        raise ValueError(f"no GWAS SNPs on the graph's chromosomes ({sorted(set(reg_chrom))[:5]}...)")
    hit = (snps["p"] < pvalue).to_numpy()
    # full summary statistics are mostly non-significant rows; hit lists (catalog: p < 1e-5) are not
    summary_stats = mode == "summary" or (mode == "auto" and (snps["p"] > 1e-5).mean() > 0.5)
    s_chrom, s_pos = snps["chrom"].to_numpy(), snps["pos"].to_numpy()

    # haploblock span per chromosome (min start .. max end); SNPs outside it are not tested
    span = reg.assign(s=starts, e=ends).groupby("chrom").agg(lo=("s", "min"), hi=("e", "max"))
    lo = span.loc[s_chrom, "lo"].to_numpy(); hi = span.loc[s_chrom, "hi"].to_numpy()
    in_span = (s_pos >= lo) & (s_pos <= hi)
    frac_outside = float(1 - in_span.mean())
    order = np.lexsort((s_pos, s_chrom))[in_span[np.lexsort((s_pos, s_chrom))]]   # by chromosome, position
    snps, hit, s_chrom, s_pos, lo, hi = snps.iloc[order].reset_index(drop=True), hit[order], \
        s_chrom[order], s_pos[order], lo[order], hi[order]
    K, M = int(hit.sum()), len(snps)
    if M == 0:
        raise ValueError("none of the GWAS SNPs falls inside the haploblock span - check the genome build (GRCh38)")
    clump = (250_000 if summary_stats else 25_000) if clump == "auto" else int(clump)
    pv = snps["p"].to_numpy()
    group = clump_hits(s_chrom, s_pos, pv, hit, clump)            # lead index of each hit's locus, -1 = no hit
    locus = group == np.arange(M)
    Kl, R = int(locus.sum()), len(reg)
    length = (ends - starts + 1).astype(np.float64)
    si, ri = region_pairs(reg_chrom, starts, ends, s_chrom, s_pos)
    if len(si) == 0:
        raise ValueError("none of the GWAS SNPs falls inside a haploblock - check the genome build (GRCh38)")
    inside = np.zeros(M, bool)
    inside[si] = True
    n_snps = np.bincount(ri, minlength=R)
    if not summary_stats:                           # loci per bp of haploblock span, per chromosome
        loci_chrom = pd.Series(locus).groupby(s_chrom).sum()
        rate = (loci_chrom / (span["hi"] - span["lo"] + 1)).reindex(span.index).fillna(0)
        expected_len = length * rate.loc[reg_chrom].to_numpy()

    # hits only from here on. The nulls move hits: along the chromosome span for a hit list; for
    # summary statistics over the SNP order (hit labels onto other tested SNPs, so SNP counts stay)
    h = np.where(hit)[0]
    hc, hpos, hloc, hlo, L = s_chrom[h], s_pos[h], locus[h], lo[h], (hi - lo + 1)[h]
    ccode, cu = pd.factorize(hc)
    gcode, gu = pd.factorize(group[h])
    first = pd.Series(np.arange(M)).groupby(s_chrom).min().loc[hc].to_numpy()   # SNP index range per chromosome
    n_chrom = pd.Series(s_chrom).value_counts().loc[hc].to_numpy()

    def count(pos):
        """Hits and loci per region for hit positions `pos`, and how many hits are inside any region."""
        sj, rj = region_pairs(reg_chrom, starts, ends, hc, pos)
        return (np.bincount(rj, minlength=R), np.bincount(rj, weights=hloc[sj], minlength=R).astype(int),
                np.unique(sj).size)

    def region_test(nh, nl):
        if summary_stats:
            exp = n_snps * Kl / M
            return exp, np.where(n_snps > 0, stats.hypergeom.sf(nl - 1, M, Kl, n_snps), np.nan)
        return expected_len, stats.poisson.sf(nl - 1, np.maximum(expected_len, 1e-300))

    def pairs(nh):
        return int((nh * (nh - 1) // 2).sum())      # hit pairs sharing a haploblock

    def moved(u):                                    # circular move within the chromosome
        if summary_stats:
            return s_pos[first + (h - first + (u * n_chrom).astype(np.int64)) % n_chrom]
        return hlo + (hpos - hlo + (u * L).astype(np.int64)) % L

    n_hits, n_loci, obs_inside = count(hpos)
    expected, p = region_test(n_hits, n_loci)
    reg["n_snps"], reg["n_hits"], reg["n_loci"], reg["expected_loci"] = n_snps, n_hits, n_loci, expected
    reg["fold"] = np.where(expected > 0, n_loci / np.maximum(expected, 1e-300), np.nan)
    reg["p"], reg["q"] = p, bh(p)
    lead = pd.DataFrame({"r": ri, "p": pv[si], "snp": snps["snp"].to_numpy()[si] if cols["snp"] else
                         (s_chrom[si] + ":" + s_pos[si].astype(str))})
    lead = lead.sort_values("p", kind="stable").drop_duplicates("r").set_index("r")
    reg["lead_snp"] = lead["snp"].reindex(np.arange(R)).to_numpy()
    reg["lead_p"] = lead["p"].reindex(np.arange(R)).to_numpy()
    n_sig = int((reg["q"] < 0.05).sum())
    obs_pairs = pairs(n_hits)

    null = np.zeros((permutations, 4))              # loci moved: calls | rotation: calls, pairs, inside
    for j in range(permutations):
        nh, nl, _ = count(moved(ctx.rng.random(len(gu))[gcode]))
        null[j, 0] = (bh(region_test(nh, nl)[1]) < 0.05).sum()
        nh, nl, ins = count(moved(ctx.rng.random(len(cu))[ccode]))
        null[j, 1:] = (bh(region_test(nh, nl)[1]) < 0.05).sum(), pairs(nh), ins
    ran = permutations > 0
    mean = lambda k: round(float(null[:, k].mean()), 2) if ran else None
    pval = lambda k, o: round(float((1 + (null[:, k] >= o).sum()) / (1 + permutations)), 4) if ran else None
    iv = pd.DataFrame({"c": reg_chrom, "s": starts, "e": ends}).sort_values(["c", "s"])
    reach = iv.groupby("c")["e"].cummax().groupby(iv["c"]).shift().fillna(-1).to_numpy()
    covered = float(np.maximum(0, iv["e"] - np.maximum(iv["s"], reach + 1) + 1).sum()
                    / (span["hi"] - span["lo"] + 1).sum())        # union of the (window-extended) haploblocks
    max_loci = length // clump + 1 if clump else n_hits       # loci a haploblock can hold after clumping
    best_p = (stats.hypergeom.sf(np.minimum(max_loci, n_snps) - 1, M, Kl, n_snps) if summary_stats
              else stats.poisson.sf(max_loci - 1, np.maximum(expected_len, 1e-300)))

    path = ctx.out(f"gwas_enrichment_{safe}_regions.csv.gz")
    reg[["region", "chrom", "start", "end", "n_snps", "n_hits", "n_loci", "expected_loci", "fold", "p", "q",
         "lead_snp", "lead_p"]].to_csv(path, index=False, float_format="%.6g")
    s = {
        "gwas_file": gwas, "label": label, "columns_used": ",".join(f"{k}={v}" for k, v in cols.items() if v),
        "file_type": "summary statistics" if summary_stats else "hits only",
        "pvalue_threshold": pvalue, "clump_bp": clump, "window_bp": window, "permutations": permutations,
        "snps_in_span": M, "hits": K, "loci": Kl,
        "regions_with_loci": int((n_loci > 0).sum()), "regions": R,
        "per_region_background": "tested SNPs (hypergeometric)" if summary_stats else "haploblock length (Poisson)",
        "regions_enriched_q<0.05": n_sig,
        "regions_q<0.05_by_chance": mean(0),
        "empirical_fdr": round(min(1.0, mean(0) / n_sig), 3) if ran and n_sig else None,
        "regions_q<0.05_under_rotation": mean(1),
        "empirical_fdr_rotation": round(min(1.0, mean(1) / n_sig), 3) if ran and n_sig else None,
        "top_region": reg.loc[reg["p"].idxmin(), "region"] if n_loci.any() else None,
        "top_region_q": float(reg["q"].min()) if n_loci.any() else None,
        "hit_pairs_same_haploblock": obs_pairs,
        "hit_pairs_same_haploblock_under_rotation": mean(2),
        "hit_pairs_same_haploblock_p": pval(2, obs_pairs),
        "hits_in_haploblocks": obs_inside,
        "hits_in_haploblocks_under_rotation": mean(3),
        "hits_in_haploblocks_p": pval(3, obs_inside),
        "haploblock_coverage_of_span": f"{covered:.1%}",
        "snps_outside_haploblock_span": f"{frac_outside:.1%}",
    }
    notes = [f"per-haploblock tests count {Kl:,} loci ({K:,} hits clumped within {clump:,} bp)"]
    if ran:
        notes.append(f"loci moved at random: {mean(0):.2f} haploblocks with q < 0.05 by chance"
                     + (f" (empirical FDR {s['empirical_fdr']:.0%})" if n_sig else ""))
        if n_sig and mean(1) > max(2 * mean(0), 0.05):
            notes.append(f"all hits rotated with their spacing kept: {mean(1):.1f} haploblocks with q < 0.05 "
                         f"(FDR {s['empirical_fdr_rotation']:.0%}), so the calls reflect dense clusters of hits; they "
                         f"are independent signals only if LD there does not reach beyond {clump:,} bp (raise `clump` "
                         "to check)")
        notes.append(f"hit pairs sharing a haploblock: {obs_pairs:,} vs {mean(2)} under rotation "
                     f"(p = {s['hit_pairs_same_haploblock_p']:.3g}); more than expected means the haploblock "
                     "boundaries keep GWAS loci together" + (" (window > 0: overlapping haploblocks share pairs)"
                                                              if window else ""))
    if Kl and np.nanmin(best_p) > 0.05 / R:
        notes.append(f"with clump={clump:,} bp a haploblock holds at most length/clump + 1 loci, so no haploblock "
                     f"can reach p < 0.05/{R} even if every possible locus is a hit (no power); lower `clump` to test "
                     "single haploblocks")
    if covered > 0.95:
        notes.append(f"haploblocks cover {covered:.0%} of each chromosome span, so the hits-in-haploblocks test is "
                     "uninformative here; use the per-haploblock and same-haploblock results")
    if frac_outside > 0.05:                         # GRCh38 catalog hits on chr21: 1.3%; hg19 positions: ~10%
        notes.append(f"{frac_outside:.0%} of the GWAS SNPs fall outside the haploblock spans - check that the GWAS "
                     "file and the haploblock regions use the same genome build (e.g. GIANT 2022 is hg19/GRCh37)")
    if n_bad:
        notes.append(f"{n_bad:,} GWAS rows had no usable position or p-value and were skipped")
    top = reg[reg["q"] < 0.05].sort_values("p").head(5)
    if len(top):
        notes.append("top enriched haploblocks: " + "; ".join(
            f"{r.region} ({r.n_loci} loci, {r.n_hits} hits, {r.fold:.1f}x, q={r.q:.2g})" for r in top.itertuples()))
    return {"summary": s, "notes": notes, "files": [path]}


# ---- querying the fine-tuned model ----------------------------------------

def bigram_table(data, chrom, individuals):
    """Most frequent next cluster after each cluster (local indices), counted on `individuals`;
    clusters never seen as 'current' fall back to the most frequent next cluster overall.
    Returns (best next [n], sum over transitions of -log p(next | current), number of transitions):
    the plug-in conditional entropy H(next | current) of the training sequences."""
    n = len(data.candidates(chrom))
    keys = []
    for i in individuals:
        _, local = data.tokens(chrom, i)
        if len(local) >= 2:
            keys.append(local[:-1].astype(np.int64) * n + local[1:])
    best = np.full(n, -1, np.int64)
    if not keys:
        return best, 0.0, 0
    keys, counts = np.unique(np.concatenate(keys), return_counts=True)
    cur, nxt = keys // n, keys % n
    order = np.lexsort((-counts, cur))                      # per current cluster: highest count first
    first = np.unique(cur[order], return_index=True)[1]
    best[cur[order][first]] = nxt[order][first]
    fallback = np.bincount(nxt, weights=counts, minlength=n).argmax()
    best[best < 0] = fallback
    per_cur = np.bincount(cur, weights=counts, minlength=n)
    nll = float(-(counts * np.log(counts / per_cur[cur])).sum())
    return best, nll, int(counts.sum())


def paired_bootstrap(per_ind, n_boot, rng, alpha=0.05):
    """per_ind: [individuals, 3] sums of (model correct, bigram correct, predictions). Resamples
    INDIVIDUALS (their predictions are correlated). Returns CIs of model, bigram and model - bigram."""
    m, b, n = per_ind[:, 0], per_ind[:, 1], per_ind[:, 2]
    idx = rng.integers(0, len(per_ind), (n_boot, len(per_ind)))
    tot = n[idx].sum(1)
    tot = np.where(tot > 0, tot, 1)
    stats_ = np.stack([m[idx].sum(1) / tot, b[idx].sum(1) / tot, (m[idx].sum(1) - b[idx].sum(1)) / tot])
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=1)
    return [(round(float(a), 4), round(float(c), 4)) for a, c in zip(lo, hi)]


@task("test_set_performance", "Query the fine-tuned model on the test individuals (next-cluster prediction)")
def test_set_performance(ctx, max_examples: int = 500, train_examples: int = 200, top_k: int = 5,
                         baseline_individuals: int = 0, bootstrap: int = 1000, device: str = "",
                         ckpt_dir: str = ""):
    """max_examples / train_examples = (individual, chromosome) sequences evaluated from the test set /
    from the model's training individuals (0 = all); baseline_individuals = training individuals used
    to count the bigram baseline (0 = all); bootstrap = resamples of test individuals for the 95% CIs
    (0 = none); device = cpu/cuda for the mock backbone; ckpt_dir overrides the NeMo2 checkpoint
    folder stored in a --bionemo model."""
    import torch
    from evo2_model import load_finetuned
    meta = ctx.model.meta
    if not model_kind(meta).startswith("4a"):
        raise ValueError(f"test_set_performance needs a 4a model; this is {model_kind(meta)}")
    model, data = load_finetuned(meta, ctx.graph.dir, device or None, ckpt_dir or None)
    rng = ctx.rng

    def sample(examples, k):
        if k and len(examples) > k:
            return [examples[j] for j in np.sort(rng.choice(len(examples), k, replace=False))]
        return examples

    test_ex = sample(data.examples(ctx.test_idx), max_examples)
    trained = set(meta.get("train_individuals") or [])
    pos = {x: i for i, x in enumerate(ctx.graph.individuals)}
    tr_pool = np.array(sorted(pos[x] for x in trained if x in pos), dtype=np.int64) if trained else ctx.train_idx
    train_ex = sample(data.examples(tr_pool), train_examples)
    if not test_ex:
        raise ValueError("no test sequences with at least 2 clusters on the model's chromosomes")

    base_pool = ctx.train_idx
    if baseline_individuals and len(base_pool) > baseline_individuals:
        base_pool = np.sort(rng.choice(base_pool, baseline_individuals, replace=False))
    chroms = sorted({c for c, _ in test_ex})
    bigram, h_nll, h_n = {}, 0.0, 0
    for c in chroms:
        bigram[c], nll_c, n_c = bigram_table(data, c, base_pool)
        h_nll += nll_c; h_n += n_c
    # region of every local cluster index (clusters of one region are contiguous in genomic order)
    region_local = {c: torch.from_numpy(pd.factorize(ctx.graph.region_codes[data.candidates(c)])[0]).to(model.rnr.dev)
                    for c in chroms}

    def run(examples, with_baseline):
        rows, per_ind = {}, {}
        with torch.no_grad():
            for chrom, i in examples:
                logits, target = model.logits(chrom, i)
                k = min(top_k, logits.shape[1])
                nll = torch.nn.functional.cross_entropy(logits, target, reduction="sum").item()
                top = logits.topk(k, dim=-1).indices
                r = rows.setdefault(chrom, {"seq": 0, "n": 0, "nll": 0.0, "top1": 0, "topk": 0, "bigram": 0,
                                            "region_top1": 0, "rare": 0})
                r["rare"] += int(data.rare[data.candidates(chrom)[target.cpu().numpy()]].sum())
                r["seq"] += 1; r["n"] += len(target); r["nll"] += nll
                hit = int((top[:, 0] == target).sum())
                r["top1"] += hit; r["topk"] += int((top == target[:, None]).any(1).sum())
                big_hit = 0
                if with_baseline:
                    # region-conditioned: is the target the best-scored cluster of its own region?
                    reg = region_local[chrom]
                    n_reg = int(reg.max()) + 1
                    others = logits.scatter(1, target[:, None], float("-inf"))      # all but the target
                    rmax = torch.full((logits.shape[0], n_reg), float("-inf"), device=logits.device)
                    rmax = rmax.scatter_reduce(1, reg.expand_as(others), others, reduce="amax")
                    t_logit = logits.gather(1, target[:, None])[:, 0]
                    # strictly above every other cluster of its region (a tie is a miss)
                    r["region_top1"] += int((t_logit > rmax.gather(1, reg[target][:, None])[:, 0]).sum())
                    _, local = data.tokens(chrom, i)
                    s, e = model.window(len(local))
                    big_hit = int((bigram[chrom][local[s:e - 1]] == local[s + 1:e]).sum())
                    r["bigram"] += big_hit
                acc = per_ind.setdefault(i, [0, 0, 0])
                acc[0] += hit; acc[1] += big_hit; acc[2] += len(target)
        return rows, np.array(list(per_ind.values()), dtype=np.float64).reshape(-1, 3)

    (t_rows, t_ind), (tr_rows, _) = run(test_ex, True), run(train_ex, False)
    tot = lambda rows, key: sum(r[key] for r in rows.values())
    n_test, n_train = tot(t_rows, "n"), max(tot(tr_rows, "n"), 1)
    loss = tot(t_rows, "nll") / n_test
    top1, topk = tot(t_rows, "top1") / n_test, tot(t_rows, "topk") / n_test
    tr_top1 = tot(tr_rows, "top1") / n_train
    big = tot(t_rows, "bigram") / n_test
    region_top1 = tot(t_rows, "region_top1") / n_test
    floor = h_nll / h_n if h_n else float("nan")          # bigram reference cross-entropy
    cis = paired_bootstrap(t_ind, bootstrap, rng) if bootstrap > 0 and len(t_ind) > 1 else None

    path = ctx.out("test_set_performance_chroms.csv")
    pd.DataFrame([{"chrom": c, "test_sequences": r["seq"], "test_predictions": r["n"],
                   "test_loss": r["nll"] / r["n"], "test_top1": r["top1"] / r["n"],
                   f"test_top{top_k}": r["topk"] / r["n"], "test_region_top1": r["region_top1"] / r["n"],
                   "bigram_top1": r["bigram"] / r["n"]}
                  for c, r in t_rows.items()]).to_csv(path, index=False, float_format="%.6g")
    stored = (meta.get("metrics") or {})
    s = {
        "backend": meta.get("backend", "evo2"), "embeddings": meta.get("graph_method"),
        "test_individuals": len({i for _, i in test_ex}), "test_sequences": len(test_ex),
        "test_predictions": n_test,
        "test_loss": round(loss, 4), "test_perplexity": round(float(np.exp(min(loss, 50))), 2),
        "test_top1_accuracy": round(top1, 4),
        "test_top1_95ci": cis[0] if cis else None,
        f"test_top{top_k}_accuracy": round(topk, 4),
        "test_region_conditioned_top1": round(region_top1, 4),
        "test_targets_merged_rare_share": round(tot(t_rows, "rare") / n_test, 4),
        "train_sample_top1_accuracy": round(tr_top1, 4) if tr_rows else None,
        "generalization_gap_top1": round(tr_top1 - top1, 4) if tr_rows else None,
        "bigram_baseline_top1_accuracy": round(big, 4),
        "bigram_baseline_top1_95ci": cis[1] if cis else None,
        "model_minus_bigram_top1": round(top1 - big, 4),
        "model_minus_bigram_95ci": cis[2] if cis else None,
        "beats_bigram_baseline": ("yes" if cis[2][0] > 0 else "no" if cis[2][1] < 0 else "not significant")
                                 if cis else ("yes" if top1 > big else "no"),
        "bigram_reference_entropy": round(floor, 4),
        "test_loss_minus_bigram_reference": round(loss - floor, 4),
        "stored_val_top1_before_training": round(float((stored.get("before") or {}).get("accuracy", np.nan)), 4),
        "stored_val_top1_after_training": round(float((stored.get("after") or {}).get("accuracy", np.nan)), 4),
    }
    notes = []
    if meta.get("backend") == "mock":
        notes.append("MOCK backbone: this model is a small stand-in, not Evo 2; the numbers only show that "
                     "the pipeline runs")
    leaked = {x for x in (ctx.graph.individuals[i] for _, i in test_ex) if x in trained}
    if leaked:
        notes.append(f"WARNING: {len(leaked)} evaluated test individuals were in the model's training set")
    if s["beats_bigram_baseline"] != "yes":
        notes.append("the model does not beat the bigram baseline on the test set (paired bootstrap over test "
                     "individuals): it has not shown more than which cluster usually follows the current one")
    notes.append("bigram_reference_entropy is the plug-in H(next | current) of the training sequences (nats), "
                 "measured on the data it is counted from: a reference, not a floor. A model that uses more "
                 "context than the current cluster can go below it (negative test_loss_minus_bigram_reference "
                 "is good); far above it, the model does not even capture the bigram's predictability")
    notes.append("region-conditioned top-1 asks whether the target is the model's best cluster within the "
                 "target's own haploblock region: it separates 'which haplotype' from 'where on the chromosome'")
    if tr_rows and tr_top1 - top1 > 0.1:
        notes.append("training accuracy is much higher than test accuracy: likely overfitting")
    return {"summary": s, "notes": notes, "files": [path]}


def auc(scores, labels):
    """ROC AUC (Mann-Whitney U / n_pos n_neg), ties averaged; NaN if a class is missing."""
    labels = np.asarray(labels, bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binned_auc(scores, labels, dist, n_bins=4):
    """Mean AUC within genomic-distance quantile bins: compares pairs at similar distance only, so a
    score that merely encodes position cannot do well."""
    edges = np.unique(np.quantile(dist, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, max(len(edges) - 2, 0))
    vals = [auc(scores[b == k], labels[b == k]) for k in np.unique(b)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


@task("embedding_cooccurrence", "Do the node embeddings predict cluster co-occurrence in the TEST individuals?")
def embedding_cooccurrence(ctx, min_lift: float = 2.0, min_support: int = 5, min_expected: float = 5.0,
                           pairs: int = 50000, max_distance: int = 1_000_000, bootstrap: int = 200):
    """Held-out test of the step-3 embeddings, no GPU needed. Samples distinct pairs of clusters in
    DIFFERENT haploblock regions of the same chromosome, at most max_distance bp apart, each carried by
    at least min_support test individuals and with at least min_expected co-carriers expected under
    independence among the test individuals (so that one chance co-carrier of two rare clusters cannot
    make a pair 'linked'). A pair is positive when its test-individual lift is >= min_lift and negative
    when it is <= 1. Reports, per node_emb_<method>.npy in the graph folder, the ROC AUC of the
    embeddings' cosine similarity: overall, within genomic-distance bins and within expected-co-count
    (cluster frequency) bins, with a bootstrap CI that resamples haploblock regions (pairs sharing a
    region are not independent). References: genomic proximity, cluster frequency (expected co-count),
    and the lift counted on the TRAINING individuals."""
    import glob
    notes = []
    leaked = ctx.split_info.get("test_individuals_in_graph_training_set") or 0
    built_on_all = str(ctx.split_info.get("graph_built_on", "")).startswith("all individuals")
    if leaked or built_on_all:
        notes.append("WARNING: the graph (and so the embeddings) was built with test individuals "
                     f"({'all individuals' if built_on_all else f'{leaked:,} of them'}): this AUC is NOT held-out")
    nodes, C = ctx.graph.nodes, ctx.graph.carriers
    te, tr = ctx.test_idx, ctx.train_idx
    C_te = C[:, te].tocsr().astype(np.float32)
    C_tr = C[:, tr].tocsr().astype(np.float32)
    sup_te = np.asarray(C_te.sum(1)).ravel()
    sup_tr = np.asarray(C_tr.sum(1)).ravel()
    codes = ctx.graph.region_codes
    start = nodes["start"].to_numpy(dtype=np.float64)
    chrom = nodes["chrom"].to_numpy()
    rng = ctx.rng

    eligible = np.where((sup_te >= min_support) & np.isfinite(start))[0]
    if len(eligible) < 2:
        return {"summary": {"status": "skipped"}, "notes": notes + [
            f"skipped: fewer than 2 clusters with >= {min_support} test carriers"]}
    order = eligible[np.lexsort((start[eligible], chrom[eligible]))]
    key_c, key_s = chrom[order], start[order]
    s_idx, t_idx = [], []
    for c in pd.unique(key_c):        # a random eligible cluster + a random one within max_distance
        sel = np.where(key_c == c)[0]
        k = int(round(pairs * len(sel) / len(order)))
        if len(sel) < 2 or k == 0:
            continue
        cs = key_s[sel]
        i = rng.integers(0, len(sel), k)
        lo = np.searchsorted(cs, cs[i] - max_distance, side="left")
        hi = np.searchsorted(cs, cs[i] + max_distance, side="right")
        j = np.minimum(lo + (rng.random(k) * (hi - lo)).astype(np.int64), len(sel) - 1)
        s_idx.append(order[sel[i]]); t_idx.append(order[sel[j]])
    if not s_idx:
        return {"summary": {"status": "skipped"}, "notes": notes + ["skipped: no chromosome has two eligible clusters"]}
    s_idx, t_idx = np.concatenate(s_idx), np.concatenate(t_idx)
    pair = np.unique(np.stack([np.minimum(s_idx, t_idx), np.maximum(s_idx, t_idx)], 1), axis=0)   # distinct pairs
    s_idx, t_idx = pair[:, 0], pair[:, 1]
    n_te = len(te)
    expected = sup_te[s_idx] * sup_te[t_idx] / n_te
    keep = (codes[s_idx] != codes[t_idx]) & (expected >= min_expected)
    s_idx, t_idx, expected = s_idx[keep], t_idx[keep], expected[keep]

    def lift(Cm, sup, n):
        shared = np.asarray(Cm[s_idx].multiply(Cm[t_idx]).sum(1)).ravel()
        return shared * n / np.maximum(sup[s_idx] * sup[t_idx], 1e-12)

    lift_te, lift_tr = lift(C_te, sup_te, n_te), lift(C_tr, sup_tr, len(tr))
    pos, neg = lift_te >= min_lift, lift_te <= 1.0
    use = pos | neg
    s_idx, t_idx, labels = s_idx[use], t_idx[use], pos[use]
    expected, lift_tr = expected[use], lift_tr[use]
    dist = np.abs(start[s_idx] - start[t_idx])
    if labels.sum() == 0 or (~labels).sum() == 0:
        return {"summary": {"status": "skipped", "positive_pairs": int(labels.sum()),
                            "negative_pairs": int((~labels).sum())},
                "notes": notes + [f"skipped: need both positive (lift >= {min_lift}) and negative (lift <= 1) "
                                  "test pairs; lower min_lift, min_support or min_expected"]}
    block = codes[s_idx]                                          # bootstrap unit: haploblock region

    def evaluate(scores):
        res = {"auc": round(auc(scores, labels), 4),
               "auc_distance_matched": round(binned_auc(scores, labels, dist), 4),
               "auc_frequency_matched": round(binned_auc(scores, labels, expected), 4)}
        if bootstrap:
            blocks = np.unique(block)
            members = pd.Series(np.arange(len(block))).groupby(block).apply(np.array)
            boots = []
            for _ in range(bootstrap):
                take = np.concatenate(members.loc[rng.choice(blocks, len(blocks))].to_list())
                boots.append(auc(scores[take], labels[take]))
            lo_, hi_ = np.nanpercentile(boots, [2.5, 97.5])
            res["auc_95ci"] = (round(float(lo_), 4), round(float(hi_), 4))
        return res

    rows = [{"name": "genomic_proximity", "kind": "reference", **evaluate(-dist)},
            {"name": "cluster_frequency", "kind": "reference", **evaluate(np.log(expected))},
            {"name": "train_lift", "kind": "reference", **evaluate(lift_tr)}]
    files = sorted(glob.glob(os.path.join(ctx.graph.dir, "node_emb_*.npy")))
    gi = os.path.join(ctx.graph.dir, "graph_info.json")
    graph_sid = json.load(open(gi)).get("train_set_id", "all") if os.path.exists(gi) else "all"
    leaked_methods = []
    if not files:
        notes.append("no node_emb_<method>.npy in the graph folder (run 3_node_embeddings.py): references only")
    for f in files:
        name = re.sub(r"^node_emb_|\.npy$", "", os.path.basename(f))
        Z = np.load(f).astype(np.float32)
        if len(Z) != len(nodes):
            notes.append(f"{os.path.basename(f)} has {len(Z):,} rows but the graph has {len(nodes):,} nodes "
                         "(stale file from another graph build?): skipped")
            continue
        info_f = f[:-4] + ".json"
        emb_sid = json.load(open(info_f)).get("train_set_id", "?") if os.path.exists(info_f) else "?"
        if emb_sid != graph_sid:
            leaked_methods.append(name)
            notes.append(f"WARNING: node_emb_{name} was made from training set {emb_sid}, not the graph's "
                         f"({graph_sid}): its AUC is NOT held-out")
        Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        rows.append({"name": name, "kind": "embedding", **evaluate((Zn[s_idx] * Zn[t_idx]).sum(1))})

    path = ctx.out("embedding_cooccurrence.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    s = {"pairs": int(len(labels)), "positive_pairs": int(labels.sum()), "negative_pairs": int((~labels).sum()),
         "eligible_clusters": int(len(eligible)), "regions_in_pairs": int(len(np.unique(block))),
         "median_distance_bp": int(np.median(dist)), "held_out": "no" if (leaked or built_on_all or leaked_methods) else "yes"}
    for r in rows:
        for k in ("auc", "auc_distance_matched", "auc_frequency_matched", "auc_95ci"):
            if k in r:
                s[f"{k}_{r['name']}"] = r[k]
    notes += ["AUC 0.5 = no information about which clusters are co-inherited in individuals never seen; "
              "distance- and frequency-matched AUCs compare pairs of similar genomic distance / expected "
              "co-count only, so a score that merely encodes position or cluster frequency cannot do well",
              "references: genomic_proximity (position only), cluster_frequency (expected co-count only), "
              "train_lift (the linkage table of the training individuals)",
              "the 95% CI resamples haploblock regions, since pairs sharing a region are not independent"]
    return {"summary": s, "notes": notes, "files": [path]}


# ---- Nemotron summary ------------------------------------------------------

_SAMPLE_ID = re.compile(r"\b(HG|NA)\d{5}\b")
_DROP_KEYS = {"model_file", "gwas_file", "columns_used"}      # paths / user file contents


def _scrub(v, names=()):
    """Remove sample IDs (1000G-style and the graph's own individual names) and directories of file
    paths (POSIX or Windows, absolute or relative) before anything leaves the machine."""
    if isinstance(v, str):
        v = _SAMPLE_ID.sub("<sample>", v)
        v = re.sub(r"\S*[/\\]\S*", lambda m: re.split(r"[/\\]", m.group(0).rstrip("/\\"))[-1], v)
        for n in names:
            v = re.sub(rf"\b{re.escape(n)}\b", "<sample>", v)
        return v
    if isinstance(v, dict):
        return {k: _scrub(x, names) for k, x in v.items() if k not in _DROP_KEYS}
    if isinstance(v, (list, tuple)):
        return [_scrub(x, names) for x in v]
    return v


def nemotron_summary(args, results, out_path, names=()):
    import urllib.parse
    import urllib.request

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):     # never forward the API key to another location
            return None

    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        log("warning: --nemotron-summary needs $NVIDIA_API_KEY (from build.nvidia.com); skipped")
        return
    url = urllib.parse.urlparse(args.nemotron_url)
    if url.scheme != "https" and url.hostname not in ("localhost", "127.0.0.1", "::1"):
        log(f"warning: refusing to send the API key to {args.nemotron_url}: use https (or a local NIM); skipped")
        return
    names = [n for n in names if not _SAMPLE_ID.fullmatch(n)]   # 1000G IDs are already covered
    payload = _scrub([{"task": r["name"], "status": r["status"], "summary": r.get("summary", {}),
                       "notes": r.get("notes", [])} for r in results], names)
    prompt = ("These are validation results of an Evo 2 DNA language model fine-tuned on haplograph node "
              "embeddings (1000 Genomes haploblock clusters) to predict the next haploblock cluster along a "
              "chromosome, evaluated on held-out test individuals. Explain in plain language, in at most "
              "250 words: does the model generalize to the test set, does it beat the bigram baseline, is "
              "there evidence of overfitting or train/test leakage, and what should be checked next. Do not "
              "invent numbers.\n\n" + json.dumps(payload, indent=1, default=str))
    body = json.dumps({"model": args.nemotron_model, "temperature": 0.2, "max_tokens": 1024,
                       "messages": [{"role": "system", "content": "You are a careful genomics data scientist."},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(args.nemotron_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "Accept": "application/json"})
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=180) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        log(f"warning: Nemotron request failed ({type(e).__name__}: {e}); no summary written")
        return
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # reasoning models may include it
    with open(out_path, "w") as f:
        f.write(f"# Nemotron summary ({args.nemotron_model})\n\n"
                "_Generated by a language model from the aggregate numbers in report.md; check it against "
                "the report._\n\n" + text + "\n")
    log(f"Nemotron summary: {out_path}")


def check_self_test(summaries, hotspot):
    """Properties the synthetic self-test must satisfy (see IMPROVEMENT_PLAN.md, testing steps)."""
    problems = []

    def ci_ok(value, ci, what):
        if ci is not None and not (ci[0] - 1e-9 <= value <= ci[1] + 1e-9):
            problems.append(f"{what} {value} outside its 95% CI {ci}")

    t = summaries["test_set_performance"]
    ci_ok(t["test_top1_accuracy"], t["test_top1_95ci"], "test top-1")
    ci_ok(t["bigram_baseline_top1_accuracy"], t["bigram_baseline_top1_95ci"], "bigram top-1")
    ci_ok(t["model_minus_bigram_top1"], t["model_minus_bigram_95ci"], "model - bigram")
    if t["test_region_conditioned_top1"] + 1e-9 < t["test_top1_accuracy"]:
        problems.append("region-conditioned top-1 is below unconditioned top-1")
    e = summaries["embedding_cooccurrence"]
    if e.get("status") == "skipped":
        problems.append("embedding_cooccurrence was skipped on the synthetic data")
    else:
        if not 0.3 <= e["auc_genomic_proximity"] <= 0.7:   # the synthetic data has no positional linkage
            problems.append(f"genomic proximity AUC {e['auc_genomic_proximity']} should be near 0.5")
        if e["auc_train_lift"] <= 0.55:                    # planted group structure replicates in test
            problems.append(f"train-lift AUC {e['auc_train_lift']} should be clearly above 0.5")
        if e["auc_gnn"] <= 0.5:
            problems.append(f"GNN AUC {e['auc_gnn']} should be above 0.5")
        for k, v in e.items():
            if k.startswith("auc_95ci_"):
                ci_ok(e["auc_" + k[len("auc_95ci_"):]], v, k)
    g, c = summaries["gwas_enrichment:independent"], summaries["gwas_enrichment:clumped"]
    if g["top_region"] != hotspot or g["regions_enriched_q<0.05"] < 1:      # positive control
        problems.append(f"gwas_enrichment missed the 10 loci planted in {hotspot} (top region {g['top_region']})")
    rot, fdr = g["regions_q<0.05_under_rotation"], g["empirical_fdr"]
    if rot is None or rot < 0.5 or (fdr is not None and fdr >= 0.2):         # cluster seen, loci not chance
        problems.append(f"gwas_enrichment nulls: rotation {rot}, FDR {fdr}")
    if c["regions_enriched_q<0.05"] != 0:              # clumped, the planted hits are one locus
        problems.append(f"gwas_enrichment with clump={c['clump_bp']} still calls {c['regions_enriched_q<0.05']} "
                        "haploblocks enriched")
    if problems:
        die("self-test checks failed:\n  " + "\n  ".join(problems))
    log("self-test checks passed")


def self_test(args):
    """Synthetic data -> step 4 --test (mock backbone) -> validate it with every data-level task."""
    importlib.import_module("4a_finetune_evo2_embeddings").main(["--test"])
    args.graph_dir = selftest_graph_dir()
    gwas, args.hotspot = planted_gwas(args.graph_dir, test_workdir("validate_gwas"))
    args.model = os.path.join(args.graph_dir, "evo2_mock_next_gnn.pt")
    args.splits, args.tasks, args.gwas, args.out_dir = GRAPH_SPLIT, None, [], None
    args.task = ["model_info", "test_set_performance max_examples=40 train_examples=20 bootstrap=200 device=cpu",
                 "embedding_cooccurrence min_lift=1.5 min_support=3 min_expected=1 pairs=5000 bootstrap=50",
                 "cluster_consistency permutations=20",
                 f"gwas_enrichment gwas={shlex.quote(gwas)} clump=500 label=independent permutations=200",
                 f"gwas_enrichment gwas={shlex.quote(gwas)} label=clumped permutations=200"]


def planted_gwas(graph_dir, workdir):
    """Synthetic catalog with 10 hits ~1.8 kb apart in the middle of the graph's longest chr21
    haploblock, filtered by 5_download_gwas.py. Returns (hit file, that haploblock)."""
    dg = importlib.import_module("5_download_gwas")
    reg = pd.read_csv(os.path.join(graph_dir, "nodes.csv.gz"), usecols=["region", "chrom", "start", "end"])
    r = reg[reg["chrom"] == "chr21"].assign(n=lambda d: d["end"] - d["start"]).sort_values("n").iloc[-1]
    assert r.n > 16_000, f"self-test graph: longest chr21 haploblock {r.region} is too short to plant 10 hits"
    mid = (r.start + r.end) // 2
    dg.make_test_catalog(os.path.join(workdir, "catalog.zip"), ("chr21", mid - 8000, mid + 8000))
    tsv = dg.main(["--workdir", str(workdir), "--catalog", os.path.join(workdir, "catalog.zip"),
                   "--trait-id", "EFO_0000001", "--label", "planted"])
    return tsv, r.region


# ===========================================================================
# Runner and report
# ===========================================================================

def write_report(ctx, results, path):
    lines = ["# Evo 2 validation report", "",
             f"- model: `{ctx.model.path}`",
             f"- graph: `{ctx.graph.dir}`",
             f"- train: {len(ctx.train_idx):,} individuals ({ctx.split_info.get('train_source')})",
             f"- test: {len(ctx.test_idx):,} individuals ({ctx.split_info.get('test_source')})",
             f"- graph individuals in neither set: {ctx.split_info.get('graph_individuals_in_neither_set', 0):,}",
             f"- graph built on: {ctx.split_info.get('graph_built_on', 'unknown')}"
             + (f" ({ctx.split_info['test_individuals_in_graph_training_set']:,} of them are test individuals)"
                if ctx.split_info.get("test_individuals_in_graph_training_set") else ""),
             f"- created: {time.strftime('%Y-%m-%d %H:%M')}", ""]
    for r in results:
        params = " ".join(f"{k}={v}" for k, v in r["params"].items())
        lines += [f"## {r['name']}" + (f" ({params})" if params else ""), ""]
        if r["status"] != "ok":
            lines += [f"**{r['status']}**: {r.get('error', '')}", ""]
            continue
        lines += ["| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in r["summary"].items()] + [""]
        lines += [f"- {n}" for n in r.get("notes", [])]
        lines += [f"- output: `{os.path.basename(f)}`" for f in r.get("files", [])] + [""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def list_tasks():
    for name, t in TASKS.items():
        print(f"{name}: {t['description']}")
        for p in task_params(t["fn"]):
            typ = getattr(p.annotation, "__name__", "str") if p.annotation is not p.empty else "str"
            default = "required" if p.default is p.empty else f"default {p.default!r}"
            print(f"    {p.name} ({typ}, {default})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="fine-tuned model file from step 4 (.pt)")
    ap.add_argument("--graph-dir", default="haplograph/genome_train_half_A", help="output folder of 2_build_graph.py")
    sg = ap.add_argument_group("train / test split")
    sg.add_argument("--splits", default=GRAPH_SPLIT,
                    help="'graph' (default) = the graph folder's train/test_individuals.txt; a folder or URL "
                         f"with the split files, e.g. {BH26_SPLITS}/; 'none' = model's held-out set")
    sg.add_argument("--train-file", default="half_A.txt", help="train file name inside --splits")
    sg.add_argument("--test-file", default="half_B.txt", help="test file name inside --splits")
    sg.add_argument("--train-individuals", help="explicit train file or URL (overrides --splits)")
    sg.add_argument("--test-individuals", help="explicit test file or URL (overrides --splits)")
    sg.add_argument("--sample-map", help="CSV: graph individual name, sample ID (if they differ)")
    ap.add_argument("--tasks", help="tasks file, one task per line")
    ap.add_argument("--task", action="append", default=[], help='one task line, e.g. "gwas_enrichment gwas=x.gz"')
    ap.add_argument("--gwas", action="append", default=[], help="shortcut for 'gwas_enrichment gwas=FILE'")
    ap.add_argument("--plugins", action="append", default=[], help="python file(s) registering extra tasks")
    ap.add_argument("--out-dir", help="default <graph-dir>/validation_<model file name>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list-tasks", action="store_true", help="print available tasks and their parameters")
    ap.add_argument("--cudf", action="store_true",
                    help="run the pandas parts (e.g. reading GWAS files) on the GPU with RAPIDS cuDF")
    ng = ap.add_argument_group("Nemotron summary (optional)")
    ng.add_argument("--nemotron-summary", action="store_true",
                    help="ask an NVIDIA Nemotron model (NIM API, key in $NVIDIA_API_KEY) to interpret the report")
    ng.add_argument("--nemotron-model", default="nvidia/llama-3.3-nemotron-super-49b-v1")
    ng.add_argument("--nemotron-url", default="https://integrate.api.nvidia.com/v1",
                    help="OpenAI-compatible endpoint (build.nvidia.com or a self-hosted NIM)")
    ap.add_argument("--test", action="store_true",
                    help="self-test: synthetic data, a mock-backbone model from step 4 --test, all tasks (CPU)")
    args = ap.parse_args(argv)
    if args.test:
        self_test(args)

    for p in args.plugins:
        load_plugin(p)
    if args.list_tasks:
        list_tasks()
        return
    if not args.model:
        ap.error("--model is required")
    if not os.path.exists(args.model):
        raise SystemExit(f"error: model file {args.model} not found - run 4a_finetune_evo2_embeddings.py "
                         "(or 4b_finetune_evo2_hash.py) first")

    model = Model(args.model)                     # loaded once, on first use, shared with the tasks
    try:
        specs = read_tasks_file(args.tasks) if args.tasks else []
        specs += [parse_task_line(t, "--task") for t in args.task]
    except (OSError, ValueError) as e:
        die(f"bad task list: {e}")
    if not specs:
        specs = [("model_info", {})]
        if model.kind.startswith("4a"):
            specs.append(("test_set_performance", {}))
        if glob.glob(os.path.join(args.graph_dir, "node_emb_*.npy")):
            specs.append(("embedding_cooccurrence", {}))
        specs.append(("cluster_consistency", {}))
    specs += [("gwas_enrichment", {"gwas": g}) for g in args.gwas]
    jobs = []
    for name, raw in specs:
        if name not in TASKS:
            raise SystemExit(f"unknown task {name!r}; available: {', '.join(TASKS)} (see --list-tasks)")
        try:
            jobs.append((name, convert_params(name, TASKS[name]["fn"], raw)))
        except ValueError as e:
            raise SystemExit(str(e))

    stem = re.sub(r"\.pt$", "", os.path.basename(args.model))
    args.out_dir = args.out_dir or os.path.join(args.graph_dir, f"validation_{stem}")
    os.makedirs(args.out_dir, exist_ok=True)
    ctx = Context(args, model)
    ctx.setup_split()

    results = []
    seen = {}
    for name, params in jobs:
        log(f"[{name}] " + " ".join(f"{k}={v}" for k, v in params.items()))
        # each task gets its own generator, so adding / changing one task never shifts another's results
        seen[name] = seen.get(name, 0) + 1
        ctx.rng = np.random.default_rng([args.seed, zlib.crc32(name.encode()), seen[name]])
        t0 = time.time()
        rec = {"name": name, "params": params}
        try:
            out = TASKS[name]["fn"](ctx, **params) or {}
            rec.update(status="ok", summary=out.get("summary", {}), notes=out.get("notes", []),
                       files=out.get("files", []))
            for k, v in rec["summary"].items():
                log(f"    {k}: {v}")
            for n in rec["notes"]:
                log(f"    note: {n}")
        except SystemExit:
            raise
        except Exception as e:
            rec.update(status="failed", error=f"{type(e).__name__}: {e}")
            log(f"    FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stdout)
        rec["seconds"] = round(time.time() - t0, 1)
        results.append(rec)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"model": args.model, "graph_dir": args.graph_dir,
                   "split": ctx.split_info,
                   "tasks": results}, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_report(ctx, results, os.path.join(args.out_dir, "report.md"))
    n_fail = sum(r["status"] != "ok" for r in results)
    log(f"report: {os.path.join(args.out_dir, 'report.md')}"
        + (f"  ({n_fail} task(s) failed)" if n_fail else ""))
    if args.nemotron_summary:
        nemotron_summary(args, results, os.path.join(args.out_dir, "nemotron_summary.md"), ctx.graph.individuals)
    if args.test:
        if n_fail:
            die(f"self-test: {n_fail} task(s) failed")
        check_self_test({r["name"] + (":" + r["summary"]["label"] if r["name"] == "gwas_enrichment" else ""):
                         r["summary"] for r in results}, args.hotspot)
    elif n_fail:                                  # a chained run (set -e, CI) must not report success
        die(f"{n_fail} task(s) failed - see {os.path.join(args.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
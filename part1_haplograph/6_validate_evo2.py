"""
Step 6 - validate fine-tuned models (steps 4a / 4b) on the held-out TEST individuals.

A small task runner: the checks to run are listed one per line in a tasks file (validation_tasks.txt;
`--list-tasks` shows every task and its parameters); new checks go in a --plugins file using the
`task` decorator (example and `ctx` fields: DOCUMENTATION.md section 7). --model can be repeated:
model_comparison uses every model, the other tasks the first (or their model= parameter).
  model_info              model type and settings, stored metrics, train/test leakage
  test_set_performance    a next-token 4a model on test haplotypes: loss, top-1/top-5 against frequency,
                          bigram, trigram and Li-Stephens HMM baselines (identity-free and bigram for
                          individual units), paired bootstrap CIs over test individuals, by stratum
  head_ceiling            the lowest cross-entropy the model's output head can reach
  masked_imputation       a masked 4a model imputing masked haploblocks of test individuals, against the
                          HMM, a flank lookup and the frequency (concordance, dosage r^2 by frequency)
  model_comparison        every --model on the same test examples, paired CIs between every pair
  embedding_cooccurrence  do the step-3 embeddings predict which clusters share a test haplotype? (vs
                          carrier_svd, references, pairs without graph edges)
  cluster_consistency     do train and test carry the same clusters per haploblock? (permutation null);
                          related pairs across the split
  gwas_enrichment         haploblocks with more independent GWAS loci than expected (hits clumped by
                          distance), chance levels from randomly placed loci (empirical FDR) and from
                          rotations that keep the hits' spacing, and whether nearby hits share a
                          haploblock (GRCh38; columns auto-detected, see GWAS_COLS; 5_download_gwas.py)
  model_gwas              a 4a model's per-haploblock scores at GWAS-hit haploblocks vs matched background
  phenotype_prediction    a 4b model's test R^2 vs ridge on cluster dosages; causal-haploblock recovery
  junction_strength       next-haploblock unpredictability (data, model) vs recombination rate

Split: by default the graph's train_individuals.txt / test_individuals.txt (steps 1-2); --splits
DIR_OR_URL (half_A.txt / half_B.txt), --train/--test-individuals, or --splits none (the model's own
held-out individuals). Split files: one sample ID per line, or a CSV with an "individual" column.

OUTPUT (--out-dir, default <graph-dir>/validation_<first model name>/)
  report.md, summary.json, split.csv (individual, set)
  test_set_performance_strata.csv     stratum, value, predictions, model_top1, model_top1_95ci,
                                      <baseline>_top1, model_minus_<baseline>_top1_95ci,
                                      model_minus_<baseline>_loss_95ci (haplotype units), model_loss
  masked_imputation.csv               span, <method>_concordance, model_minus_<method>_95ci, <method>_r2_<bin>
  model_comparison.csv / _pairs.csv   per model: top1, loss; per pair: differences with 95% CIs
  embedding_cooccurrence.csv          name, kind, auc, auc_distance_matched, auc_frequency_matched,
                                      auc_no_graph_edges, auc_95ci, auc_minus_carrier_svd(_95ci)
  cluster_consistency_regions.csv.gz  region, chrom, start, end, n_clusters_total/_train/_test,
                                      test_expected, test_ci_low/high, outside_ci, n_test_only,
                                      test_only_carriers, tvd, tvd_null_mean, tvd_p, chi2, chi2_df, chi2_p, chi2_q
  gwas_enrichment_<label>_regions.csv.gz   region, chrom, start, end, n_snps, n_hits, n_loci,
                                      expected_loci, fold, p, q, lead_snp, lead_p
  model_gwas.csv                      score, hit_mean, matched_mean, difference(_95ci), p_random_loci, p_rotation
  nemotron_summary.md                 --nemotron-summary: an NVIDIA Nemotron reading of the aggregate
                                      numbers ($NVIDIA_API_KEY; IDs and paths removed, https only)
  (a task listed twice writes its files with a _2, _3 ... suffix)
A failed task makes the script exit with code 1 after the report is written.

USAGE
  python 6_validate_evo2.py --model $G/evo2_next_gnn.pt --graph-dir $G --tasks validation_tasks.txt
  python 6_validate_evo2.py --model $G/evo2_next_gnn.pt --graph-dir $G --gwas haplograph/gwas/height.tsv
  python 6_validate_evo2.py --model $G/evo2_next_gnn.pt --model $G/evo2-random_next_gnn.pt \
      --model $G/small_next_gnn.pt --task model_comparison     # does pretraining help?
  python 6_validate_evo2.py --model $G/evo2_masked_gnn.pt --graph-dir $G --task masked_imputation
  python 6_validate_evo2.py --model $G/evo2_pheno_cis_gnn.pt --graph-dir $G --task phenotype_prediction
  python 6_validate_evo2.py --test                       # synthetic data, small backbone, CPU
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

from haplo_common import (BH26_SPLITS, die, download, is_url, load_sample_map, log, read_graph_individuals,
                          read_individual_list, selftest_graph_dir, test_workdir, url_cache_path)


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
    if meta.get("mode") == "phenotype":
        return f"4b phenotype ({meta.get('input')} input)"
    if meta.get("mode") == "embeddings" or "proj_in" in meta:
        backend = meta.get("backend", "evo2")
        return "4a embeddings" + {"bionemo": " (BioNeMo)", "small": " (small backbone trained from scratch, not Evo 2)"
                                  }.get(backend, "") + (" (random init)" if meta.get("init") == "random" else "")
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
        self.cache = {}                                  # loaded models, shared by the tasks
        self.models = [self.model] + [Model(f) for f in args.model[1:]]

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
        for k in ("backend", "init", "units", "objective", "head", "softmax", "graph_method", "input", "task", "trait",
                  "trainable_params", "model_size", "layer_index"):
            if meta.get(k) is not None:
                s[k] = meta[k]
        for k in ("evo2_model", "layer", "train_blocks"):
            if (meta.get("config") or {}).get(k) is not None and meta.get("backend") != "small":
                s[k] = meta["config"][k]
        if meta.get("chroms"):
            s["trained_chromosomes"] = ",".join(meta["chroms"])
            missing = set(meta["chroms"]) - set(ctx.graph.nodes["chrom"].unique())
            if missing:
                notes.append(f"WARNING: model was trained on chromosomes not in this graph: {sorted(missing)}")
        tb = meta.get("trained_blocks") or {}
        s["stored_backbone_params"] = int(sum(getattr(t, "numel", lambda: 0)() for t in tb.values()))
        metrics = meta.get("metrics") or {}
        for phase in ("before", "best"):
            for k, v in (metrics.get(phase) or {}).items():
                if isinstance(v, (int, float)):
                    s[f"stored_val_{k}_{phase}"] = round(float(v), 4)
        if "best_step" in metrics:
            s["best_step"] = metrics["best_step"]
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
        elif meta.get("input") == "hash":
            notes.append("hash-input 4b models are trained on hash-phenotype rows, not individuals; their held-out "
                         "rows are stored as val_rows (indices into the training TSV)")
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
    rel = os.path.join(ctx.graph.dir, "related_pairs.tsv")
    if os.path.exists(rel):
        pairs = pd.read_csv(rel, sep="\t", dtype=str)
        tr_n, te_n = ({ctx.graph.individuals[i] for i in x} for x in (tr, te))
        across = sum(1 for a, b in zip(pairs["individual_a"], pairs["individual_b"])
                     if (a in tr_n and b in te_n) or (a in te_n and b in tr_n))
        s["related_pairs"], s["related_pairs_across_split"] = len(pairs), across
    per_chrom = regions.groupby("chrom", sort=False)["outside_ci"].mean()
    worst = per_chrom.sort_values(ascending=False).head(3)
    notes = ["chromosomes with most regions outside the interval: "
             + ", ".join(f"{c} {v:.1%}" for c, v in worst.items())]
    if s.get("related_pairs_across_split"):
        notes.append(f"WARNING: {s['related_pairs_across_split']} related pairs (1000 Genomes pedigree) straddle the "
                     "split: relatives share long haplotypes, so the test set is not fully independent")
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

def ratio_ci(num, den, idx):
    """95% interval of sum(num) / sum(den) over bootstrap resamples idx [n_boot, n] of individuals."""
    v = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return (round(float(lo), 4), round(float(hi), 4))


def verdict(ci):
    return "yes" if ci and ci[0] > 0 else "no" if ci and ci[1] < 0 else "not significant"


def query_model(ctx, model_file=None, device="", ckpt_dir=""):
    """(HaploModel, GenomeData, meta) of a 4a checkpoint, cached on ctx."""
    from evo2_model import load_finetuned
    m = ctx.model if model_file is None else Model(model_file)
    key = (m.path, device, ckpt_dir)
    if key not in ctx.cache:
        if not model_kind(m.meta).startswith("4a"):
            raise ValueError(f"{m.path} is {model_kind(m.meta)}; this task needs a 4a model")
        model, data = load_finetuned(m.meta, ctx.graph.dir, device or None, ckpt_dir or None)
        ctx.cache[key] = (model, data, m.meta)
    return ctx.cache[key]


def sample_examples(data, individuals, k, rng):
    ex = data.examples(individuals)
    return ex if not k or len(ex) <= k else [ex[j] for j in np.sort(rng.choice(len(ex), k, replace=False))]


def block_tokens(data, chrom, rows):
    """Block-local tokens [len(rows), haploblocks of chrom] (haplotype units) and tokens per block."""
    cols = data.block_cols[chrom]
    tok = data.canon[data.H[np.asarray(rows)][:, cols]]
    reg = data.region_code[data.canon[data.H[0, cols]]]
    return data.block_pos[tok], data.block_len[reg]


def hap_rows(individuals):
    return np.ravel([[2 * i, 2 * i + 1] for i in individuals]).astype(np.int64)


def individual_baselines(data, chrom, train_inds):
    """Individual units: next-token predictors counted on the training individuals' sequences:
    identity_free = the most frequent next token given the current token's haploblock and slot (1st /
    2nd cluster of the individual in that haploblock), bigram = given the current token."""
    nxt = {"identity_free": {}, "bigram": {}}
    for i in train_inds:
        t = data.tokens(chrom, i)
        reg = data.region_code[t]
        slot = np.r_[0, (reg[1:] == reg[:-1]).astype(int)]
        for key, name in (((reg[:-1] * 2 + slot[:-1]), "identity_free"), (t[:-1], "bigram")):
            for a, b in zip(key, t[1:]):
                d = nxt[name].setdefault(int(a), {})
                d[int(b)] = d.get(int(b), 0) + 1
    best = {n: {a: max(d, key=d.get) for a, d in v.items()} for n, v in nxt.items()}

    def predict(t):
        reg = data.region_code[t]
        slot = np.r_[0, (reg[1:] == reg[:-1]).astype(int)]
        return {"identity_free": np.array([best["identity_free"].get(int(k), -1) for k in reg[:-1] * 2 + slot[:-1]]),
                "bigram": np.array([best["bigram"].get(int(k), -1) for k in t[:-1]])}
    return predict


def bigram_entropy(data, chrom, train_inds):
    """Plug-in H(next | current) of the training sequences: (sum of -log p, number of transitions)."""
    keys = []
    for ex in data.examples(train_inds):
        if ex[0] == chrom:
            t = data.local[data.tokens(*ex)].astype(np.int64)
            keys.append(t[:-1] * data.n_nodes + t[1:])
    if not keys:
        return 0.0, 0
    keys, counts = np.unique(np.concatenate(keys), return_counts=True)
    cur = keys // data.n_nodes
    per_cur = pd.Series(counts).groupby(cur).transform("sum").to_numpy()
    return float(-(counts * np.log(counts / per_cur)).sum()), int(counts.sum())


def freq_bin(f):
    return np.select([f < 0.01, f < 0.05, f < 0.2], ["<1%", "1-5%", "5-20%"], ">20%")


def populations(ctx):
    f = os.path.join(ctx.graph.dir, "populations.tsv")
    if not os.path.exists(f):
        return None
    pop = pd.read_csv(f, sep="\t", dtype=str).set_index("individual")["superpopulation"]
    return np.array([pop.get(x, "unknown") for x in ctx.graph.individuals])


def score_next(model, data, examples, train_inds):
    """Model and baselines on every next-token prediction of `examples`. Returns a DataFrame with one
    row per prediction: individual, chrom, block, target, rare, freq, length, model_top1/_top5/_logp
    and <baseline>_top1 (and _logp for haplotype units)."""
    from haplo_baselines import NEXT_BASELINES, BlockBaselines
    rows, by_chrom = [], {}
    for c, ex in examples:
        by_chrom.setdefault(c, []).append(ex)
    for chrom, exs in by_chrom.items():
        if data.units == "haplotype":
            tr = hap_rows(train_inds)
            R, K = block_tokens(data, chrom, tr)
            bl = BlockBaselines(R, K)
            Y, _ = block_tokens(data, chrom, exs)
            base = {n: bl.predict_next(Y, n) for n in NEXT_BASELINES}
            freq = [f / f.sum() for f in bl.freq]
        else:
            predict = individual_baselines(data, chrom, train_inds)
        for j, ex in enumerate(exs):
            s = model.score(chrom, ex)
            d = {"individual": ex // 2 if data.units == "haplotype" else ex, "chrom": chrom,
                 "target": s["target"], "block": data.region_code[s["target"]], "rare": data.rare[s["target"]],
                 "model_top1": s["top1"], "model_top5": s["top5"], "model_logp": s["logp"]}
            if data.units == "haplotype":
                pos = s["pos"]
                y = Y[j, pos]
                d["freq"] = np.array([freq[p][t] for p, t in zip(pos, y)])
                for n in NEXT_BASELINES:
                    top, pt = base[n]
                    d[f"{n}_top1"] = top[j, pos - 1] == y
                    d[f"{n}_logp"] = np.log(pt[j, pos - 1])
            else:
                t = data.tokens(chrom, ex)
                pr = predict(t)
                d["freq"] = data.support[s["target"]] / max(len(train_inds), 1)
                d["same_block"] = data.region_code[t[s["pos"] - 1]] == d["block"]
                for n, p in pr.items():
                    d[f"{n}_top1"] = p[s["pos"] - 1] == t[s["pos"]]
            rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


@task("test_set_performance", "Query the fine-tuned model on the test haplotypes / individuals: next-token "
                              "prediction against frequency, n-gram and HMM baselines, by stratum")
def test_set_performance(ctx, max_examples: int = 500, train_examples: int = 200, bootstrap: int = 1000,
                         model: str = "", device: str = "", ckpt_dir: str = ""):
    """max_examples / train_examples = (haplotype or individual, chromosome) sequences evaluated from the
    test set / from the model's training individuals (0 = all); bootstrap = resamples of test
    individuals for the 95% CIs (paired: the same resamples for the model and every baseline); device =
    cpu/cuda for a small backbone; ckpt_dir overrides the NeMo2 checkpoint folder of a bionemo model;
    model = another --model file (default: the first). Baselines are counted on the graph's training
    individuals."""
    model, data, meta = query_model(ctx, model or None, device, ckpt_dir)
    if meta["objective"] != "next":
        raise ValueError("test_set_performance scores next-token models; use masked_imputation for a masked model")
    rng = ctx.rng
    test_ex = sample_examples(data, ctx.test_idx, max_examples, rng)
    if not test_ex:
        raise ValueError("no test sequences with at least 2 tokens on the model's chromosomes")
    pos = {x: i for i, x in enumerate(ctx.graph.individuals)}
    tr_pool = np.array(sorted(pos[x] for x in meta["train_individuals"] if x in pos), dtype=np.int64)
    df = score_next(model, data, test_ex, ctx.train_idx)
    tr = score_next(model, data, sample_examples(data, tr_pool, train_examples, rng), ctx.train_idx) \
        if train_examples and len(tr_pool) else None
    names = [c[:-5] for c in df.columns if c.endswith("_top1") and c != "model_top1"]
    with_ce = data.units == "haplotype" and meta["softmax"] == "block"

    inds = df["individual"].to_numpy()
    uniq, ind_code = np.unique(inds, return_inverse=True)
    per = lambda col: np.bincount(ind_code, weights=df[col].astype(float), minlength=len(uniq))
    count = np.bincount(ind_code, minlength=len(uniq)).astype(float)
    idx = rng.integers(0, len(uniq), (bootstrap, len(uniq))) if bootstrap and len(uniq) > 1 else None
    ci = lambda num: ratio_ci(num, count, idx) if idx is not None else None
    top1 = float(df["model_top1"].mean())
    loss = float(-df["model_logp"].mean())
    s = {"backend": meta["backend"], "init": meta["init"], "units": data.units, "head": meta["head"],
         "softmax": meta["softmax"], "embeddings": meta["graph_method"],
         "test_individuals": len(uniq), "test_sequences": len(test_ex), "test_predictions": len(df),
         "test_loss": round(loss, 4), "test_perplexity": round(float(np.exp(min(loss, 50))), 2),
         "test_top1_accuracy": round(top1, 4), "test_top1_95ci": ci(per("model_top1")),
         "test_top5_accuracy": round(float(df["model_top5"].mean()), 4),
         "test_targets_merged_rare_share": round(float(df["rare"].mean()), 4)}
    beats = []
    for n in names:
        s[f"{n}_top1_accuracy"] = round(float(df[f"{n}_top1"].mean()), 4)
        d_ci = ci(per("model_top1") - per(f"{n}_top1"))
        s[f"model_minus_{n}_top1"] = round(top1 - s[f"{n}_top1_accuracy"], 4)
        s[f"model_minus_{n}_top1_95ci"] = d_ci
        if with_ce:
            s[f"{n}_loss"] = round(float(-df[f"{n}_logp"].mean()), 4)
            s[f"model_minus_{n}_loss_95ci"] = ci(per(f"{n}_logp") - per("model_logp"))
        if verdict(d_ci) == "yes":
            beats.append(n)
    s["beats_baselines"] = ",".join(beats) or "none"
    s["beats_every_baseline"] = "yes" if len(beats) == len(names) else "no"
    if tr is not None:
        s["train_sample_top1_accuracy"] = round(float(tr["model_top1"].mean()), 4)
        s["generalization_gap_top1"] = round(s["train_sample_top1_accuracy"] - top1, 4)
    h_nll, h_n = map(sum, zip(*(bigram_entropy(data, c, ctx.train_idx) for c in sorted(set(df["chrom"])))))
    s["bigram_reference_entropy"] = round(h_nll / h_n, 4) if h_n else None
    t_all = df["target"].to_numpy()
    floor = head_floor(model, data, t_all[rng.choice(len(t_all), min(2000, len(t_all)), replace=False)])
    s["head_ce_floor"] = round(floor, 4) if floor is not None else None
    s["stored_best_val_top1"] = round(float(meta["metrics"]["best"]["accuracy"]), 4)

    # strata
    pops = populations(ctx)
    df["superpopulation"] = pops[df["individual"]] if pops is not None else "unknown"
    starts = ctx.graph.regions.set_index("region")
    length = (starts["end"] - starts["start"]).reindex(data.regions).to_numpy()[df["block"]]
    df["block_length"] = pd.qcut(length, 3, labels=["short", "medium", "long"], duplicates="drop").astype(str) \
        if len(np.unique(length)) >= 3 else "all"
    df["target_frequency"] = freq_bin(df["freq"].to_numpy())
    df["merged_rare_target"] = np.where(df["rare"], "yes", "no")
    kinds = ["target_frequency", "superpopulation", "block_length", "merged_rare_target"]
    if "same_block" in df:
        df["target_block"] = np.where(df["same_block"], "same haploblock", "next haploblock")
        kinds.append("target_block")
    strata = []
    for kind in kinds:
        for value, g in df.groupby(kind):
            gi = np.searchsorted(uniq, g["individual"].to_numpy())
            cnt = np.bincount(gi, minlength=len(uniq)).astype(float)
            row = {"stratum": kind, "value": value, "predictions": len(g),
                   "model_top1": round(float(g["model_top1"].mean()), 4),
                   "model_top1_95ci": ratio_ci(np.bincount(gi, weights=g["model_top1"].astype(float),
                                                           minlength=len(uniq)), cnt, idx) if idx is not None else None}
            per_g = lambda col: np.bincount(gi, weights=g[col].astype(float), minlength=len(uniq))
            for n in names:
                row[f"{n}_top1"] = round(float(g[f"{n}_top1"].mean()), 4)
                if idx is not None:
                    row[f"model_minus_{n}_top1_95ci"] = ratio_ci(per_g("model_top1") - per_g(f"{n}_top1"), cnt, idx)
                if with_ce and idx is not None:
                    row[f"model_minus_{n}_loss_95ci"] = ratio_ci(per_g(f"{n}_logp") - per_g("model_logp"), cnt, idx)
            if with_ce:
                row["model_loss"] = round(float(-g["model_logp"].mean()), 4)
            strata.append(row)
    path = ctx.out("test_set_performance_strata.csv")
    pd.DataFrame(strata).to_csv(path, index=False)

    notes = []
    if meta["backend"] == "small":
        notes.append("small backbone: a transformer trained from scratch, not Evo 2")
    leaked = set(meta["train_individuals"]) & {ctx.graph.individuals[i] for i in uniq}
    if leaked:
        notes.append(f"WARNING: {len(leaked)} evaluated test individuals were in the model's training set")
    if s["beats_every_baseline"] != "yes":
        notes.append("the model does not beat every baseline on the test set (paired bootstrap over test individuals): "
                     f"not beaten: {', '.join(n for n in names if n not in beats)}")
    if floor is not None and s["bigram_reference_entropy"] is not None and floor > s["bigram_reference_entropy"]:
        notes.append(f"the head's cross-entropy floor ({floor:.3f}, optimised oracle) is above the bigram reference entropy: comparing the "
                     "test loss with that reference is not meaningful for this head (see head_ceiling)")
    notes.append("baselines are counted on the training individuals: frequency (= identity-free for haplotype units), "
                 "add-k bigram, trigram with backoff, and a Li-Stephens HMM copying the training haplotypes"
                 if data.units == "haplotype" else
                 "baselines: identity_free (the current token's haploblock and slot, not its identity) and bigram")
    if tr is not None and s["generalization_gap_top1"] > 0.1:
        notes.append("training accuracy is much higher than test accuracy: likely overfitting")
    return {"summary": s, "notes": notes, "files": [path]}


def head_floor(model, data, targets, optimise=True, steps=200):
    """Mean cross-entropy of an oracle that chooses, for each target, the query direction of the model's
    cosine head (at its current logit scale): pointed exactly at the target's key (optimise=False), or
    optimised by gradient descent to minimise the cross-entropy over the target's candidates, which
    approximates the lowest cross-entropy the head can reach. None for a learned output table, which
    has no fixed floor."""
    import torch
    if model.head_kind == "table":
        return None
    targets = np.asarray(targets)
    total = 0.0
    for chrom in np.unique(data.chrom_of[targets]):
        sel = np.where(data.chrom_of[targets] == chrom)[0]
        tg = targets[sel]
        q = model.keys[torch.from_numpy(tg).to(model.rnr.dev)].detach().clone().requires_grad_(optimise)
        if optimise:
            opt = torch.optim.Adam([q], lr=0.05)
            for _ in range(steps):
                lg, t, _ = model.logits(q, tg, chrom)
                loss = torch.nn.functional.cross_entropy(lg, t)
                opt.zero_grad()
                q.grad = torch.autograd.grad(loss, q)[0]            # only q: the model gets no gradient
                opt.step()
        with torch.no_grad():
            lg, t, _ = model.logits(q, tg, chrom)
            total += torch.nn.functional.cross_entropy(lg, t, reduction="sum").item()
    return total / max(len(targets), 1)


@task("head_ceiling", "The lowest cross-entropy the model's output head can reach on the test targets")
def head_ceiling(ctx, max_targets: int = 5000, model: str = "", device: str = "", ckpt_dir: str = ""):
    """With a cosine head the softmax over near-collinear keys cannot be sharp, so even a perfect model
    has a cross-entropy floor; a learned table has none. Reports, at the model's logit scale and at
    the largest allowed one, the cross-entropy of an oracle pointing at each target's key and the
    floor (the query direction optimised per target), and how collinear the keys are."""
    import torch
    model, data, meta = query_model(ctx, model or None, device, ckpt_dir)
    ex = sample_examples(data, ctx.test_idx, 0, ctx.rng)
    targets = np.concatenate([data.tokens(c, e)[1:] for c, e in ex])
    if len(targets) > max_targets:
        targets = ctx.rng.choice(targets, max_targets, replace=False)
    s = {"head": meta["head"], "softmax": meta["softmax"], "targets": int(len(targets))}
    if meta["head"] == "table":
        s["ce_floor"] = 0.0
        notes = ["learned output table: every token has its own output vector, so the head has no fixed floor"]
    else:
        scale = model.scale()
        s["logit_scale"] = round(float(scale.detach() if hasattr(scale, "detach") else scale), 3)
        s["ce_at_target_keys"] = round(head_floor(model, data, targets, optimise=False), 4)
        s["ce_floor"] = round(head_floor(model, data, targets), 4)
        if meta["head"] == "scaled-cosine":
            with torch.no_grad():
                keep = model.extra["log_scale"].clone()
                model.extra["log_scale"].fill_(float(np.log(100.0)))
            s["ce_floor_max_scale"] = round(head_floor(model, data, targets), 4)
            with torch.no_grad():
                model.extra["log_scale"].copy_(keep)
        with torch.no_grad():
            near = []
            for chrom in data.present:
                for b in np.unique(data.region_code[data.cands[chrom]]):
                    toks = data.block_pad[b][data.block_pad[b] >= 0]
                    if len(toks) > 1:
                        k = model.keys[torch.from_numpy(toks).to(model.rnr.dev)]
                        near.append((k @ k.T).fill_diagonal_(-1).max(1).values.cpu().numpy())
        near = np.concatenate(near) if near else np.zeros(1)
        s["median_cosine_to_nearest_other_key"] = round(float(np.median(near)), 3)
        s["share_of_keys_above_0.95"] = round(float((near > 0.95).mean()), 3)
        notes = ["ce_floor = the cross-entropy an oracle reaches with the best query direction per target (optimised): "
                 "compare it with test_set_performance's bigram_reference_entropy"]
    return {"summary": s, "notes": notes}


@task("masked_imputation", "Impute masked haploblocks of test individuals from both flanks: model vs "
                           "frequency, flank lookup and a Li-Stephens HMM")
def masked_imputation(ctx, spans: str = "1,2,5", per_individual: int = 5, max_individuals: int = 300,
                      bootstrap: int = 500, model: str = "", device: str = "", ckpt_dir: str = ""):
    """For each span length in `spans`, `per_individual` non-overlapping spans of exactly that many
    contiguous haploblocks (separated by at least one observed haploblock) are masked on both haplotypes of up to max_individuals test individuals per
    chromosome (the same blocks on both, as in genotype imputation). Reports concordance (top-1 per
    haplotype) and r^2 between true and expected cluster dosage (0-2 copies), pooled over the masked
    (individual, haploblock, cluster) entries of each training-frequency bin, for the model (a masked-
    objective model; a next-token model is skipped) and the baselines; 95% CIs of model - baseline
    concordance resample individuals."""
    from haplo_baselines import BlockBaselines
    model, data, meta = query_model(ctx, model or None, device, ckpt_dir)
    if data.units != "haplotype":
        raise ValueError("masked_imputation needs a model with --units haplotype")
    use_model = meta["objective"] == "masked"
    rng = ctx.rng
    test = ctx.test_idx if len(ctx.test_idx) <= max_individuals else np.sort(rng.choice(ctx.test_idx, max_individuals, replace=False))
    rows = []
    for chrom in data.chroms:
        R, K = block_tokens(data, chrom, hap_rows(ctx.train_idx))
        bl = BlockBaselines(R, K)
        freq = [f / f.sum() for f in bl.freq]
        Y, _ = block_tokens(data, chrom, hap_rows(test))
        B = Y.shape[1]
        w0, w1 = model.window(B)                         # the model sees the central --max-len haploblocks
        for L in [int(x) for x in spans.split(",")]:
            slots = (w1 - w0 - 1) // (L + 1)            # spans of exactly L, an observed block between them
            if slots < 1:
                continue
            mask = np.zeros((len(test), B), bool)
            for i in range(len(test)):
                for j in rng.choice(slots, min(per_individual, slots), replace=False):
                    s0 = w0 + 1 + j * (L + 1)
                    mask[i, s0:s0 + L] = True
            M = np.repeat(mask, 2, axis=0)                                  # both haplotypes
            hmm = bl.hmm(Y, causal=False, mask=M)
            for i in range(len(test)):
                for h in (0, 1):
                    r = 2 * i + h
                    blocks = np.where(M[r])[0]
                    preds = {"frequency": [freq[b] for b in blocks], "hmm": [hmm[b][r] for b in blocks],
                             "flank": [flank(R, Y[r], b, M[r], K[b]) for b in blocks]}
                    if use_model:
                        sc = model.score(chrom, int(hap_rows([test[i]])[h]), mask=blocks, probs=True)
                        dense = {}
                        for b, cand, p, tgt in zip(sc["pos"], sc["cand"], sc["p"], sc["target"]):
                            v = np.zeros(K[b])
                            ok = (cand >= 0) & (data.region_code[np.maximum(cand, 0)] == data.region_code[tgt])
                            v[data.block_pos[cand[ok]]] = p[ok]                 # --softmax all: renormalised
                            v /= max(v.sum(), 1e-12)                            # within the haploblock
                            dense[b] = v
                        preds["model"] = [dense[b] for b in blocks]
                    for b_i, b in enumerate(blocks):
                        rows.append({"span": L, "individual": i, "hap": h, "chrom": chrom, "block": b, "true": Y[r, b],
                                     "fvec": freq[b], **{n: v[b_i] for n, v in preds.items()}})
    if not rows:
        raise ValueError("no chromosome has enough haploblocks for the requested spans")
    df = pd.DataFrame(rows)
    methods = (["model"] if use_model else []) + ["hmm", "flank", "frequency"]
    out, uniq = [], np.unique(df["individual"])
    for L, g in df.groupby("span"):
        res = {"span": L}
        idx = rng.integers(0, len(uniq), (bootstrap, len(uniq))) if bootstrap else None
        code = np.searchsorted(uniq, g["individual"].to_numpy())
        cnt = np.bincount(code, minlength=len(uniq)).astype(float)
        hits = {m: np.array([p.argmax() == t for p, t in zip(g[m], g["true"])]) for m in methods}
        for m in methods:
            res[f"{m}_concordance"] = round(float(hits[m].mean()), 4)
        if use_model and idx is not None:
            for m in methods[1:]:
                d = np.bincount(code, weights=hits["model"].astype(float) - hits[m], minlength=len(uniq))
                res[f"model_minus_{m}_95ci"] = ratio_ci(d, cnt, idx)
        # dosage r^2 per training-frequency bin: both haplotypes of an individual at one haploblock
        dos = g.groupby(["individual", "chrom", "block"])
        for m in methods:
            true_d, exp_d, fb = [], [], []
            for _, gg in dos:
                if len(gg) != 2:
                    continue
                f = gg["fvec"].iloc[0]
                true_d.append(np.bincount(gg["true"], minlength=len(f)))
                exp_d.append(np.sum(np.stack(gg[m].to_list()), 0))
                fb.append(freq_bin(f))
            if true_d:
                t, e, f = np.concatenate(true_d), np.concatenate(exp_d), np.concatenate(fb)
                for bin_ in ("<1%", "1-5%", "5-20%", ">20%"):
                    sel = f == bin_
                    if sel.sum() > 2 and t[sel].std() > 0 and e[sel].std() > 0:
                        res[f"{m}_r2_{bin_}"] = round(float(np.corrcoef(t[sel], e[sel])[0, 1] ** 2), 4)
        out.append(res)
    path = ctx.out("masked_imputation.csv")
    pd.DataFrame(out).to_csv(path, index=False)
    s = {"model_scored": "yes" if use_model else "no (next-token model)", "test_individuals": int(len(uniq)),
         "masked_haploblocks": int(len(df))}
    for r in out:
        for k, v in r.items():
            if k != "span":
                s[f"span{r['span']}_{k}"] = v
    notes = ["flank = the most frequent cluster among training haplotypes that match the test haplotype at the "
             "haploblocks just left and right of the masked span (left only, then frequency, if none match)",
             "r^2 compares true and expected cluster dosage (0-2 copies) over masked (individual, haploblock, "
             "cluster) entries, pooled per training-frequency bin of the cluster"]
    if not use_model:
        notes.append("the model has --objective next, so it cannot use the right flank: only the baselines are scored")
    return {"summary": s, "notes": notes, "files": [path]}


def flank(R, y, b, masked, K):
    """Training-haplotype cluster distribution at block b given the unmasked flanks of its masked span."""
    left = b
    while left >= 0 and masked[left]:
        left -= 1
    right = b
    while right < len(y) and masked[right]:
        right += 1
    match = np.ones(len(R), bool)
    for f in (left, right):
        if 0 <= f < len(y):
            m = match & (R[:, f] == y[f])
            if m.any():
                match = m
    c = np.bincount(R[match, b], minlength=K).astype(float)
    return (c + 1e-3) / (c.sum() + 1e-3 * K)


@task("model_comparison", "One table across the --model files (e.g. pretrained / random-init / small / frozen) "
                          "with paired CIs between every pair")
def model_comparison(ctx, max_examples: int = 300, bootstrap: int = 1000, device: str = "", ckpt_dir: str = ""):
    """Every 4a model given with --model is scored on the same test examples: next-token top-1 and
    cross-entropy (next-token models) or masked-position top-1 and cross-entropy at fixed masks
    (masked models). Differences between every pair of models get a 95% CI from a paired bootstrap
    over test individuals; a pair is compared only when both models predict exactly the same targets
    at the same positions (same readout, units, softmax, rare-cluster merging, windows and masks).
    Models are loaded one at a time."""
    import gc
    import torch
    files = [m.path for m in ctx.models]
    if len(files) < 2:
        raise ValueError("model_comparison needs at least two --model files")
    rng = ctx.rng
    per, table = {}, []
    test = ctx.test_idx
    for f in files:
        model, data, meta = query_model(ctx, f, device, ckpt_dir)
        ex = sample_examples(data, test, max_examples, np.random.default_rng(0))   # same examples per model
        hits, lps, inds, targets = [], [], [], []
        for c, e in ex:
            sc = model.score(c, e)
            hits.append(sc["top1"]); lps.append(sc["logp"])
            inds.append(np.full(len(sc["top1"]), e // 2 if data.units == "haplotype" else e))
            targets.append(np.stack([np.full(len(sc["pos"]), data.cidx[c]), sc["pos"], sc["target"]], 1))
        hits, lps, inds, targets = map(np.concatenate, (hits, lps, inds, targets))
        key = (meta["objective"], data.units, meta["softmax"])
        per[f] = (key, inds, hits, lps, targets)
        ctx.cache.pop((f, device, ckpt_dir), None)                  # one model in memory at a time
        del model
        gc.collect()                                                # the runner's hooks form a reference cycle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        table.append({"model": os.path.basename(f), "backend": meta["backend"], "init": meta["init"],
                      "units": data.units, "objective": meta["objective"], "head": meta["head"],
                      "train_blocks": meta["config"]["train_blocks"], "trainable_params": meta["trainable_params"],
                      "top1": round(float(hits.mean()), 4), "loss": round(float(-lps.mean()), 4),
                      "predictions": int(len(hits))})
    uniq = np.unique(np.concatenate([v[1] for v in per.values()]))
    idx = rng.integers(0, len(uniq), (bootstrap, len(uniq))) if bootstrap else None
    pairs = []
    for i, a in enumerate(files):
        for b in files[i + 1:]:
            (ka, ia, ha, la, ta), (kb, ib, hb, lb, tb) = per[a], per[b]
            row = {"model_a": os.path.basename(a), "model_b": os.path.basename(b)}
            if ka != kb or ta.shape != tb.shape or not np.array_equal(ta, tb):
                row["comparable"] = "no (different readout, softmax, targets or positions)"
            else:
                code = np.searchsorted(uniq, ia)
                cnt = np.bincount(code, minlength=len(uniq)).astype(float)
                d1 = np.bincount(code, weights=ha.astype(float) - hb, minlength=len(uniq))
                d2 = np.bincount(code, weights=lb - la, minlength=len(uniq))
                row.update(comparable="yes", top1_a_minus_b=round(float(ha.mean() - hb.mean()), 4),
                           top1_95ci=ratio_ci(d1, cnt, idx) if idx is not None else None,
                           loss_a_minus_b=round(float(lb.mean() - la.mean()), 4),
                           loss_95ci=ratio_ci(d2, cnt, idx) if idx is not None else None)
            pairs.append(row)
    p1, p2 = ctx.out("model_comparison.csv"), ctx.out("model_comparison_pairs.csv")
    pd.DataFrame(table).to_csv(p1, index=False)
    pd.DataFrame(pairs).to_csv(p2, index=False)
    s = {"models": len(files)}
    for r in table:
        s[f"{r['model']}_top1"] = r["top1"]
    for r in pairs:
        if r.get("comparable") == "yes":
            s[f"{r['model_a']}_minus_{r['model_b']}_top1"] = f"{r['top1_a_minus_b']} {r['top1_95ci']}"
    return {"summary": s, "notes": ["pretraining helps only if the pretrained model beats both the random-init and the "
                                    "small model at equal data (CI excluding 0)"], "files": [p1, p2]}


def auc(scores, labels):
    """ROC AUC (Mann-Whitney U / n_pos n_neg), ties averaged; NaN if a class is missing."""
    labels = np.asarray(labels, bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binned_auc(scores, labels, dist, n_bins=4):
    """Mean AUC within quantile bins of `dist`: compares pairs at similar distance (or frequency) only."""
    edges = np.unique(np.quantile(dist, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, max(len(edges) - 2, 0))
    vals = [auc(scores[b == k], labels[b == k]) for k in np.unique(b)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


@task("embedding_cooccurrence", "Do the node embeddings predict which clusters co-occur in the TEST individuals?")
def embedding_cooccurrence(ctx, labels: str = "auto", min_lift: float = 2.0, min_support: int = 5,
                           min_expected: float = 5.0, pairs: int = 50000, max_distance: int = 1_000_000,
                           bootstrap: int = 200):
    """Held-out test of the step-3 embeddings, no GPU needed. Samples distinct pairs of clusters in
    DIFFERENT haploblocks of the same chromosome, at most max_distance bp apart, each seen at least
    min_support times in the test set and with at least min_expected co-occurrences expected under
    independence. labels: haplotype (default when the graph has haplotypes.npz) = co-occurrence on the
    same TEST haplotype; individual = carried by the same test individual. A pair is positive when its
    test lift is >= min_lift and negative when it is <= 1. Per node_emb_<method>.npy: the ROC AUC of
    the embeddings' cosine similarity, overall, within genomic-distance bins and within expected-count
    (frequency) bins, with a bootstrap CI that resamples haploblocks; the AUC minus carrier_svd's
    (the GNN's and XGBoost's own input features) with a paired CI; and the AUC on pairs of nodes that
    have no edge in the graph. References: genomic proximity, cluster frequency, training lift."""
    notes = []
    leaked = ctx.split_info.get("test_individuals_in_graph_training_set") or 0
    built_on_all = str(ctx.split_info.get("graph_built_on", "")).startswith("all individuals")
    if leaked or built_on_all:
        notes.append("WARNING: the graph (and so the embeddings) was built with test individuals "
                     f"({'all individuals' if built_on_all else f'{leaked:,} of them'}): this AUC is NOT held-out")
    nodes = ctx.graph.nodes
    hf = os.path.join(ctx.graph.dir, "haplotypes.npz")
    labels = ("haplotype" if os.path.exists(hf) else "individual") if labels == "auto" else labels
    if labels == "haplotype":
        H = np.load(hf)["node"]

        def unit_matrix(inds):                     # nodes x haplotypes, 0/1
            rows = hap_rows(inds)
            return sp.csr_matrix((np.ones(rows.size * H.shape[1], np.float32),
                                  (H[rows].ravel(), np.repeat(np.arange(len(rows)), H.shape[1]))),
                                 shape=(len(nodes), len(rows)))
        C_te, C_tr = unit_matrix(ctx.test_idx), unit_matrix(ctx.train_idx)
    else:
        C = ctx.graph.carriers
        C_te, C_tr = (C[:, i].tocsr().astype(np.float32) for i in (ctx.test_idx, ctx.train_idx))
    sup_te, sup_tr = (np.asarray(m.sum(1)).ravel() for m in (C_te, C_tr))
    n_te, n_tr = C_te.shape[1], C_tr.shape[1]
    codes = ctx.graph.region_codes
    start = nodes["start"].to_numpy(dtype=np.float64)
    chrom = nodes["chrom"].to_numpy()
    rng = ctx.rng

    eligible = np.where((sup_te >= min_support) & np.isfinite(start))[0]
    if len(eligible) < 2:
        return {"summary": {"status": "skipped"}, "notes": notes + [
            f"skipped: fewer than 2 clusters seen >= {min_support} times in the test set"]}
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
    pair = np.unique(np.stack([np.minimum(np.concatenate(s_idx), np.concatenate(t_idx)),
                               np.maximum(np.concatenate(s_idx), np.concatenate(t_idx))], 1), axis=0)
    s_idx, t_idx = pair[:, 0], pair[:, 1]
    expected = sup_te[s_idx] * sup_te[t_idx] / n_te
    keep = (codes[s_idx] != codes[t_idx]) & (expected >= min_expected)
    s_idx, t_idx, expected = s_idx[keep], t_idx[keep], expected[keep]

    def lift(Cm, sup, n):
        shared = np.asarray(Cm[s_idx].multiply(Cm[t_idx]).sum(1)).ravel()
        return shared * n / np.maximum(sup[s_idx] * sup[t_idx], 1e-12)

    lift_te, lift_tr = lift(C_te, sup_te, n_te), lift(C_tr, sup_tr, n_tr)
    pos, neg = lift_te >= min_lift, lift_te <= 1.0
    use = pos | neg
    s_idx, t_idx, y = s_idx[use], t_idx[use], pos[use]
    expected, lift_tr = expected[use], lift_tr[use]
    dist = np.abs(start[s_idx] - start[t_idx])
    if y.sum() == 0 or (~y).sum() == 0:
        return {"summary": {"status": "skipped", "positive_pairs": int(y.sum()), "negative_pairs": int((~y).sum())},
                "notes": notes + [f"skipped: need both positive (lift >= {min_lift}) and negative (lift <= 1) "
                                  "test pairs; lower min_lift, min_support or min_expected"]}
    block = codes[s_idx]                                          # bootstrap unit: haploblock
    blocks = np.unique(block)
    members = pd.Series(np.arange(len(block))).groupby(block).apply(np.array)
    boot = [np.concatenate(members.loc[rng.choice(blocks, len(blocks))].to_list()) for _ in range(bootstrap)]
    A = sp.load_npz(os.path.join(ctx.graph.dir, "graph.npz"))
    isolated = np.diff(A.tocsr().indptr) == 0
    no_edges = isolated[s_idx] & isolated[t_idx]

    def evaluate(scores):
        res = {"auc": round(auc(scores, y), 4),
               "auc_distance_matched": round(binned_auc(scores, y, dist), 4),
               "auc_frequency_matched": round(binned_auc(scores, y, expected), 4),
               "auc_no_graph_edges": round(auc(scores[no_edges], y[no_edges]), 4)}
        if boot:
            lo_, hi_ = np.nanpercentile([auc(scores[t], y[t]) for t in boot], [2.5, 97.5])
            res["auc_95ci"] = (round(float(lo_), 4), round(float(hi_), 4))
        return res

    rows = [{"name": "genomic_proximity", "kind": "reference", **evaluate(-dist)},
            {"name": "cluster_frequency", "kind": "reference", **evaluate(np.log(expected))},
            {"name": "train_lift", "kind": "reference", **evaluate(lift_tr)}]
    files = sorted(glob.glob(os.path.join(ctx.graph.dir, "node_emb_*.npy")))
    gi = os.path.join(ctx.graph.dir, "graph_info.json")
    graph_sid = json.load(open(gi)).get("train_set_id", "all") if os.path.exists(gi) else "all"
    leaked_methods, cos = [], {}
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
        cos[name] = (Zn[s_idx] * Zn[t_idx]).sum(1)
        rows.append({"name": name, "kind": "embedding", **evaluate(cos[name])})
    if "carrier_svd" in cos:
        ref = cos["carrier_svd"]
        for r in rows:
            if r["kind"] == "embedding" and r["name"] != "carrier_svd":
                d = [auc(cos[r["name"]][t], y[t]) - auc(ref[t], y[t]) for t in boot]
                r["auc_minus_carrier_svd"] = round(r["auc"] - auc(ref, y), 4)
                if d:
                    lo_, hi_ = np.nanpercentile(d, [2.5, 97.5])
                    r["auc_minus_carrier_svd_95ci"] = (round(float(lo_), 4), round(float(hi_), 4))

    path = ctx.out("embedding_cooccurrence.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    s = {"labels": labels, "pairs": int(len(y)), "positive_pairs": int(y.sum()), "negative_pairs": int((~y).sum()),
         "pairs_without_graph_edges": int(no_edges.sum()), "eligible_clusters": int(len(eligible)),
         "regions_in_pairs": int(len(blocks)), "median_distance_bp": int(np.median(dist)),
         "held_out": "no" if (leaked or built_on_all or leaked_methods) else "yes"}
    for r in rows:
        for k in ("auc", "auc_distance_matched", "auc_frequency_matched", "auc_no_graph_edges", "auc_95ci",
                  "auc_minus_carrier_svd", "auc_minus_carrier_svd_95ci"):
            if k in r:
                s[f"{k}_{r['name']}"] = r[k]
    notes += ["AUC 0.5 = no information about which clusters co-occur in individuals never seen; distance- and "
              "frequency-matched AUCs compare pairs of similar genomic distance / expected co-count only",
              "auc_minus_carrier_svd: a graph method is useful only if it beats its own input features",
              "references: genomic_proximity, cluster_frequency (expected co-count), train_lift (the training "
              "individuals' linkage table); the 95% CI resamples haploblocks"]
    return {"summary": s, "notes": notes, "files": [path]}


# ---- phenotypes, GWAS and recombination through the model ------------------

GENETIC_MAP_URL = "https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh38.map.zip"


def ridge_fit_predict(Xtr, ytr, Xte, lambdas=np.logspace(-2, 4, 13)):
    """Ridge regression in its kernel form (n << features), lambda by generalized cross-validation.
    Returns (test predictions, feature weights)."""
    K = (Xtr @ Xtr.T).toarray() if sp.issparse(Xtr) else Xtr @ Xtr.T
    w, U = np.linalg.eigh(K)
    w = np.maximum(w, 0)
    yc = ytr - ytr.mean()
    Uy = U.T @ yc
    best = None
    for lam in lambdas * max(w.mean(), 1e-12):
        shrink = w / (w + lam)
        resid = yc - U @ (shrink * Uy)
        gcv = np.mean(resid ** 2) / (1 - shrink.sum() / len(yc)) ** 2
        if best is None or gcv < best[0]:
            best = (gcv, lam)
    alpha = U @ (Uy / (w + best[1]))
    beta = np.asarray(Xtr.T @ alpha).ravel()
    return np.asarray(Xte @ beta).ravel() + ytr.mean(), beta


def r2(pred, y):
    return float(1 - np.sum((pred - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))


@task("phenotype_prediction", "Predict a (simulated) phenotype of the test individuals: 4b model vs ridge on "
                              "cluster dosages, and recovery of the causal haploblocks")
def phenotype_prediction(ctx, model: str = "", phenotypes: str = "", bootstrap: int = 1000, device: str = "",
                         ckpt_dir: str = ""):
    """model = a 4b checkpoint with token input (default: the first --model); phenotypes = folder with
    <trait>_train.tsv / _test.tsv / _answer.json (default <graph-dir>/phenotypes, written by 4b
    --simulate). Test R^2 of the model and of ridge regression on cluster dosages (0-2 copies of each
    step-4a token; lambda by generalized cross-validation), with paired 95% CIs over test
    individuals. With an answer key: the haploblocks each method ranks first (as many as are causal)
    against the causal ones, as precision / recall (ridge: sum of |weights| of a haploblock's tokens;
    model: mean attention of the chromosome pooling on the haploblock's tokens)."""
    import torch
    from evo2_model import load_phenotype_model
    m = ctx.model if not model else Model(model)
    meta = m.meta
    if meta.get("mode") != "phenotype" or meta.get("input") != "tokens":
        raise ValueError(f"{m.path} is {model_kind(meta)}; phenotype_prediction needs a 4b model with token input")
    trait = meta["trait"]
    folder = phenotypes or os.path.join(ctx.graph.dir, "phenotypes")
    read = lambda part: pd.read_csv(os.path.join(folder, f"{trait}_{part}.tsv"), sep="\t", dtype={"individual": str})
    train, test = read("train"), read("test")
    pos = {x: i for i, x in enumerate(ctx.graph.individuals)}
    tr_i, te_i = (np.array([pos[x] for x in d["individual"] if x in pos]) for d in (train, test))
    y_tr, y_te = (d.set_index("individual").iloc[:, 0].reindex([ctx.graph.individuals[i] for i in idx]).to_numpy(float)
                  for d, idx in ((train, tr_i), (test, te_i)))
    pm, data = load_phenotype_model(meta, ctx.graph.dir, device or None, ckpt_dir or None)
    preds, attn = [], {}
    with torch.no_grad():
        for i in te_i:
            v, a = pm.encode(int(i))
            preds.append(float(pm.head(v)[0]) * meta["target_std"] + meta["target_mean"])
            for toks, w in a.values():
                for b, wt in zip(data.region_code[toks], w):
                    attn[b] = attn.get(b, 0.0) + wt / len(te_i)
    p_model = np.array(preds)

    # ridge on token dosages: individual x token
    cols = np.concatenate([data.block_cols[c] for c in data.chroms])
    tok = data.canon[data.H[:, cols]]
    vocab, inv = np.unique(tok, return_inverse=True)
    inv = inv.reshape(tok.shape)
    X = sp.csr_matrix((np.ones(inv.size), (np.repeat(np.arange(len(tok)) // 2, inv.shape[1]), inv.ravel())),
                      shape=(len(ctx.graph.individuals), len(vocab)))
    p_ridge, beta = ridge_fit_predict(X[tr_i], y_tr, X[te_i])
    idx = ctx.rng.integers(0, len(te_i), (bootstrap, len(te_i))) if bootstrap else None
    boot = lambda f: tuple(round(float(v), 4) for v in np.percentile([f(t) for t in idx], [2.5, 97.5])) if idx is not None else None
    s = {"trait": trait, "train_individuals": len(tr_i), "test_individuals": len(te_i),
         "model_r2": round(r2(p_model, y_te), 4), "model_r2_95ci": boot(lambda t: r2(p_model[t], y_te[t])),
         "ridge_r2": round(r2(p_ridge, y_te), 4), "ridge_r2_95ci": boot(lambda t: r2(p_ridge[t], y_te[t])),
         "model_minus_ridge_r2_95ci": boot(lambda t: r2(p_model[t], y_te[t]) - r2(p_ridge[t], y_te[t]))}
    s["model_beats_ridge"] = verdict(s["model_minus_ridge_r2_95ci"])
    notes = []
    ans = os.path.join(folder, f"{trait}_answer.json")
    if os.path.exists(ans):
        key = json.load(open(ans))
        causal = {b for c in key["causal"] for b in c["haploblocks"]}
        s["causal_haploblocks"] = len(causal)
        if causal:
            k = len(causal)
            ridge_imp = pd.Series(np.abs(beta)).groupby(data.region_code[vocab]).sum()
            for name, imp in (("ridge", ridge_imp), ("model", pd.Series(attn))):
                top = {data.regions[b] for b in imp.sort_values(ascending=False).index[:k]}
                s[f"{name}_causal_recall"] = round(len(top & causal) / k, 4)
        if key["trait"] == "cis":
            notes.append("cis trait: the effect needs two clusters on the SAME haplotype, which an additive model of "
                         "dosages cannot represent exactly; this is where a haplotype model should beat ridge")
        if key["trait"] == "null":
            notes.append("null trait: both R^2 should be ~0 (CI including 0)")
    notes.append("causal recall = share of the causal haploblocks among the haploblocks a method ranks first "
                 "(as many as are causal)")
    return {"summary": s, "notes": notes}


def gwas_hit_blocks(ctx, gwas, pvalue, clump):
    """Haploblocks (indices into ctx.graph.regions) containing a clumped GWAS locus, and the loci."""
    snps, _, _ = read_gwas(gwas)
    reg = ctx.graph.regions
    snps = snps[snps["chrom"].isin(set(reg["chrom"])) & (snps["p"] < pvalue)].reset_index(drop=True)
    if not len(snps):
        raise ValueError(f"no GWAS hits (p < {pvalue}) on the graph's chromosomes")
    c, p_, pv = snps["chrom"].to_numpy(), snps["pos"].to_numpy(), snps["p"].to_numpy()
    lead = clump_hits(c, p_, pv, np.ones(len(snps), bool), 25_000 if clump == "auto" else int(clump))
    loci = lead == np.arange(len(snps))
    si, ri = region_pairs(reg["chrom"].to_numpy(), reg["start"].to_numpy(np.int64), reg["end"].to_numpy(np.int64),
                          c[loci], p_[loci])
    return np.unique(ri), c[loci], p_[loci]


def block_scores(ctx, model, data, max_examples):
    """Per haploblock (index into ctx.graph.regions): mean surprisal (-log p) of the model's
    predictions of that haploblock on test haplotypes, mean entropy of those predictions, and the
    dispersion of the haploblock's cluster embeddings (1 - mean cosine to their frequency-weighted mean)."""
    reg_index = pd.Index(ctx.graph.regions["region"])
    sums = {}
    for chrom, ex in sample_examples(data, ctx.test_idx, max_examples, ctx.rng):
        sc = model.score(chrom, ex)
        for b, lp, en in zip(reg_index.get_indexer(data.regions[data.region_code[sc["target"]]]), sc["logp"], sc["entropy"]):
            v = sums.setdefault(b, [0.0, 0.0, 0])
            v[0] -= lp; v[1] += en; v[2] += 1
    n = len(reg_index)
    surprisal, entropy = np.full(n, np.nan), np.full(n, np.nan)
    for b, (a, e, k) in sums.items():
        surprisal[b], entropy[b] = a / k, e / k
    Zn = data.Z / np.maximum(np.linalg.norm(data.Z, axis=1, keepdims=True), 1e-12)
    w = np.maximum(data.support, 1e-9)
    code = reg_index.get_indexer(data.regions[data.region_code])
    mean = np.zeros((n, Zn.shape[1])); np.add.at(mean, code, Zn * w[:, None])
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    cos = (Zn * mean[code]).sum(1)
    disp = np.bincount(code, weights=(1 - cos) * w, minlength=n) / np.maximum(np.bincount(code, weights=w, minlength=n), 1e-12)
    return {"surprisal": surprisal, "entropy": entropy, "embedding_dispersion": disp}


@task("model_gwas", "Do the model's per-haploblock scores differ at GWAS-hit haploblocks from matched background?")
def model_gwas(ctx, gwas: str, pvalue: float = 5e-8, clump: str = "auto", matched: int = 10, permutations: int = 200,
               max_examples: int = 500, bootstrap: int = 500, model: str = "", device: str = "", ckpt_dir: str = ""):
    """Scores per haploblock (block_scores) of a 4a model at haploblocks holding a clumped GWAS locus
    vs `matched` background haploblocks each (same chromosome, nearest in log length, log number of
    clusters and cluster entropy; no hit): mean difference with a CI resampling hit haploblocks. Null
    p-values for the mean score at hit haploblocks: loci moved to random places on their chromosome,
    and all loci rotated along the chromosome with their spacing kept (as in gwas_enrichment)."""
    if not os.path.exists(gwas):
        raise ValueError(f"GWAS file {gwas} not found (5_download_gwas.py writes haplograph/gwas/<label>.tsv)")
    qm, data, meta = query_model(ctx, model or None, device, ckpt_dir)
    hits, lc, lp = gwas_hit_blocks(ctx, gwas, pvalue, clump)
    scores = block_scores(ctx, qm, data, max_examples)
    reg = ctx.graph.regions
    codes = ctx.graph.region_codes
    sup = ctx.graph.carriers.sum(1).A.ravel().astype(float)
    tot = np.bincount(codes, weights=sup, minlength=len(reg))
    f = sup / np.maximum(tot[codes], 1)
    ent = np.bincount(codes, weights=-f * np.log(np.maximum(f, 1e-12)), minlength=len(reg))
    feats = np.column_stack([np.log(np.maximum(reg["end"] - reg["start"], 1)), np.log(reg["n_clusters_total"]), ent])
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    is_hit = np.zeros(len(reg), bool); is_hit[hits] = True
    match = {}
    for b in hits:
        cand = np.where((reg["chrom"].to_numpy() == reg["chrom"].iloc[b]) & ~is_hit)[0]
        match[b] = cand[np.argsort(((feats[cand] - feats[b]) ** 2).sum(1))[:matched]]
    starts, ends = reg["start"].to_numpy(np.int64), reg["end"].to_numpy(np.int64)
    span = reg.groupby("chrom").agg(lo=("start", "min"), hi=("end", "max"))
    lo, hi = span.loc[lc, "lo"].to_numpy(), span.loc[lc, "hi"].to_numpy()
    L = hi - lo + 1
    rng, chroms = ctx.rng, pd.factorize(lc)

    def blocks_at(pos_):
        return np.unique(region_pairs(reg["chrom"].to_numpy(), starts, ends, lc, pos_)[1])
    nulls = {"random_loci": [], "rotation": []}
    for _ in range(permutations):
        nulls["random_loci"].append(blocks_at(lo + (rng.random(len(lc)) * L).astype(np.int64)))
        nulls["rotation"].append(blocks_at(lo + (lp - lo + (rng.random(len(chroms[1]))[chroms[0]] * L).astype(np.int64)) % L))
    s, rows = {"hit_haploblocks": int(len(hits)), "loci": int(len(lc)), "backbone": meta["backend"]}, []
    for name, v in scores.items():
        ok = [b for b in hits if np.isfinite(v[b]) and np.isfinite(v[match[b]]).any()]
        if not ok:
            continue
        diff = np.array([v[b] - np.nanmean(v[match[b]]) for b in ok])
        obs = float(np.nanmean(v[ok]))
        row = {"score": name, "hit_mean": round(obs, 4),
               "matched_mean": round(float(np.nanmean([np.nanmean(v[match[b]]) for b in ok])), 4),
               "difference": round(float(diff.mean()), 4)}
        if bootstrap:
            bt = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(bootstrap)]
            row["difference_95ci"] = tuple(round(float(x), 4) for x in np.percentile(bt, [2.5, 97.5]))
        for kind, draws in nulls.items():
            null = np.array([np.nanmean(v[d]) for d in draws if np.isfinite(v[d]).any()])
            if len(null):
                row[f"p_{kind}"] = round(float((1 + (np.abs(null - null.mean()) >= abs(obs - null.mean())).sum())
                                               / (1 + len(null))), 4)
        rows.append(row)
        for k, x in row.items():
            if k != "score":
                s[f"{name}_{k}"] = x
    path = ctx.out("model_gwas.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"summary": s, "notes": [
        "surprisal / entropy: the model's next-token (or masked-token) predictions of the haploblock on test "
        "haplotypes; embedding_dispersion: how different the haploblock's cluster embeddings are",
        "p-values are two-sided; 'rotation' keeps the spacing of the loci, so it also keeps true clustering of "
        "GWAS signals (see gwas_enrichment)"], "files": [path]}


def read_genetic_map(src, chroms, cache):
    """PLINK .map files (chrom, id, cM, bp) for `chroms`, from a zip (Beagle's GRCh38 maps) or a folder."""
    import io
    import zipfile
    path = download(src, url_cache_path(src, cache)) if is_url(src) else src
    out = {}
    names = (lambda: zipfile.ZipFile(path).namelist()) if path.endswith(".zip") else (lambda: os.listdir(path))
    for c in chroms:
        hit = [n for n in names() if re.search(rf"(^|[/.]){c}\.GRCh38\.map$|plink\.{c}\.|plink\.chr{c}\.", n)]
        if not hit:
            continue
        raw = zipfile.ZipFile(path).read(hit[0]) if path.endswith(".zip") else open(os.path.join(path, hit[0]), "rb").read()
        m = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None, usecols=[2, 3], names=["cM", "bp"])
        out[c] = m.sort_values("bp")
    return out


def spearman_ci(x, y, n_boot, rng, chunk=25):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho, p = stats.spearmanr(x, y)
    groups = np.arange(len(x)) // min(chunk, max(1, len(x) // 10))   # neighbouring boundaries are correlated
    boot = []
    for _ in range(n_boot):
        take = np.concatenate([np.where(groups == g)[0] for g in rng.choice(groups.max() + 1, groups.max() + 1)])
        boot.append(stats.spearmanr(x[take], y[take])[0])
    return round(float(rho), 4), float(p), tuple(round(float(v), 4) for v in np.nanpercentile(boot, [2.5, 97.5])), len(x)


@task("junction_strength", "Is the next haploblock less predictable where recombination is frequent?")
def junction_strength(ctx, genetic_map: str = GENETIC_MAP_URL, max_examples: int = 500, bootstrap: int = 200,
                      model: str = "", device: str = "", ckpt_dir: str = ""):
    """Recombination rate (cM/Mb) between the midpoints of adjacent haploblocks, from a GRCh38 genetic
    map (default: Beagle's PLINK maps), against (1) the conditional entropy H(next | current) of the
    training haplotypes' transitions (junctions.npz) and (2) the mean entropy of a next-token 4a
    model's predictions of the next haploblock on test haplotypes. Spearman rho with a CI resampling
    chunks of 25 neighbouring boundaries (rates and entropies are autocorrelated along a chromosome)."""
    jf = os.path.join(ctx.graph.dir, "junctions.npz")
    if not os.path.exists(jf):
        raise ValueError("junction_strength needs junctions.npz (step 1 --haplotypes, then step 2)")
    reg = ctx.graph.regions
    gmap = read_genetic_map(genetic_map, list(pd.unique(reg["chrom"])), os.path.join(ctx.out_dir, "inputs"))
    if not gmap:
        raise ValueError(f"no chromosome of the graph found in the genetic map {genetic_map}")
    h = np.load(os.path.join(ctx.graph.dir, "haplotypes.npz"))
    blocks = h["blocks"]
    reg_i = pd.Index(reg["region"]).get_indexer(blocks)
    mid = ((reg["start"] + reg["end"]) / 2).to_numpy()[reg_i]
    chrom = h["block_chrom"]
    rate = np.full(len(blocks) - 1, np.nan)
    for k in range(len(blocks) - 1):
        if chrom[k] == chrom[k + 1] and chrom[k] in gmap:
            m = gmap[chrom[k]]
            cm = np.interp([mid[k], mid[k + 1]], m["bp"], m["cM"])
            rate[k] = (cm[1] - cm[0]) / max((mid[k + 1] - mid[k]) / 1e6, 1e-9)
    j = np.load(jf)
    df = pd.DataFrame({"k": j["block"], "a": j["a"], "n": j["count"]})
    row_tot = df.groupby(["k", "a"])["n"].transform("sum")
    df["h"] = -df["n"] * np.log(df["n"] / row_tot)
    data_h = np.full(len(blocks) - 1, np.nan)
    agg = df.groupby("k")[["h", "n"]].sum()
    data_h[agg.index] = agg["h"] / agg["n"]
    s = {"boundaries_with_rate": int(np.isfinite(rate).sum()), "median_rate_cM_per_Mb": round(float(np.nanmedian(rate)), 3)}
    rho, p, ci, n = spearman_ci(rate, data_h, bootstrap, ctx.rng)
    s.update(data_entropy_rho=rho, data_entropy_p=p, data_entropy_rho_95ci=ci, boundaries=n)
    try:
        qm, data, meta = query_model(ctx, model or None, device, ckpt_dir)
    except ValueError:
        qm = None
    if qm is not None and meta["objective"] == "next":
        ent = block_scores(ctx, qm, data, max_examples)["entropy"]
        model_h = ent[reg_i][1:]                     # prediction of block k+1 = boundary k
        rho, p, ci, n = spearman_ci(rate, model_h, bootstrap, ctx.rng)
        s.update(model_entropy_rho=rho, model_entropy_p=p, model_entropy_rho_95ci=ci)
    return {"summary": s, "notes": ["positive rho: the next haploblock is harder to predict where recombination "
                                    "between the two is more frequent, as expected if the data (and the model) "
                                    "track linkage"]}


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
    """Properties the synthetic self-test must satisfy."""
    problems = []

    def ci_ok(value, ci, what):
        if ci is not None and not (ci[0] - 1e-9 <= value <= ci[1] + 1e-9):
            problems.append(f"{what} {value} outside its 95% CI {ci}")

    t = summaries["test_set_performance"]
    ci_ok(t["test_top1_accuracy"], t["test_top1_95ci"], "test top-1")
    for n in ("frequency", "bigram", "trigram", "hmm"):
        ci_ok(t[f"model_minus_{n}_top1"], t[f"model_minus_{n}_top1_95ci"], f"model - {n}")
    if t["test_top1_accuracy"] <= t["frequency_top1_accuracy"] - 0.15:
        problems.append("the small model is far below the frequency baseline: the learning setup is broken")
    h = summaries["head_ceiling"]
    if not 0 <= h["ce_floor"] <= h["ce_at_target_keys"] + 1e-6:
        problems.append(f"head_ceiling floor {h['ce_floor']} is not below the cross-entropy at the target keys")
    m = summaries["masked_imputation"]
    if m["model_scored"] != "yes" or not 0 <= m["span1_model_concordance"] <= 1:
        problems.append("masked_imputation did not score the masked model")
    ph = summaries["phenotype_prediction"]
    if not (np.isfinite(ph["ridge_r2"]) and ph["causal_haploblocks"] >= 1 and "ridge_causal_recall" in ph):
        problems.append("phenotype_prediction did not score ridge and causal recovery")
    if not any(k.startswith("surprisal_") for k in summaries["model_gwas"]):
        problems.append("model_gwas did not score surprisal at the hit haploblocks")
    if "data_entropy_rho" not in summaries["junction_strength"]:
        problems.append("junction_strength did not correlate entropy with the recombination rate")
    c = summaries["model_comparison"]
    if c["models"] != 2 or not any(k.endswith("_top1") and "_minus_" in k for k in c):
        problems.append("model_comparison did not compare the two next-token models")
    e = summaries["embedding_cooccurrence"]
    if e.get("status") == "skipped":
        problems.append("embedding_cooccurrence was skipped on the synthetic data")
    else:
        if e["auc_train_lift"] <= 0.55:                    # planted linkage replicates in test
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
    """Synthetic data -> step 4a --test (small backbone, next-token) plus a masked-objective and a frozen
    model -> every task."""
    step4 = importlib.import_module("4a_finetune_evo2_embeddings")
    step4.main(["--test"])
    args.graph_dir = selftest_graph_dir()
    small = ["--graph-dir", args.graph_dir, "--method", "gnn", "--backbone", "small", "--small-hidden", "32",
             "--small-layers", "2", "--device", "cpu", "--batch", "4", "--epochs", "2", "--max-len", "64",
             "--log-every", "1000", "--eval-every", "10000"]
    step4.main(small + ["--objective", "masked"])
    step4.main(small + ["--train-blocks", "0", "--out", os.path.join(args.graph_dir, "small_frozen_next_gnn.pt")])
    gwas, args.hotspot = planted_gwas(args.graph_dir, test_workdir("validate_gwas"))
    importlib.import_module("4b_finetune_evo2_phenotype").main(small + [
        "--simulate", "cis", "--causal-blocks", "3", "--h2", "0.8"])
    gmap = test_workdir("validate_map") / "map"
    gmap.mkdir()
    for c in ("chr21", "chr22"):                      # 1 cM per 100 kb, faster between 1.2 and 1.3 Mb
        bp = np.arange(0, 4_000_000, 10_000)
        (gmap / f"plink.{c}.GRCh38.map").write_text("".join(
            f"{c} . {x / 1e5 + 5 * np.clip((x - 1_200_000) / 1e5, 0, 1):.4f} {x}\n" for x in bp))
    args.model = [os.path.join(args.graph_dir, f) for f in ("small_next_gnn.pt", "small_frozen_next_gnn.pt")]
    masked = os.path.join(args.graph_dir, "small_masked_gnn.pt")
    pheno = os.path.join(args.graph_dir, "small_pheno_cis_gnn.pt")
    args.splits, args.tasks, args.gwas, args.out_dir = GRAPH_SPLIT, None, [], None
    args.task = ["model_info", "test_set_performance max_examples=60 train_examples=20 bootstrap=200 device=cpu",
                 "head_ceiling device=cpu",
                 f"masked_imputation model={shlex.quote(masked)} spans=1,2 per_individual=2 bootstrap=100 device=cpu",
                 "model_comparison bootstrap=100 device=cpu",
                 "embedding_cooccurrence min_lift=1.5 min_support=3 min_expected=1 pairs=5000 bootstrap=50",
                 "cluster_consistency permutations=20",
                 f"phenotype_prediction model={shlex.quote(pheno)} bootstrap=100 device=cpu",
                 f"model_gwas gwas={shlex.quote(gwas)} clump=500 permutations=50 bootstrap=50 device=cpu",
                 f"junction_strength genetic_map={shlex.quote(str(gmap))} bootstrap=50 device=cpu",
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
    ap.add_argument("--model", action="append", default=[],
                    help="fine-tuned model file from step 4 (.pt); repeat it to compare models (model_comparison); "
                         "the other tasks use the first")
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
                    help="self-test: synthetic data, small-backbone models from step 4a --test, every task (CPU)")
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
    for f in args.model:
        if not os.path.exists(f):
            raise SystemExit(f"error: model file {f} not found - run 4a_finetune_evo2_embeddings.py "
                             "(or 4b_finetune_evo2_phenotype.py) first")

    model = Model(args.model[0])                  # loaded once, on first use, shared with the tasks
    try:
        specs = read_tasks_file(args.tasks) if args.tasks else []
        specs += [parse_task_line(t, "--task") for t in args.task]
    except (OSError, ValueError) as e:
        die(f"bad task list: {e}")
    if not specs:
        specs = [("model_info", {})]
        if model.kind.startswith("4a"):
            next_model = model.meta.get("objective") == "next"
            specs += [("test_set_performance" if next_model else "masked_imputation", {}), ("head_ceiling", {})]
            if len(args.model) > 1:
                specs.append(("model_comparison", {}))
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

    stem = re.sub(r"\.pt$", "", os.path.basename(args.model[0]))
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
        json.dump({"model": args.model[0], "models": args.model, "graph_dir": args.graph_dir,
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
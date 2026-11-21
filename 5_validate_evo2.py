"""
Step 5 - validate a fine-tuned Evo 2 model on a held-out TEST set of individuals
(the model is assumed to have been trained on the remaining TRAIN individuals).

The script is a small task runner: every check is a TASK, and you choose which tasks to run (and
with which parameters) in a plain-text TASKS FILE, one task per line. Add a line to run a check
again with other settings (e.g. another GWAS trait); write a plugin file to add a new kind of check.

BUILT-IN TASKS
  model_info           What the model is (4a, 4a BioNeMo, 4b, or unknown), what it was trained on,
                       its stored training metrics, and a LEAKAGE check: test individuals that the
                       model did not hold out during training.
  cluster_consistency  Data-level internal consistency of the haploblock clusters between the train
                       and the test individuals, per haploblock region:
                         - number of distinct clusters carried in train and in test
                         - clusters seen only in test (the model never saw them during training)
                         - total variation distance (TVD) between train and test cluster frequencies
                         - chi-square test of homogeneity (BH-corrected)
                       Permutation null: the same statistics for random train/test splits of the
                       same sizes, so a smaller test set is not mistaken for a shift. If the split
                       is consistent, about 5% of regions fall outside the 95% interval.
  gwas_enrichment      Data-level enrichment of GWAS hits (SNPs with P < pvalue) in haploblocks:
                         - per region: hits vs expected; background = all SNPs of the GWAS file in
                           the region (hypergeometric test) when the file has full summary
                           statistics, or region length (Poisson test) when it only has hits;
                           BH-corrected
                         - global: hits inside all graph haploblocks vs a circular-shift
                           permutation null per chromosome (only informative if haploblocks don't
                           cover the whole chromosome span; the script says so when they do)

INPUT
  --model FILE.pt           one fine-tuned model from step 4 (4a / 4a BioNeMo / 4b). Any torch.save
                            dict works; unknown kinds run as "unknown" and only data-level tasks use
                            them. Loaded only when a task needs it (needs torch).
  --graph-dir               output folder of 1_build_graph.py: nodes.csv.gz, individuals.txt,
                            carriers.npz (formats documented in 1_build_graph.py)
  TRAIN / TEST SPLIT  (default: the BH_26 split, https://data.haploblocks.org/BH_26/splits/)
    --splits DIR_OR_URL     folder (local path or URL) holding the split files; default the BH_26
                            folder above. Train = <splits>/half_A.txt (the individuals steps 3-4 were
                            trained on), test = <splits>/half_B.txt (the individuals Evo 2 is
                            queried on here). Other names: --train-file / --test-file.
                            Files given as URLs are downloaded once into <out-dir>/inputs/.
                            Use --splits none to fall back to the model's own held-out individuals.
    --train-individuals / --test-individuals FILE_OR_URL
                            explicit split files; override --splits. Without a train file, train =
                            every graph individual not in the test set.
    Split file format: plain text, one sample ID per line, no header ('#' = comment), e.g.
                              HG00096
                              NA06985
                            or a CSV with an "individual" column. half_A.txt has 1,252 and
                            half_B.txt 1,251 1000 Genomes sample IDs. Individuals of the graph in
                            neither file are left out of both sets (and counted in the report).
    --sample-map CSV        only if the graph's individuals (individuals.txt) are not named by
                            sample ID (e.g. IND1, IND2 ...): 2 columns, graph name, sample ID
                            (same file format as for 2_ancestry_labels.py)
  --tasks FILE              tasks file (see below); default: model_info + cluster_consistency
                            (+ gwas_enrichment if --gwas is given)
  --gwas FILE               shortcut: adds "gwas_enrichment gwas=FILE"
  GWAS file (gwas_enrichment gwas=...): tab-, comma- or whitespace-separated text, may be .gz/.zip.
      Needs chromosome, position (bp, SAME GENOME BUILD as the haploblock regions) and p-value
      columns; an SNP id column is optional. Auto-detected names (case-insensitive):
        chromosome  CHR, CHROM, #CHROM, CHR_ID, CHROMOSOME
        position    POS, BP, CHR_POS, POSITION, BASE_PAIR_LOCATION
        p-value     P, PVAL, P_VALUE, P-VALUE, PVALUE, P_BOLT_LMM
        SNP id      RSID, SNP, SNPS, SNPID, MARKERNAME, VARIANT_ID, ID
      e.g. GIANT height (Yengo et al. 2022: SNPID RSID CHR POS ... P N, hg19) or the GWAS Catalog
      associations file (CHR_ID CHR_POS SNPS P-VALUE ...). Override with chrom_col= pos_col= p_col=
      snp_col=. Chromosomes may be written 1 or chr1. Rows that don't parse are skipped.

TASKS FILE  (plain text, one task per line; '#' starts a comment; values with spaces need quotes)
    model_info
    cluster_consistency min_carriers=1 permutations=200
    gwas_enrichment gwas=Yengo.2022.height.GWAS.ALL.gz pvalue=5e-8 label=height
    gwas_enrichment gwas=bmi_catalog.tsv pvalue=1e-6 window=10000 label=bmi_hits
  `python 5_validate_evo2.py --list-tasks` prints every task with its parameters and defaults.

PLUGINS  (--plugins my_checks.py, repeatable) - a Python file whose functions are registered with
the `task` decorator, which the script provides to the file:
    @task("mean_clusters_per_individual", "Average number of clusters carried per test individual")
    def mean_clusters(ctx, min_support: int = 1):
        n = ctx.graph.carriers[:, ctx.test_idx].sum(axis=0)
        return {"summary": {"mean": float(n.mean())}}
  A task gets `ctx` plus its parameters (converted using the type annotations) and returns a dict
  with "summary" (dict of numbers/strings), optional "notes" (list of str), optional "files".
  ctx provides: ctx.graph.nodes (DataFrame), ctx.graph.carriers (CSR uint8 [N, I]),
  ctx.graph.individuals (list), ctx.graph.region_codes / ctx.graph.regions (per-node region index
  and the region table), ctx.train_idx / ctx.test_idx (column indices into carriers), ctx.model
  (.meta = the checkpoint dict, .kind), ctx.out(name) (path inside the output folder), ctx.rng, log().

OUTPUT  (--out-dir, default <graph-dir>/validation_<model file name>/)
  report.md                          human-readable summary of every task
  summary.json                       {"model", "test_set", "tasks": [{name, params, status, summary,
                                       notes, files}]}
  split.csv                          individual, set (train / test)
  cluster_consistency_regions.csv.gz per region: region, chrom, start, end, n_clusters_total,
                                     n_clusters_train, n_clusters_test, test_expected, test_ci_low,
                                     test_ci_high, outside_ci, n_test_only, test_only_carriers, tvd,
                                     tvd_null_mean, tvd_p, chi2, chi2_df, chi2_p, chi2_q
  gwas_enrichment_<label>_regions.csv.gz
                                     per region: region, chrom, start, end, n_snps, n_hits,
                                     expected_hits, fold, p, q, lead_snp, lead_p
  (a task that is listed twice writes its files with a _2, _3 ... suffix)

USAGE
  python 5_validate_evo2.py --model haplograph/genome/evo2_next_gnn.pt     # BH_26 half_A / half_B
  python 5_validate_evo2.py --model evo2_next_gnn.pt --splits splits/ --gwas height_hg38.tsv
  python 5_validate_evo2.py --model evo2_next_gnn.pt --sample-map ind_to_sample.csv
  python 5_validate_evo2.py --model evo2_hash_height.pt --test-individuals test.txt --tasks my_tasks.txt
  python 5_validate_evo2.py --list-tasks

REQUIRES  numpy, pandas, scipy; torch to read the model file.
          optional: RAPIDS cuDF (pip install cudf-cu12) for --cudf, which runs the pandas parts (mainly
          reading GWAS summary statistics) on the GPU
"""
import argparse
import inspect
import json
import os
import re
import shlex
import time
import traceback

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


def log(msg=""):
    print(msg, flush=True)


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
    """Lazy access to the files written by 1_build_graph.py."""

    def __init__(self, graph_dir):
        self.dir = graph_dir
        self._nodes = self._carriers = self._individuals = self._regions = None

    def _f(self, name):
        p = os.path.join(self.dir, name)
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} - run 1_build_graph.py first")
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
            self._individuals = open(self._f("individuals.txt")).read().split("\n")
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
        return "4a embeddings (BioNeMo)" if meta.get("backend") == "bionemo" else "4a embeddings"
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
            try:
                self._meta = torch.load(self.path, map_location="cpu", weights_only=True)
            except Exception:
                # checkpoints from steps 4a/4b are plain dicts, but older torch versions or extra
                # objects may need the full unpickler; this is the user's own file
                self._meta = torch.load(self.path, map_location="cpu", weights_only=False)
        return self._meta

    @property
    def kind(self):
        return model_kind(self.meta)

    def held_out_individuals(self):
        v = self.meta.get("val_individuals") if isinstance(self.meta, dict) else None
        return list(v) if v else None

    def trained_individuals(self):
        """Individuals the model was trained on, if the checkpoint records them."""
        v = self.meta.get("train_individuals") if isinstance(self.meta, dict) else None
        return list(v) if v else None


BH26_SPLITS = "https://data.haploblocks.org/BH_26/splits/"


def is_url(p):
    return str(p).startswith(("http://", "https://"))


def join_path(base, name):
    return base.rstrip("/") + "/" + name if is_url(base) else os.path.join(base, name)


def fetch(path, cache_dir):
    """Local path as is; a URL is downloaded once into cache_dir."""
    if not is_url(path):
        if not os.path.exists(path):
            raise SystemExit(f"file not found: {path}")
        return path
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, os.path.basename(path.rstrip("/")))
    if not os.path.exists(dest):
        log(f"  downloading {path}")
        try:
            tmp = dest + ".part"
            urllib.request.urlretrieve(path, tmp)
            os.replace(tmp, dest)
        except Exception as e:
            raise SystemExit(f"could not download {path} ({e}). Download the split files yourself, e.g.\n"
                             f"  wget -r -np -nH --cut-dirs=2 -R 'index.html*' {BH26_SPLITS}\n"
                             f"and pass the local folder with --splits splits/")
    return dest


def read_individual_list(path):
    if path.endswith((".csv", ".csv.gz")):
        df = pd.read_csv(path, dtype=str)
        col = "individual" if "individual" in df.columns else df.columns[0]
        return [x.strip() for x in df[col].dropna()]
    out = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line.split()[0])
    return out


def load_sample_map(path):
    """CSV with 2 columns (graph individual name, sample ID) -> {sample ID: graph name}."""
    m = pd.read_csv(path, dtype=str)
    if m.shape[1] < 2:
        raise SystemExit(f"{path}: --sample-map needs 2 columns (graph name, sample ID)")
    return dict(zip(m.iloc[:, 1].str.strip(), m.iloc[:, 0].str.strip()))


class Context:
    def __init__(self, args):
        self.args = args
        self.graph = Graph(args.graph_dir)
        self.model = Model(args.model)
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
                raise SystemExit("--train-individuals given without --test-individuals")
            test_src, train_src = a.test_individuals, a.train_individuals
        elif a.splits and a.splits.lower() != "none":
            test_src, train_src = join_path(a.splits, a.test_file), join_path(a.splits, a.train_file)
        else:
            test_src = train_src = None

        def load(src, what):
            raw = read_individual_list(fetch(src, cache))
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
        else:
            train_set, n_train_file = set(inds) - test_set, None
            train_source = "all other graph individuals"

        # the graph's own training set (1_build_graph.py --train-individuals) must not contain test individuals
        graph_train_file = os.path.join(self.graph.dir, "train_individuals.txt")
        if os.path.exists(graph_train_file):
            graph_train = set(read_individual_list(graph_train_file))
            in_both = test_set & graph_train
            self.split_info["graph_built_on"] = f"{len(graph_train):,} training individuals (train_individuals.txt)"
            self.split_info["test_individuals_in_graph_training_set"] = len(in_both)
            if in_both:
                log(f"WARNING: {len(in_both):,} test individuals are in the graph's training set "
                    f"(e.g. {sorted(in_both)[:3]}): steps 1-4 saw them. Rebuild the graph with "
                    "--train-individuals set to the train half")
            elif graph_train != train_set:
                log(f"note: the graph was built on {len(graph_train):,} training individuals, the train set "
                    f"here has {len(train_set):,} ({len(graph_train - train_set):,} / {len(train_set - graph_train):,} "
                    "differ); test individuals are not in it")
        else:
            self.split_info["graph_built_on"] = "all individuals (no train_individuals.txt)"
            log("note: the graph was built from ALL individuals (1_build_graph.py without --train-individuals), "
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
            # the checkpoint lists exactly who it was trained on
            trained = set(trained)
            leaked = [x for x in test_names if x in trained]
            s["test_individuals_seen_in_training"] = len(leaked)
            notes.append(f"WARNING: {len(leaked):,} of {len(test_names):,} test individuals are in the model's "
                         f"training set (e.g. {leaked[:3]})" if leaked else
                         "no leakage: no test individual is in the model's training set")
        elif held is not None and test_names:
            # 4a checkpoints without a training list: trained on every graph individual except val_individuals
            held = set(held)
            leaked = [x for x in test_names if x not in held]
            s["test_individuals_seen_in_training"] = len(leaked)
            if leaked:
                notes.append(f"WARNING: {len(leaked):,} of {len(test_names):,} test individuals were NOT held "
                             f"out by the model (e.g. {leaked[:3]}), so they were in its training data. "
                             "Retrain step 4 on the train half only, so the test half stays unseen")
            else:
                notes.append("no leakage: every test individual was held out during training")
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
    notes = [f"chromosomes with most regions outside the interval: "
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


def assign_to_regions(reg_chrom, starts, ends, snp_chrom, snp_pos):
    """Index of the region containing each SNP (start <= pos <= end), -1 if none."""
    out = np.full(len(snp_pos), -1, dtype=np.int64)
    for c in pd.unique(reg_chrom):
        ri = np.where(reg_chrom == c)[0]
        si = np.where(snp_chrom == c)[0]
        if len(ri) == 0 or len(si) == 0:
            continue
        order = ri[np.argsort(starts[ri], kind="stable")]
        k = np.searchsorted(starts[order], snp_pos[si], side="right") - 1
        ok = k >= 0
        cand = order[np.where(ok, k, 0)]
        inside = ok & (snp_pos[si] <= ends[cand])
        out[si[inside]] = cand[inside]
    return out


@task("gwas_enrichment", "Are haploblocks enriched in GWAS hits (per region and overall)?")
def gwas_enrichment(ctx, gwas: str, pvalue: float = 5e-8, window: int = 0, permutations: int = 1000,
                    label: str = "", chrom_col: str = "", pos_col: str = "", p_col: str = "",
                    snp_col: str = "", sep: str = "auto"):
    """window = bp added on each side of every haploblock region."""
    label = label or re.sub(r"(\.gz|\.zip|\.tsv|\.txt|\.csv)+$", "", os.path.basename(gwas))
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
    snps, cols, n_bad = read_gwas(gwas, chrom_col, pos_col, p_col, snp_col, sep)
    reg = ctx.graph.regions.dropna(subset=["start", "end"]).reset_index(drop=True).copy()
    reg_chrom = reg["chrom"].to_numpy()
    starts = reg["start"].to_numpy(np.int64) - window
    ends = reg["end"].to_numpy(np.int64) + window

    on = snps["chrom"].isin(set(reg_chrom))
    snps = snps[on].reset_index(drop=True)
    if len(snps) == 0:
        raise ValueError(f"no GWAS SNPs on the graph's chromosomes ({sorted(set(reg_chrom))[:5]}...)")
    hit = (snps["p"] < pvalue).to_numpy()
    summary_stats = (~hit).mean() > 0.5            # mostly non-significant rows -> full summary stats
    s_chrom, s_pos = snps["chrom"].to_numpy(), snps["pos"].to_numpy()

    # haploblock span per chromosome (min start .. max end); SNPs outside it are not tested
    span = reg.assign(s=starts, e=ends).groupby("chrom").agg(lo=("s", "min"), hi=("e", "max"))
    lo = span.loc[s_chrom, "lo"].to_numpy(); hi = span.loc[s_chrom, "hi"].to_numpy()
    in_span = (s_pos >= lo) & (s_pos <= hi)
    frac_outside = float(1 - in_span.mean())
    snps, hit, s_chrom, s_pos, lo, hi = snps[in_span].reset_index(drop=True), hit[in_span], \
        s_chrom[in_span], s_pos[in_span], lo[in_span], hi[in_span]
    idx = assign_to_regions(reg_chrom, starts, ends, s_chrom, s_pos)

    # ---- per region
    R = len(reg)
    inside = idx >= 0
    n_snps = np.bincount(idx[inside], minlength=R)
    n_hits = np.bincount(idx[inside], weights=hit[inside], minlength=R).astype(int)
    K, M = int(hit.sum()), len(snps)
    length = (ends - starts + 1).astype(np.float64)
    if summary_stats:
        expected = n_snps * K / max(M, 1)
        p = stats.hypergeom.sf(n_hits - 1, M, K, n_snps)
        background = "all GWAS SNPs in the region (hypergeometric)"
    else:
        hits_chrom = pd.Series(hit).groupby(s_chrom).sum()
        chrom_len = (span["hi"] - span["lo"] + 1).astype(np.float64)
        rate = (hits_chrom / chrom_len.loc[hits_chrom.index]).reindex(span.index).fillna(0)
        expected = length * rate.loc[reg_chrom].to_numpy()
        p = stats.poisson.sf(n_hits - 1, np.maximum(expected, 1e-300))
        background = "region length (Poisson)"
    p = np.where(n_snps + n_hits > 0, p, np.nan) if summary_stats else p
    reg["n_snps"], reg["n_hits"], reg["expected_hits"] = n_snps, n_hits, expected
    reg["fold"] = np.where(expected > 0, n_hits / np.maximum(expected, 1e-300), np.nan)
    reg["p"], reg["q"] = p, bh(p)
    lead = pd.DataFrame({"r": idx[inside], "p": snps["p"].to_numpy()[inside],
                         "snp": snps["snp"].to_numpy()[inside] if cols["snp"] else
                         (s_chrom[inside] + ":" + s_pos[inside].astype(str))})
    lead = lead.sort_values("p").drop_duplicates("r").set_index("r")
    reg["lead_snp"] = lead["snp"].reindex(np.arange(R)).to_numpy()
    reg["lead_p"] = lead["p"].reindex(np.arange(R)).to_numpy()

    # ---- global: hits inside haploblocks vs circular shifts of the SNP positions per chromosome
    obs = int(hit[inside].sum())
    covered = float(length.sum() / (span["hi"] - span["lo"] + 1).sum())
    hp, hc, hlo, hhi = s_pos[hit], s_chrom[hit], lo[hit], hi[hit]
    chroms_h = pd.unique(hc)
    null = np.zeros(permutations)
    L = hhi - hlo + 1
    for j in range(permutations):
        shift = dict(zip(chroms_h, ctx.rng.random(len(chroms_h))))
        u = (np.array([shift[c] for c in hc]) * L).astype(np.int64)
        shifted = hlo + (hp - hlo + u) % L
        null[j] = (assign_to_regions(reg_chrom, starts, ends, hc, shifted) >= 0).sum()
    p_global = (1 + (null >= obs).sum()) / (1 + permutations)
    if summary_stats:
        frac_snps_in = float(inside.mean())
        bt = stats.binomtest(obs, max(K, 1), frac_snps_in, alternative="greater") if K else None

    path = ctx.out(f"gwas_enrichment_{safe}_regions.csv.gz")
    reg[["region", "chrom", "start", "end", "n_snps", "n_hits", "expected_hits", "fold", "p", "q",
         "lead_snp", "lead_p"]].to_csv(path, index=False, float_format="%.6g")

    n_sig = int((reg["q"] < 0.05).sum())
    s = {
        "gwas_file": gwas, "label": label, "columns_used": ",".join(f"{k}={v}" for k, v in cols.items() if v),
        "file_type": "summary statistics" if summary_stats else "hits only",
        "pvalue_threshold": pvalue, "window_bp": window,
        "snps_tested": M, "hits": K, "hits_in_haploblocks": obs,
        "regions_with_hits": int((n_hits > 0).sum()), "regions": R,
        "regions_enriched_q<0.05": n_sig, "per_region_background": background,
        "global_hits_expected_by_shift": round(float(null.mean()), 2),
        "global_fold": round(float(obs / null.mean()), 3) if null.mean() > 0 else None,
        "global_p_shift": round(float(p_global), 4),
        "haploblock_coverage_of_span": f"{covered:.1%}",
        "snps_outside_haploblock_span": f"{frac_outside:.1%}",
    }
    if summary_stats and bt is not None:
        s["global_p_binomial_vs_snp_density"] = float(bt.pvalue)
    notes = []
    if covered > 0.95:
        notes.append(f"haploblocks cover {covered:.0%} of each chromosome span, so almost every SNP is inside one: "
                     "the GLOBAL test is uninformative here; use the per-region results")
    if frac_outside > 0.5:
        notes.append(f"{frac_outside:.0%} of the GWAS SNPs fall outside the haploblock spans - check that the GWAS "
                     "file and the haploblock regions use the same genome build (e.g. GIANT 2022 is hg19/GRCh37)")
    if n_bad:
        notes.append(f"{n_bad:,} GWAS rows had no usable position or p-value and were skipped")
    top = reg[reg["q"] < 0.05].sort_values("p").head(5)
    if len(top):
        notes.append("top enriched haploblocks: " + "; ".join(
            f"{r.region} ({r.n_hits} hits, {r.fold:.1f}x, q={r.q:.2g})" for r in top.itertuples()))
    return {"summary": s, "notes": notes, "files": [path]}


# ===========================================================================
# Runner and report
# ===========================================================================

def write_report(ctx, results, path):
    lines = [f"# Evo 2 validation report", "",
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="fine-tuned model file from step 4 (.pt)")
    ap.add_argument("--graph-dir", default="haplograph/genome", help="output folder of 1_build_graph.py")
    sg = ap.add_argument_group("train / test split")
    sg.add_argument("--splits", default=BH26_SPLITS,
                    help="folder or URL with the split files (default: BH_26 split); 'none' = model's held-out set")
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
    args = ap.parse_args()

    for p in args.plugins:
        load_plugin(p)
    if args.list_tasks:
        list_tasks()
        return
    if not args.model:
        ap.error("--model is required")
    if not os.path.exists(args.model):
        raise SystemExit(f"model file {args.model} not found")

    specs = read_tasks_file(args.tasks) if args.tasks else []
    specs += [parse_task_line(t, "--task") for t in args.task]
    specs += [("gwas_enrichment", {"gwas": g}) for g in args.gwas]
    if not specs:
        specs = [("model_info", {}), ("cluster_consistency", {})]
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
    ctx = Context(args)
    ctx.setup_split()

    results = []
    for name, params in jobs:
        log(f"[{name}] " + " ".join(f"{k}={v}" for k, v in params.items()))
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


if __name__ == "__main__":
    main()
"""Convert SynthTAN builder files to the per-chromosome layout that 1_build_graph.py reads.

SynthTAN ships one genome-wide pair of builder files per cohort (haploblock-graph-builder format):
  builder/nodes.csv   id (<chrom>_<start>-<end>_cluster<N>), high_dim_edge (<chrom>_<start>-<end>), then one
                      0/1 column per patient (S001 ...)
  builder/edges.csv   source, target, weight (number of patients carrying both nodes)
1_build_graph.py reads <local-dir>/<chrom>/nodes.csv.gz and <local-dir>/<chrom>/edges.csv.gz instead, and keeps
only edges whose two nodes are on the chromosome it is reading. This script splits the files by chromosome and
writes edges within a chromosome only; edges between chromosomes (the SynthTAN co-loss motifs pair regions on
different chromosomes) are counted and reported, not written, because step 1 would drop them anyway.
The builder's bookkeeping region chrM_1-2 (label ALL, carried by every patient so that event-free patients still
appear in the builder's input) is not a genome event and is left out (--keep-sentinel keeps it).

OUTPUT  <out>/<chrom>/nodes.csv.gz, <out>/<chrom>/edges.csv.gz, <out>/chroms.txt (comma-separated list for
        1_build_graph.py --chroms), <out>/conversion.json (counts, including dropped cross-chromosome edges, and
        the sha256 of the builder files, which run_genome_folds.sh checks against the fusion folder's manifest)

USAGE   python fusion/synthtan_to_haplograph.py --builder-dir synthetic_v0_1/cohort_n200/builder \\
            --out haplograph_input/cohort_n200
        python 1_build_graph.py --local-dir haplograph_input/cohort_n200 \\
            --chroms "$(cat haplograph_input/cohort_n200/chroms.txt)" --edges all ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ID_RE = re.compile(r"(chr[^_]+)_(\d+)-(\d+)_cluster\d+")


SENTINEL = "chrM_1-2"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chrom_key(c: str):
    m = re.fullmatch(r"chr(\d+)", c)
    return (0, int(m.group(1)), "") if m else (1, 0, c)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--builder-dir", required=True, help="folder with nodes.csv and edges.csv")
    ap.add_argument("--out", required=True, help="output folder (one subfolder per chromosome)")
    ap.add_argument("--keep-sentinel", action="store_true", help=f"keep the builder's {SENTINEL} ALL region")
    args = ap.parse_args(argv)
    src, out = Path(args.builder_dir), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new folder")

    nodes = pd.read_csv(src / "nodes.csv", dtype={"id": str, "high_dim_edge": str})
    if list(nodes.columns[:2]) != ["id", "high_dim_edge"]:
        raise SystemExit(f"{src}/nodes.csv: expected columns id, high_dim_edge, <patients...>")
    if nodes["id"].duplicated().any():
        raise SystemExit(f"{src}/nodes.csv: duplicated node ids")
    parsed = nodes["id"].str.fullmatch(ID_RE.pattern)
    if not parsed.all():
        raise SystemExit(f"{src}/nodes.csv: ids not like <chrom>_<start>-<end>_cluster<N>, "
                         f"e.g. {nodes.loc[~parsed, 'id'].iloc[0]}")
    region = nodes["id"].str.replace(r"_cluster\d+$", "", regex=True)
    if not (region == nodes["high_dim_edge"]).all():
        raise SystemExit(f"{src}/nodes.csv: high_dim_edge does not match the id's region")
    patients = nodes.columns[2:]
    vals = nodes[patients]
    if not vals.isin([0, 1]).all().all():
        raise SystemExit(f"{src}/nodes.csv: patient columns must be 0/1")
    all_nodes = nodes
    sentinel = nodes["high_dim_edge"] == SENTINEL
    if sentinel.any() and not args.keep_sentinel:
        if not (nodes.loc[sentinel, patients] == 1).all().all():
            raise SystemExit(f"{SENTINEL} is not carried by every patient; is it really the builder's ALL region?")
        nodes = nodes[~sentinel].reset_index(drop=True)
    chrom_of = nodes["id"].str.extract(ID_RE)[0]
    node_chrom = dict(zip(all_nodes["id"], all_nodes["id"].str.extract(ID_RE)[0]))
    kept = set(nodes["id"])

    edges = pd.read_csv(src / "edges.csv", dtype={"source": str, "target": str})
    if list(edges.columns) != ["source", "target", "weight"]:
        raise SystemExit(f"{src}/edges.csv: expected columns source, target, weight")
    unknown = ~(edges["source"].isin(node_chrom) & edges["target"].isin(node_chrom))
    if unknown.any():
        raise SystemExit(f"{src}/edges.csv: {int(unknown.sum())} edges refer to unknown node ids")
    n_sentinel_edges = int((~(edges["source"].isin(kept) & edges["target"].isin(kept))).sum())
    edges = edges[edges["source"].isin(kept) & edges["target"].isin(kept)]
    e_chrom_s = edges["source"].map(node_chrom)
    e_chrom_t = edges["target"].map(node_chrom)
    within = e_chrom_s == e_chrom_t

    chroms = sorted(chrom_of.unique(), key=chrom_key)
    per_chrom = {}
    out.mkdir(parents=True, exist_ok=True)
    for c in chroms:
        d = out / c
        d.mkdir()
        n_c = nodes[chrom_of == c]
        e_c = edges[within & (e_chrom_s == c)]
        n_c.to_csv(d / "nodes.csv.gz", index=False)
        e_c.to_csv(d / "edges.csv.gz", index=False)
        per_chrom[c] = {"nodes": int(len(n_c)), "edges": int(len(e_c))}
    (out / "chroms.txt").write_text(",".join(chroms) + "\n")
    info = {"builder_dir": str(src.resolve()), "sha256_builder_nodes": sha256(src / "nodes.csv"),
            "sha256_builder_edges": sha256(src / "edges.csv"), "sentinel_removed": bool(sentinel.any() and not args.keep_sentinel),
            "n_sentinel_edges_removed": n_sentinel_edges, "n_nodes": int(len(nodes)), "n_patients": int(len(patients)),
            "n_edges_total": int(len(edges)), "n_edges_within_chromosome": int(within.sum()),
            "n_edges_between_chromosomes_dropped": int((~within).sum()), "chromosomes": per_chrom}
    (out / "conversion.json").write_text(json.dumps(info, indent=2))
    print(f"{len(nodes):,} nodes on {len(chroms)} chromosomes, {len(patients)} patients; "
          f"{int(within.sum()):,} within-chromosome edges written, "
          f"{int((~within).sum()):,} between-chromosome edges dropped -> {out}")


if __name__ == "__main__":
    main()

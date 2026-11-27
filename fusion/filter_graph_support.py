"""Remove graph edges whose nodes are not eligible among the fold's TRAINING patients (run after 1_build_graph.py).

The SynthTAN builder listed candidate edges only between nodes with at least --min-support carriers and at least
--min-support non-carriers among ALL patients of the cohort (SynthTAN config builder min_support = 3).
1_build_graph.py in train-only mode recomputes weights and lift from the training patients but keeps every listed
pair, so an edge can survive whose node is eligible only thanks to test patients. This script applies the same
eligibility rule to the training patients (support_train in nodes.csv.gz) and drops the edges that fail it.
Edges that would be eligible among the training patients but were not listed by the builder cannot be added:
the candidate list itself stays a cohort-wide choice.

Rewrites <graph-dir>/graph.npz (the original is kept as graph_unfiltered.npz), writes <graph-dir>/graph_filter.json
and adds a "fusion_edge_filter" entry to graph_info.json (whose own edge count describes the unfiltered graph).
Run it before 3_node_embeddings.py. Once graph_filter.json exists it refuses to run again; if an earlier run was
interrupted (graph_unfiltered.npz but no graph_filter.json), it starts again from graph_unfiltered.npz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--min-support", type=int, default=3, help="min carriers and min non-carriers (training)")
    args = ap.parse_args(argv)
    g = Path(args.graph_dir)
    if (g / "graph_filter.json").exists():
        raise SystemExit(f"{g} is already filtered")
    if not (g / "graph_unfiltered.npz").exists():                 # first run: keep the original
        (g / "graph.npz").rename(g / "graph_unfiltered.npz")
    nodes = pd.read_csv(g / "nodes.csv.gz")
    if "support_train" not in nodes.columns:
        raise SystemExit(f"{g}/nodes.csv.gz has no support_train column: the graph was not built in train-only mode")
    n_train = len([x for x in (g / "train_individuals.txt").read_text().split("\n") if x.strip()])
    sup = nodes["support_train"].to_numpy()
    ok = np.minimum(sup, n_train - sup) >= args.min_support
    A = sp.load_npz(g / "graph_unfiltered.npz").tocoo()
    keep = ok[A.row] & ok[A.col]
    B = sp.csr_matrix((A.data[keep], (A.row[keep], A.col[keep])), shape=A.shape, dtype=A.dtype)
    sp.save_npz(g / "graph_tmp.npz", B)
    (g / "graph_tmp.npz").replace(g / "graph.npz")
    info = {"min_support": args.min_support, "n_train": n_train, "nodes_eligible": int(ok.sum()),
            "nodes_total": int(len(ok)), "edges_before": int(A.nnz // 2), "edges_after": int(B.nnz // 2),
            "edges_removed": int((A.nnz - B.nnz) // 2)}
    gi = json.loads((g / "graph_info.json").read_text())
    gi["fusion_edge_filter"] = info | {"note": "graph.npz is filtered; the edge counts above are unfiltered"}
    (g / "graph_info.json.tmp").write_text(json.dumps(gi, indent=2))
    (g / "graph_info.json.tmp").replace(g / "graph_info.json")
    (g / "graph_filter.json.tmp").write_text(json.dumps(info, indent=2))
    (g / "graph_filter.json.tmp").replace(g / "graph_filter.json")       # completion marker, written last
    print(f"{g.name}: {info['edges_removed']:,} of {info['edges_before']:,} edges removed "
          f"({len(ok) - int(ok.sum())} of {len(ok)} nodes not eligible among {n_train} training patients)")


if __name__ == "__main__":
    main()

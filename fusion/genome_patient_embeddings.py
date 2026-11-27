"""Per-patient genome embeddings from a fold's genome graph: node2vec / gnn / xgboost means, or Evo 2.

The repository's genome steps stop at node embeddings (3_node_embeddings.py, one row per graph node) or at a
fine-tuned Evo 2 model (4a_finetune_evo2_embeddings.py, a .pt file). Late fusion needs one vector per patient.

METHODS
  --method node2vec|gnn|xgboost   mean of the patient's node embeddings (node_emb_<method>.npy, standardized per
                                  dimension over all nodes, as step 4a does), over every node the patient carries
  --method evo2 --evo2-pt FILE    the fine-tuned Evo 2 model from step 4a: for each chromosome, the patient's
                                  nodes in genomic order are fed to Evo 2 exactly as in step 4a's evaluation
                                  (node embedding -> proj_in + chromosome vector, in place of token embeddings;
                                  centered window of --max-len nodes), the read-out layer's states are averaged
                                  over the nodes fed in (all of the patient's nodes unless a chromosome has more
                                  than --max-len). Needs the evo2 package and a GPU (see install_evo2.sh).
  Nodes whose id matches --exclude-nodes (default: the SynthTAN builder's chrM ALL region, carried by everyone)
  are left out of both. A patient without any remaining node gets a zero vector (count in the .json).

INPUT   --graph-dir   a fold's graph folder from 1_build_graph.py (nodes.csv.gz, individuals.txt, carriers.npz),
                      with node_emb_<method>.npy from 3_node_embeddings.py
OUTPUT  --out FILE.csv   eid, g0 ... g<d-1>, one row per patient of the graph (training and test patients)
        FILE.json        method, sizes, the training set the embeddings came from, nodes per patient

The node embeddings (and Evo 2) must have been trained on the fold's training patients only
(--train-individuals <run>_fold<k>_train.txt in steps 1, 3 and 4a). --expect-train checks the graph's training
list, the node embeddings' and the Evo 2 model's training-set ids against that file; --expect-patients checks
that the graph has exactly the patients of the fusion folder's folds.csv.

USAGE
  python fusion/genome_patient_embeddings.py --graph-dir work/cohort_n200/graph_fold0 --method node2vec \\
      --expect-train fusion_inputs/cohort_n200/fold0/cohort_n200_fold0_train.txt \\
      --out fusion_inputs/cohort_n200/fold0/genome_embeddings_node2vec.csv
  python fusion/genome_patient_embeddings.py --graph-dir work/cohort_n200/graph_fold0 --method evo2 \\
      --evo2-pt work/cohort_n200/graph_fold0/evo2_next_node2vec.pt --expect-train ... \\
      --out fusion_inputs/cohort_n200/fold0/genome_embeddings_evo2.csv
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parent.parent


def load_step4a():
    """Import 4a_finetune_evo2_embeddings.py (its name starts with a digit) to reuse GenomeData and Evo2Runner."""
    spec = importlib.util.spec_from_file_location("step4a", REPO / "4a_finetune_evo2_embeddings.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_ids(path) -> list[str]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def train_set_id(ids) -> str:
    """The training-set id of steps 1-4a: first 8 hex digits of sha1 over the sorted ids joined by newlines."""
    return hashlib.sha1("\n".join(sorted(ids)).encode()).hexdigest()[:8]


def check_train(graph_dir: Path, expect: str | None, emb_info: Path | None) -> dict:
    info = {}
    gtrain = graph_dir / "train_individuals.txt"
    if expect:
        want = read_ids(expect)
        if not gtrain.exists():
            raise SystemExit(f"{graph_dir} has no train_individuals.txt: it was not built in train-only mode")
        have = read_ids(gtrain)
        if sorted(want) != sorted(have):
            raise SystemExit(f"{gtrain} does not match {expect}: this graph was built from other training "
                             "patients (stale or reused folder?)")
        if len(set(want)) != len(want):
            raise SystemExit(f"{expect} lists a patient twice")
        info["train_individuals"] = str(Path(expect).resolve())
        info["n_train"] = len(want)
        info["train_set_id"] = train_set_id(want)
    if emb_info is not None:
        if not emb_info.exists():
            raise SystemExit(f"missing {emb_info} (written by 3_node_embeddings.py)")
        e = json.loads(emb_info.read_text())
        info["node_embedding_train_source"] = e.get("train_source")
        if expect and e.get("train_set_id") != info["train_set_id"]:
            raise SystemExit(f"{emb_info}: node embeddings were trained on set {e.get('train_set_id')}, "
                             f"the fold's training list is set {info['train_set_id']}")
    return info


def excluded_mask(graph_dir: Path, pattern: str) -> np.ndarray:
    ids = pd.read_csv(graph_dir / "nodes.csv.gz", usecols=["id"])["id"].astype(str)
    return ids.str.contains(pattern, regex=True).to_numpy() if pattern else np.zeros(len(ids), bool)


def node_mean(graph_dir: Path, method: str, exclude: np.ndarray):
    nodes = pd.read_csv(graph_dir / "nodes.csv.gz")
    Z = np.load(graph_dir / f"node_emb_{method}.npy").astype(np.float64)
    if len(Z) != len(nodes):
        raise SystemExit(f"node_emb_{method}.npy has {len(Z)} rows, nodes.csv.gz has {len(nodes)}")
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)                    # as in step 4a
    C = sp.load_npz(graph_dir / "carriers.npz").tocsr().astype(np.float64)   # [N, I]
    C = sp.diags((~exclude).astype(np.float64)) @ C                        # drop excluded nodes
    n_tok = np.asarray(C.sum(axis=0)).ravel()
    G = (C.T @ Z) / np.maximum(n_tok, 1)[:, None]               # [I, dim]; zero vector if no nodes
    return G.astype(np.float32), n_tok.astype(int)


def evo2_package(mock: bool) -> dict:
    import importlib.util as iu
    import evo2
    # the real package has an evo2.models module and no IS_MOCK; anything else counts as a stand-in
    is_mock = iu.find_spec("evo2.models") is None or bool(getattr(evo2, "IS_MOCK", False))
    if is_mock != mock:
        raise SystemExit(f"evo2 resolves to {evo2.__file__} ({'the fake' if is_mock else 'a real'} package); "
                         f"{'use --mock only with the fake package' if mock else 'pass --mock to use the fake one'}")
    try:
        version = metadata.version("evo2")
    except metadata.PackageNotFoundError:
        version = None
    return {"evo2_module": evo2.__file__, "evo2_version": version, "evo2_is_mock": is_mock}


def evo2_export(graph_dir: Path, pt_path: str, max_len: int | None, device_note: str, exclude: np.ndarray):
    import torch
    s4a = load_step4a()
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    if ck.get("mode") != "embeddings":
        raise SystemExit(f"{pt_path} is not a step-4a embeddings model (mode {ck.get('mode')!r})")
    cfg = ck.get("config", {})
    args = SimpleNamespace(graph_dir=str(graph_dir), method=ck["graph_method"], chroms=ck["chroms"],
                           evo2_model=ck["evo2_model"], layer=ck["layer"], train_blocks=0,
                           embed_module=cfg.get("embed_module"))
    max_len = max_len or cfg.get("max_len", 1024)
    emb_json = graph_dir / f"node_emb_{ck['graph_method']}.json"
    if not emb_json.exists() or json.loads(emb_json.read_text()).get("train_set_id") != ck.get("train_set_id"):
        raise SystemExit(f"{emb_json} does not belong to the training set of {pt_path}")
    data = s4a.GenomeData(args)
    rnr = s4a.Evo2Runner(args)
    res = rnr.net.load_state_dict(ck["trained_blocks"], strict=False)
    if res.unexpected_keys:
        raise SystemExit(f"trained Evo 2 blocks do not fit this model: {res.unexpected_keys[:3]}")
    missing_trained = set(ck["trained_blocks"]) & set(res.missing_keys)
    if missing_trained:
        raise SystemExit(f"trained Evo 2 blocks were not loaded: {sorted(missing_trained)[:3]}")
    last = int(re.match(r"blocks\.(\d+)\.", ck["layer"]).group(1))
    tb = int(cfg.get("train_blocks", 0))
    expected = {f"blocks.{i}.{n}" for i in range(max(0, last - tb + 1), last + 1)
                for n, _ in rnr.net.blocks[i].named_parameters()} if tb > 0 else set()
    if set(ck["trained_blocks"]) != expected:
        raise SystemExit(f"{pt_path}: its trained blocks do not match train_blocks={tb} up to {ck['layer']}")
    n_loaded = len(ck["trained_blocks"])
    dev, dim = rnr.dev, data.Z.shape[1]
    nn = torch.nn
    proj_in = nn.Sequential(nn.Linear(dim, rnr.hidden_in), nn.LayerNorm(rnr.hidden_in)).to(dev)
    proj_in.load_state_dict(ck["proj_in"])
    chrom_emb = nn.Embedding(data.n_chrom_slots, rnr.hidden_in).to(dev)
    chrom_emb.load_state_dict(ck["chrom_emb"])
    rnr.net.eval(); proj_in.eval()
    Z_t = torch.from_numpy(data.Z).to(dev)
    n_ind = data.by_ind.shape[0]
    out = np.zeros((n_ind, rnr.hidden_out), dtype=np.float64)
    n_tok = np.zeros(n_ind, dtype=int)
    with torch.no_grad():
        for i in range(n_ind):
            for chrom in data.chroms:
                node_idx, _ = data.tokens(chrom, i)
                node_idx = node_idx[~exclude[node_idx]]
                n = len(node_idx)
                if n == 0:
                    continue
                s = 0 if n <= max_len else (n - max_len) // 2         # step 4a's evaluation window
                tok = torch.from_numpy(node_idx[s:s + max_len]).to(dev)
                h = rnr.run_soft(proj_in(Z_t[tok]) + chrom_emb.weight[data.cidx[chrom]])
                out[i] += h.double().sum(0).cpu().numpy()
                n_tok[i] += len(tok)
    G = out / np.maximum(n_tok, 1)[:, None]
    meta = {"evo2_model": ck["evo2_model"], "layer": ck["layer"], "graph_method": ck["graph_method"],
            "objective": ck.get("objective"), "trained_blocks_loaded": n_loaded, "max_len": max_len,
            "evo2_train_set_id": ck.get("train_set_id"), "evo2_train_source": ck.get("train_source"),
            "n_evo2_train_individuals": len(ck.get("train_individuals", [])) + len(ck.get("val_individuals", [])),
            "device": device_note}
    return G.astype(np.float32), n_tok, meta, set(ck.get("train_individuals", [])) | set(ck.get("val_individuals", []))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--method", required=True, choices=["node2vec", "gnn", "xgboost", "evo2"])
    ap.add_argument("--evo2-pt", help="step-4a model (.pt), for --method evo2")
    ap.add_argument("--max-len", type=int, help="max nodes per chromosome sequence (evo2; default: the model's)")
    ap.add_argument("--expect-train", help="the fold's training list; must equal the graph's train_individuals.txt")
    ap.add_argument("--expect-patients", help="folds.csv (eid, fold) of the fusion folder; must list the graph's patients")
    ap.add_argument("--exclude-nodes", default=r"^chrM_1-2_", help="regex of node ids to leave out ('' = none)")
    ap.add_argument("--mock", action="store_true", help="evo2 is the fake test package (fusion/tests/mock_evo2)")
    ap.add_argument("--out", required=True, help="output CSV (eid, g0 ...)")
    args = ap.parse_args(argv)
    graph_dir = Path(args.graph_dir)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    inds = (graph_dir / "individuals.txt").read_text().split("\n")
    inds = [x for x in inds if x]
    if args.expect_patients:
        want = pd.read_csv(args.expect_patients, dtype={"eid": str})["eid"]
        if want.duplicated().any() or sorted(want) != sorted(inds) or len(set(inds)) != len(inds):
            raise SystemExit(f"the graph's patients differ from {args.expect_patients} (other cohort?)")
    exclude = excluded_mask(graph_dir, args.exclude_nodes)

    if args.method == "evo2":
        if not args.evo2_pt:
            raise SystemExit("--method evo2 needs --evo2-pt")
        info = check_train(graph_dir, args.expect_train, None) | evo2_package(args.mock)
        G, n_tok, meta, evo_train = evo2_export(graph_dir, args.evo2_pt, args.max_len,
                                                os.environ.get("CUDA_VISIBLE_DEVICES", "?"), exclude)
        if args.expect_train:
            if not evo_train <= set(read_ids(args.expect_train)):
                raise SystemExit(f"{args.evo2_pt} was trained on patients outside {args.expect_train}")
            if meta["evo2_train_set_id"] != info["train_set_id"]:
                raise SystemExit(f"{args.evo2_pt} was trained on set {meta['evo2_train_set_id']}, "
                                 f"the fold's training list is set {info['train_set_id']}")
        info |= meta | {"source_sha256": sha256(args.evo2_pt)}
    else:
        info = check_train(graph_dir, args.expect_train, graph_dir / f"node_emb_{args.method}.json")
        G, n_tok = node_mean(graph_dir, args.method, exclude)
        info["source_sha256"] = sha256(graph_dir / f"node_emb_{args.method}.npy")
    if len(G) != len(inds):
        raise SystemExit(f"{len(G)} patient vectors but {len(inds)} individuals in the graph")
    if not np.isfinite(G).all():
        raise SystemExit("non-finite values in the patient embeddings")

    table = pd.DataFrame(G, index=pd.Index(inds, name="eid"), columns=[f"g{j}" for j in range(G.shape[1])])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    table.to_csv(tmp)
    tmp.replace(out)
    info |= {"method": args.method, "graph_dir": str(graph_dir.resolve()), "n_patients": len(inds),
             "excluded_nodes": int(exclude.sum()), "exclude_pattern": args.exclude_nodes,
             "dim": int(G.shape[1]), "patients_without_nodes": int((n_tok == 0).sum()),
             "nodes_per_patient": {"min": int(n_tok.min()), "median": float(np.median(n_tok)), "max": int(n_tok.max())}}
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(f"{args.method}: {len(inds)} patients x {G.shape[1]} dims -> {out}"
          + (f" ({info['patients_without_nodes']} patients without nodes get a zero vector)"
             if info["patients_without_nodes"] else ""))


if __name__ == "__main__":
    main()

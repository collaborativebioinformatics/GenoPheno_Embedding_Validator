"""
Evo 2 on haplograph node embeddings: shared by 4a_finetune_evo2_embeddings.py (training),
4b_finetune_evo2_hash.py and 5_validate_evo2.py (querying the fine-tuned model on the test set).

  GenomeData        node embeddings + each individual's clusters per chromosome, in genomic order
  Evo2Runner        Evo 2 through the `evo2` package (Arc Institute; plain PyTorch, CUDA)
  BioNeMoEvo2Runner Evo 2 through NVIDIA BioNeMo / NeMo 2 (Megatron Core), from a NeMo2 checkpoint
  MockEvo2Runner    a small causal transformer with the same interface, so the whole pipeline can
                    be checked on a CPU without downloading Evo 2 (--mock-evo2; NOT Evo 2)
  NextClusterModel  the new layers around the backbone (input projection, chromosome vectors, head)
                    and the next-cluster loss
  load_finetuned    rebuild a fine-tuned model from a step-4 checkpoint

All runners expose: net (the backbone module), dev, hidden_in, hidden_out, trainable (list of
(name, parameter) for the unfrozen blocks, names relative to `net`), and run_soft(vectors [T,
hidden_in]) -> hidden states [T, hidden_out]. Evo2Runner and MockEvo2Runner also have run_ids(tokens).
"""
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from haplo_common import chrom_key, die, log


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class GenomeData:
    """Node embeddings + each individual's clusters per chromosome, in genomic order."""

    def __init__(self, graph_dir, method, chroms=None):
        nodes = pd.read_csv(os.path.join(graph_dir, "nodes.csv.gz"))
        emb_file = os.path.join(graph_dir, f"node_emb_{method}.npy")
        Z = np.load(emb_file).astype(np.float32)
        if len(Z) != len(nodes):
            die(f"node_emb_{method}.npy has {len(Z)} rows, nodes.csv.gz has {len(nodes)}")
        self.emb_sha1 = hashlib.sha1(np.ascontiguousarray(Z).tobytes()).hexdigest()[:12]
        self.Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)

        present = sorted(nodes["chrom"].unique(), key=chrom_key)
        self.chroms = list(chroms or present)
        unknown = set(self.chroms) - set(present)
        if unknown:
            die(f"chromosomes not in this graph: {sorted(unknown)}; graph has {present}")
        self.present = present
        self.cidx = {c: present.index(c) for c in self.chroms}
        self.n_chrom_slots = len(present)

        code = nodes["chrom"].map({c: i for i, c in enumerate(present)}).to_numpy()
        start = nodes["start"].fillna(0).to_numpy()
        self.order = np.lexsort((np.arange(len(nodes)), start, code))   # rank -> node index
        code_sorted = code[self.order]
        self.bounds = {c: (int(np.searchsorted(code_sorted, i)), int(np.searchsorted(code_sorted, i, "right")))
                       for i, c in enumerate(present)}
        C = sp.load_npz(os.path.join(graph_dir, "carriers.npz")).tocsr()
        self.by_ind = C[self.order].T.tocsr()                            # [I, N], columns = ranks
        self.by_ind.sort_indices()

    def tokens(self, chrom, i):
        """(node indices, positions among the chromosome's nodes) of individual i, genomic order."""
        row = self.by_ind.indices[self.by_ind.indptr[i]:self.by_ind.indptr[i + 1]]
        s, e = self.bounds[chrom]
        lo, hi = np.searchsorted(row, [s, e])
        ranks = row[lo:hi].astype(np.int64)
        return self.order[ranks], ranks - s

    def candidates(self, chrom):
        s, e = self.bounds[chrom]
        return self.order[s:e]

    def examples(self, individuals):
        """(chrom, individual index) pairs with at least 2 clusters, i.e. one next-cluster target."""
        return [(c, int(i)) for c in self.chroms for i in individuals if len(self.tokens(c, i)[0]) >= 2]


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

class StopForward(Exception):
    pass


def unfreeze_last(blocks, last, n, prefix):
    """Unfreeze blocks[last - n + 1 .. last]; return [(name relative to net, parameter)]."""
    out = []
    for i in range(max(0, last - n + 1), last + 1) if n > 0 else []:
        for name, p in blocks[i].named_parameters():
            p.requires_grad_(True)
            out.append((f"{prefix}.{i}.{name}", p))
    return out


def find_embedding_module(net, name):
    import torch.nn as nn
    mods = dict(net.named_modules())
    if name:
        if name not in mods:
            die(f"--embed-module {name!r} not found in the Evo 2 model")
        return mods[name]
    for n, m in mods.items():
        if n.endswith("embedding_layer"):
            return m
    for n, m in mods.items():
        if isinstance(m, nn.Embedding):
            return m
    die("could not find Evo 2's token embedding layer; pass --embed-module")


def convert_inference_tensors(net):
    """The evo2 package loads its weights under torch.inference_mode(), which makes them
    'inference tensors' that autograd can neither train nor save for backward (needed even for
    frozen blocks, so gradients can flow back to the input projection). Replace every parameter,
    buffer and plain tensor attribute that is an inference tensor with a normal clone, one tensor
    at a time (peak extra memory = one tensor). Tied parameters stay tied."""
    import torch
    memo, n = {}, 0

    def fresh(t, as_param):
        nonlocal n
        if id(t) not in memo:
            c = t.clone()                      # outside inference mode -> normal tensor
            memo[id(t)] = torch.nn.Parameter(c, requires_grad=False) if as_param else c
            n += 1
        return memo[id(t)]

    with torch.no_grad():
        for mod in net.modules():
            for name, p in list(mod._parameters.items()):
                if p is not None and p.is_inference():
                    mod._parameters[name] = fresh(p, True)
            for name, b in list(mod._buffers.items()):
                if b is not None and b.is_inference():
                    mod._buffers[name] = fresh(b, False)
            for name, v in list(vars(mod).items()):
                if (isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Parameter)
                        and v.is_inference()):
                    setattr(mod, name, fresh(v, False))
    memo.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n


class Evo2Runner:
    """Loads Evo 2 with the `evo2` package, unfreezes the last train_blocks blocks up to `layer`, and
    runs the model up to `layer` on its own tokens (run_ids) or on replacement input vectors (run_soft)."""

    backend = "evo2"

    def __init__(self, evo2_model="evo2_7b_base", layer="blocks.28.mlp.l3", train_blocks=0, embed_module=None):
        import torch
        if not torch.cuda.is_available():
            die("Evo 2 (the evo2 package) needs an NVIDIA GPU with CUDA. To check the pipeline on a CPU, "
                "use --mock-evo2 (a small stand-in model, not Evo 2).")
        try:
            from evo2 import Evo2
        except ImportError as e:
            die(f"could not import evo2 ({e}); install it with install_evo2.sh (see DOCUMENTATION.md)")
        self.torch = torch
        self.evo = Evo2(evo2_model)
        net = self.net = self.evo.model
        self.dev = next(net.parameters()).device
        n_conv = convert_inference_tensors(net)
        if n_conv:
            log(f"  converted {n_conv:,} inference-mode tensors from the evo2 loader to trainable tensors")

        m = re.match(r"blocks\.(\d+)\.", layer)
        if not m:
            die("--layer must look like blocks.<i>.<...>, e.g. blocks.28.mlp.l3")
        last = int(m.group(1))
        for p in net.parameters():
            p.requires_grad_(False)
        self.trainable = unfreeze_last(net.blocks, last, train_blocks, "blocks")

        self.state = {"soft": None, "h": None, "embed_shape": None}
        find_embedding_module(net, embed_module).register_forward_hook(self._embed_hook)
        mods = dict(net.named_modules())
        if layer not in mods:
            die(f"--layer {layer!r} not found in {evo2_model}")
        mods[layer].register_forward_hook(self._layer_hook)

        with torch.no_grad():                  # discover sizes with one plain forward pass
            self._forward(torch.zeros(1, 8, dtype=torch.int, device=self.dev))
        self.hidden_in = self.state["embed_shape"][-1]
        self.hidden_out = self.state.pop("h").shape[-1]
        self.state["h"] = None
        n_train = sum(p.numel() for _, p in self.trainable) / 1e6
        blocks = f"{last - train_blocks + 1}-{last}" if self.trainable else "none"
        log(f"  {evo2_model}: hidden {self.hidden_in}, read-out {layer}, "
            f"unfrozen blocks {blocks} ({n_train:.0f}M params)")

    def _embed_hook(self, _m, _inp, out):
        self.state["embed_shape"] = tuple(out.shape)
        soft = self.state["soft"]
        if soft is None:
            return None
        if soft.shape != out.shape:
            raise RuntimeError(f"soft input {tuple(soft.shape)} != embedding output {tuple(out.shape)}")
        return soft.to(out.dtype)

    def _layer_hook(self, _m, _inp, out):
        self.state["h"] = out[0] if isinstance(out, tuple) else out
        raise StopForward

    def _forward(self, ids):
        try:
            self.net(ids)
        except StopForward:
            pass

    def run_ids(self, ids):
        """ids: int tensor [T] of Evo 2 tokens -> hidden states at --layer, [T, hidden_out] float32."""
        self._forward(ids.unsqueeze(0))
        return self.state.pop("h").float().squeeze(0)

    def run_soft(self, soft):
        """soft: [T, hidden_in] replacement input vectors -> [T, hidden_out] float32."""
        self.state["soft"] = soft.unsqueeze(0)
        dummy = self.torch.zeros(1, soft.shape[0], dtype=self.torch.int, device=self.dev)
        try:
            self._forward(dummy)
        finally:
            self.state["soft"] = None
        return self.state.pop("h").float().squeeze(0)

    def describe(self):
        return {}


def find_megatron_hyena(obj):
    """The Megatron Core Hyena module (has .embedding and .decoder.layers) inside the NeMo wrappers."""
    import torch.nn as nn
    mods = obj.modules() if isinstance(obj, nn.Module) else []
    for m in mods:
        dec = getattr(m, "decoder", None)
        if hasattr(m, "embedding") and dec is not None and hasattr(dec, "layers"):
            return m
    die("could not find the Megatron Hyena module (with .embedding and .decoder.layers) inside the "
        "loaded model - this BioNeMo version may structure it differently")


class BioNeMoEvo2Runner:
    """Loads a NeMo2 Evo 2 checkpoint through BioNeMo / NeMo, unfreezes the last train_blocks decoder
    layers up to layer_index, and runs the Megatron model up to that layer on replacement input
    vectors (passed as `decoder_input`, which skips the nucleotide embedding).

    EXPERIMENTAL: written against the BioNeMo Evo 2 API (HYENA_MODEL_OPTIONS, HyenaModel, the
    byte-level tokenizer and the Megatron settings of BioNeMo's predict_evo2) but not yet run on a
    GPU. BioNeMo's own fine-tuning (train_evo2) only supports next-nucleotide training on DNA, so
    BioNeMo is used to LOAD the model and the training loop is plain PyTorch. Single GPU (tensor /
    pipeline parallel = 1), bf16 autocast; unfrozen layers stay bf16 and are updated by torch AdamW.
    Parts most likely to need adjusting for your BioNeMo version are marked VERSION-SENSITIVE."""

    backend = "bionemo"

    def __init__(self, ckpt_dir, model_size="7b", layer_index=28, train_blocks=0, fp8=False, pad_multiple=1):
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            die("BioNeMo Evo 2 needs an NVIDIA GPU with CUDA; for a CPU check use --mock-evo2")
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
            die(f"--ckpt-dir {ckpt_dir!r} is not a folder (a NeMo2 checkpoint with weights/ and context/)")

        # ---- VERSION-SENSITIVE: imports (paths as used by bionemo.evo2.run.predict)
        try:
            import nemo.lightning as nl
            from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS, HyenaModel
            from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
        except ImportError as e:
            die(f"could not import NeMo / BioNeMo ({e}). Run inside the BioNeMo Framework container "
                "(nvcr.io/nvidia/clara/bionemo-framework), see DOCUMENTATION.md")

        if model_size not in HYENA_MODEL_OPTIONS:
            die(f"--model-size {model_size!r} unknown; options: {sorted(HYENA_MODEL_OPTIONS)}")

        # ---- VERSION-SENSITIVE: model config
        try:
            config = HYENA_MODEL_OPTIONS[model_size](vortex_style_fp8=fp8)
        except TypeError:
            config = HYENA_MODEL_OPTIONS[model_size]()
        # plain-PyTorch training: no Megatron DDP, so no fused main_grad accumulation
        for name, value in {"gradient_accumulation_fusion": False, "sequence_parallel": False,
                            "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
                            "context_parallel_size": 1}.items():
            if hasattr(config, name):
                setattr(config, name, value)

        model = HyenaModel(config, tokenizer=get_nmt_tokenizer("byte-level"))
        trainer = nl.Trainer(
            accelerator="gpu", devices=1, num_nodes=1,
            strategy=nl.MegatronStrategy(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1,
                pipeline_dtype=torch.bfloat16, ckpt_load_optimizer=False, ckpt_save_optimizer=False,
                ckpt_load_strictness="log_all",
            ),
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", params_dtype=torch.bfloat16),
        )

        # ---- VERSION-SENSITIVE: load weights (NeMo 2 Fabric API)
        log(f"  loading NeMo2 checkpoint {ckpt_dir} (model size {model_size})")
        try:
            loaded = trainer.to_fabric().load_model(ckpt_dir, model)
        except Exception as e:
            die(f"loading the checkpoint failed: {type(e).__name__}: {e}\nCheck that --ckpt-dir is a NeMo2 "
                "checkpoint folder (with weights/ and context/) and that --model-size matches it.")
        mcore = self.net = find_megatron_hyena(loaded)
        self.dev = next(mcore.parameters()).device

        layers = mcore.decoder.layers
        n_layers = len(layers)
        last = layer_index if layer_index >= 0 else n_layers + layer_index
        if not 0 <= last < n_layers:
            die(f"--layer-index {layer_index} out of range (model has {n_layers} layers)")
        self.layer_index = last
        for p in mcore.parameters():
            p.requires_grad_(False)
        self.trainable = unfreeze_last(layers, last, train_blocks, "decoder.layers")

        self.state = {"h": None}
        layers[last].register_forward_hook(self._layer_hook)
        self.hidden_in = self.hidden_out = int(mcore.config.hidden_size)
        self.pad_multiple = max(1, pad_multiple)
        self.ckpt_dir, self.model_size = ckpt_dir, model_size
        n_train = sum(p.numel() for _, p in self.trainable) / 1e6
        blocks = f"{last - train_blocks + 1}-{last}" if self.trainable else "none"
        log(f"  BioNeMo Evo 2 {model_size}: {n_layers} layers, hidden {self.hidden_in}, read-out "
            f"decoder.layers.{last}, unfrozen layers {blocks} ({n_train:.0f}M params)")

    def _layer_hook(self, _m, _inp, out):
        self.state["h"] = out[0] if isinstance(out, (tuple, list)) else out
        raise StopForward

    def run_soft(self, soft):
        torch = self.torch
        T = soft.shape[0]
        pad = (-T) % self.pad_multiple                      # causal model: end padding doesn't
        if pad:                                             # change outputs at real positions
            soft = torch.cat([soft, soft.new_zeros(pad, soft.shape[1])])
        L = soft.shape[0]
        ids = torch.zeros(1, L, dtype=torch.long, device=self.dev)       # ignored: decoder_input given
        pos = torch.arange(L, device=self.dev).unsqueeze(0)
        # ---- VERSION-SENSITIVE: Megatron forward with decoder_input in [seq, batch, hidden] layout
        with torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                self.net(input_ids=ids, position_ids=pos, attention_mask=None,
                         decoder_input=soft.unsqueeze(1).to(torch.bfloat16))
            except StopForward:
                pass
        h = self.state.pop("h")
        self.state["h"] = None
        if h is None:
            raise RuntimeError("the read-out layer hook did not fire")
        h = h[:, 0] if h.shape[0] == L else h[0]            # [s, b, h] (Megatron) or [b, s, h]
        return h[:T].float()

    def describe(self):
        return {"ckpt_dir": self.ckpt_dir, "model_size": self.model_size, "layer_index": self.layer_index}


class MockEvo2Runner:
    """A small causal transformer standing in for Evo 2 (same interface), for CPU checks of the
    pipeline. Its 'pre-trained' weights are random; the whole backbone state is saved in the
    checkpoint so step 5 can rebuild exactly the same model."""

    backend = "mock"

    def __init__(self, hidden=64, n_layers=4, train_blocks=0, device=None, seed=0):
        import torch
        import torch.nn as nn
        from haplo_common import torch_device
        self.torch = torch
        self.dev = torch_device(device)
        g = torch.random.fork_rng(devices=[])
        with g:
            torch.manual_seed(seed)

            class Net(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.embedding_layer = nn.Embedding(256, hidden)
                    s.blocks = nn.ModuleList(
                        nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=2 * hidden, dropout=0.0,
                                                   batch_first=True, norm_first=True) for _ in range(n_layers))

                def forward(s, x):                    # x: [1, T, hidden]
                    T = x.shape[1]
                    mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
                    for blk in s.blocks:
                        x = blk(x, src_mask=mask)
                    return x

            self.net = Net().to(self.dev)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.trainable = unfreeze_last(self.net.blocks, n_layers - 1, train_blocks, "blocks")
        self.hidden_in = self.hidden_out = hidden
        self.config = {"hidden": hidden, "n_layers": n_layers, "seed": seed}
        log(f"  MOCK Evo 2 (small causal transformer, NOT Evo 2): hidden {hidden}, {n_layers} layers, "
            f"unfrozen {len(self.trainable) and train_blocks} block(s)")

    def run_ids(self, ids):
        return self.net(self.net.embedding_layer(ids.long().unsqueeze(0))).squeeze(0).float()

    def run_soft(self, soft):
        return self.net(soft.unsqueeze(0)).squeeze(0).float()

    def describe(self):
        return {"mock_config": self.config}


def add_backbone_args(ap):
    """Command-line options shared by every script that loads a backbone."""
    b = ap.add_argument_group("backbone (default: Evo 2 via the evo2 package)")
    b.add_argument("--bionemo", action="store_true",
                   help="load and run Evo 2 through NVIDIA BioNeMo from a NeMo2 checkpoint (--ckpt-dir)")
    b.add_argument("--mock-evo2", action="store_true",
                   help="small CPU stand-in model instead of Evo 2, to check the pipeline without a GPU")
    b.add_argument("--evo2-model", default="evo2_7b_base", help="evo2 package: evo2_7b_base (8K ctx) or evo2_7b")
    b.add_argument("--layer", default="blocks.28.mlp.l3", help="evo2 package: read-out layer; forward stops here")
    b.add_argument("--embed-module", help="evo2 package: name of Evo 2's token embedding module (auto-detected)")
    b.add_argument("--ckpt-dir", help="--bionemo: NeMo2 Evo 2 checkpoint folder (weights/ + context/)")
    b.add_argument("--model-size", default="7b", help="--bionemo: 7b (8K ctx), 7b_arc_longcontext, ...")
    b.add_argument("--layer-index", type=int, default=28, help="--bionemo: read-out decoder layer (negative = from end)")
    b.add_argument("--fp8", action="store_true", help="--bionemo: FP8 Hyena projections (Hopper GPU)")
    b.add_argument("--pad-multiple", type=int, default=1, help="--bionemo: pad sequences to a multiple of this")
    b.add_argument("--mock-hidden", type=int, default=64, help="--mock-evo2: hidden size")
    b.add_argument("--mock-layers", type=int, default=4, help="--mock-evo2: number of blocks")
    b.add_argument("--device", help="--mock-evo2: cpu / cuda (Evo 2 always uses the GPU)")


def make_runner(args, train_blocks):
    if args.bionemo and args.mock_evo2:
        die("--bionemo and --mock-evo2 are mutually exclusive")
    if args.mock_evo2:
        return MockEvo2Runner(args.mock_hidden, args.mock_layers, train_blocks, args.device, args.seed)
    if args.bionemo:
        if not args.ckpt_dir:
            die("--bionemo needs --ckpt-dir (a NeMo2 Evo 2 checkpoint; see DOCUMENTATION.md)")
        return BioNeMoEvo2Runner(args.ckpt_dir, args.model_size, args.layer_index, train_blocks,
                                 args.fp8, args.pad_multiple)
    return Evo2Runner(args.evo2_model, args.layer, train_blocks, args.embed_module)


# ---------------------------------------------------------------------------
# New layers and loss
# ---------------------------------------------------------------------------

class NextClusterModel:
    """Input projection (embedding dim -> backbone hidden) + per-chromosome vector + head that
    predicts the NEXT cluster along the chromosome (cosine similarity to every cluster of that
    chromosome, softmax with temperature), mirroring Evo 2's next-token pre-training."""

    def __init__(self, rnr, data, temperature=0.1, max_len=1024):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        self.rnr, self.data = rnr, data
        self.temperature, self.max_len = temperature, max_len
        dev, dim = rnr.dev, data.Z.shape[1]
        self.proj_in = nn.Sequential(nn.Linear(dim, rnr.hidden_in), nn.LayerNorm(rnr.hidden_in)).to(dev)
        self.chrom_emb = nn.Embedding(data.n_chrom_slots, rnr.hidden_in).to(dev)
        nn.init.normal_(self.chrom_emb.weight, std=0.02)
        self.head = nn.Sequential(nn.LayerNorm(rnr.hidden_out), nn.Linear(rnr.hidden_out, dim)).to(dev)
        self.modules = [self.proj_in, self.chrom_emb, self.head]
        self.Z_t = torch.from_numpy(data.Z).to(dev)
        self._cand = {}

    def candidates(self, chrom):
        if chrom not in self._cand:
            idx = self.torch.from_numpy(self.data.candidates(chrom)).to(self.rnr.dev)
            self._cand[chrom] = self.F.normalize(self.Z_t[idx], dim=-1)
        return self._cand[chrom]

    def window(self, n_tok, rng=None):
        """Training: random window of max_len clusters; evaluation (rng=None): the central one."""
        if n_tok <= self.max_len:
            return 0, n_tok
        s = int(rng.integers(0, n_tok - self.max_len + 1)) if rng is not None else (n_tok - self.max_len) // 2
        return s, s + self.max_len

    def logits(self, chrom, i, rng=None):
        """-> (logits [T-1, n clusters of chrom], targets [T-1]) for individual i on chrom."""
        torch = self.torch
        node_idx, local = self.data.tokens(chrom, i)
        s, e = self.window(len(node_idx), rng)
        tok = torch.from_numpy(node_idx[s:e]).to(self.rnr.dev)
        h = self.rnr.run_soft(self.proj_in(self.Z_t[tok]) + self.chrom_emb.weight[self.data.cidx[chrom]])
        pred = self.F.normalize(self.head(h[:-1]), dim=-1)                 # predict t+1 from <= t
        logits = pred @ self.candidates(chrom).T / self.temperature
        return logits, torch.from_numpy(local[s + 1:e]).to(self.rnr.dev)

    def loss(self, chrom, i, rng=None):
        """-> (mean cross-entropy, (n correct top-1, n predictions))."""
        logits, target = self.logits(chrom, i, rng)
        return self.F.cross_entropy(logits, target), ((logits.argmax(-1) == target).sum().item(), len(target))

    def train(self, mode=True):
        self.rnr.net.train(mode)
        for m in self.modules:
            m.train(mode)

    def state(self):
        return {"proj_in": self.proj_in.state_dict(), "chrom_emb": self.chrom_emb.state_dict(),
                "head": self.head.state_dict()}

    def load(self, meta):
        for name, m in (("proj_in", self.proj_in), ("chrom_emb", self.chrom_emb), ("head", self.head)):
            m.load_state_dict(meta[name])


def load_finetuned(meta, graph_dir, device=None, ckpt_dir=None):
    """Rebuild a step-4 model (checkpoint dict `meta`) on the graph in graph_dir.
    Returns (NextClusterModel in eval mode, GenomeData). Evo 2 / BioNeMo backbones need the GPU and
    the same base model as in training; the mock backbone is stored whole in the checkpoint."""
    import argparse
    import torch
    if meta.get("mode") != "embeddings" or meta.get("objective", "next") != "next":
        die("only next-cluster models from 4a_finetune_evo2_embeddings.py can be queried")
    cfg = meta.get("config", {})
    backend = meta.get("backend", "evo2")
    data = GenomeData(graph_dir, meta["graph_method"], meta.get("chroms"))
    from haplo_common import graph_id
    if meta.get("graph_id") and meta["graph_id"] != graph_id(graph_dir):
        raise ValueError(f"the model was trained on another build of this graph (graph_id {meta['graph_id']}, "
                         f"now {graph_id(graph_dir)}); rerun steps 3-4 on the current graph")
    if meta.get("embeddings_sha1") and meta["embeddings_sha1"] != data.emb_sha1:
        raise ValueError(f"node_emb_{meta['graph_method']}.npy changed since the model was trained (different "
                         "checksum); rerun step 4 on the current embeddings")
    if data.n_chrom_slots != meta.get("n_chrom_slots", data.n_chrom_slots):
        raise ValueError("this graph has a different set of chromosomes than the one the model was trained on")
    a = argparse.Namespace(
        bionemo=backend == "bionemo", mock_evo2=backend == "mock", seed=cfg.get("seed", 0), device=device,
        evo2_model=meta.get("evo2_model", "evo2_7b_base"), layer=meta.get("layer", "blocks.28.mlp.l3"),
        embed_module=cfg.get("embed_module"), ckpt_dir=ckpt_dir or meta.get("ckpt_dir"),
        model_size=meta.get("model_size", "7b"), layer_index=meta.get("layer_index", 28),
        fp8=cfg.get("fp8", False), pad_multiple=cfg.get("pad_multiple", 1),
        mock_hidden=(meta.get("mock_config") or {}).get("hidden", 64),
        mock_layers=(meta.get("mock_config") or {}).get("n_layers", 4))
    rnr = make_runner(a, train_blocks=0)
    tb = {k: v.to(rnr.dev) for k, v in (meta.get("trained_blocks") or {}).items()}
    if tb:
        missing = set(tb) - set(rnr.net.state_dict())
        if missing:
            die(f"{len(missing)} fine-tuned parameters do not exist in the backbone (e.g. {sorted(missing)[:2]})")
        with torch.no_grad():
            rnr.net.load_state_dict(tb, strict=False)
    model = NextClusterModel(rnr, data, meta.get("config", {}).get("temperature", 0.1),
                             meta.get("config", {}).get("max_len", 1024))
    model.load(meta)
    model.train(False)
    return model, data


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Training loop (shared by 4a and 4b)
# ---------------------------------------------------------------------------

class MasterWeights:
    """fp32 master copies of low-precision (bf16 / fp16) trainable parameters.

    The evo2 package and BioNeMo keep Evo 2 in bf16. An AdamW update of ~lr (1e-5) is far below
    bf16's resolution for weights of ordinary size (~2e-4 at 0.05), so updating bf16 weights in
    place rounds every step away and the 'fine-tuned' blocks never change. The optimizer therefore
    steps on fp32 copies, which are written back (rounded) to the model after every step."""

    def __init__(self, params):
        import torch
        self.torch = torch
        self.pairs, self.params = [], []
        for p in params:
            if p.dtype in (torch.bfloat16, torch.float16):
                m = p.detach().float().clone().requires_grad_(True)
                self.pairs.append((p, m))
                self.params.append(m)
            else:
                self.params.append(p)

    def grads_to_master(self):
        for p, m in self.pairs:
            m.grad = None if p.grad is None else p.grad.float()

    def master_to_model(self):
        with self.torch.no_grad():
            for p, m in self.pairs:
                p.copy_(m)


def fit(args, blocks, new_modules, n_examples, example_loss, evaluate, describe, set_train, rng, grad_check):
    """AdamW on the unfrozen backbone blocks (--lr, through fp32 master weights) and the new layers
    (--head-lr), gradient accumulation over --batch examples, gradient clipping at 1.
      example_loss(j) -> loss tensor of training example j;  evaluate() -> metrics dict (run without
      gradients in eval mode);  set_train(mode) switches every module;  grad_check = (description,
      parameters that must receive a gradient on the first step).
    Returns the evaluation metrics before and after training."""
    import time
    import torch
    new_params = [p for m in new_modules for p in m.parameters()]
    master = MasterWeights(blocks)
    if master.pairs:
        log(f"  {len(master.pairs)} low-precision trainable tensors: optimizing fp32 master copies")
    opt = torch.optim.AdamW([{"params": master.params, "lr": args.lr}, {"params": new_params, "lr": args.head_lr}],
                            weight_decay=0.01)

    def run_eval():
        set_train(False)
        with torch.no_grad():
            res = evaluate()
        set_train(True)
        return res

    before = run_eval()
    log(f"  before training: {describe(before)}")
    set_train(True)
    step, t0 = 0, time.time()
    batch = min(args.batch, n_examples)
    if batch < 1:
        die("no training examples")
    for epoch in range(args.epochs):
        order = rng.permutation(n_examples)
        for b in range(0, n_examples, batch):            # the last batch may be smaller
            opt.zero_grad()
            for p in blocks:
                p.grad = None
            chunk = order[b:b + batch]
            batch_loss = 0.0
            for j in chunk:                               # gradient accumulation over the batch
                loss = example_loss(int(j))
                (loss / len(chunk)).backward()
                batch_loss += loss.item() / len(chunk)
            if not np.isfinite(batch_loss):
                die(f"loss became {batch_loss} at step {step}; lower --lr / --head-lr")
            if step == 0:
                what, params = grad_check
                if all(p.grad is None for p in params):
                    die(f"no gradient reached {what}: this Evo 2 version's forward pass is not differentiable "
                        "here. Try --bionemo.")
                if blocks and all(p.grad is None for p in blocks):
                    die("no gradient reached the unfrozen Evo 2 blocks; try --train-blocks 0 "
                        "(trains only the new layers through frozen Evo 2)")
            master.grads_to_master()
            torch.nn.utils.clip_grad_norm_(master.params + new_params, 1.0)
            opt.step()
            master.master_to_model()
            step += 1
            if step % args.log_every == 0:
                log(f"  epoch {epoch + 1} step {step}: loss {batch_loss:.4f}  {(time.time() - t0) / step:.2f}s/step")
            if step % args.eval_every == 0:
                log(f"  [eval] {describe(run_eval())}")
    after = run_eval()
    log(f"  final: {describe(after)}  ({step} steps)")
    log(f"  (before training: {describe(before)})")
    return before, after


def config_of(args):
    return {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, list, type(None)))}


"""
Backbones, data and model shared by step 3 (evo2_seq embeddings), 4a / 4b (training) and 6
(querying the fine-tuned model on the test set).

  GenomeData        node embeddings + token sequences per chromosome, in genomic order: one
                    haplotype = one token per haploblock (--units haplotype), or one individual =
                    every cluster it carries (--units individual)
  Evo2Runner        Evo 2 through the `evo2` package (Arc Institute; plain PyTorch, CUDA)
  BioNeMoEvo2Runner Evo 2 through NVIDIA BioNeMo / NeMo 2 (Megatron Core), from a NeMo2 checkpoint
  SmallRunner       a causal transformer trained from scratch (--backbone small): the baseline that
                    tells whether Evo 2's pretraining helps, and a CPU-sized model for checks
  HaploModel        the new layers around a backbone (input projection, chromosome vectors, mask
                    vector, head) and the loss of the next-block or masked-block objective
  PhenotypeModel    the 4b model: both haplotypes per chromosome (or a hash), attention pooling, phenotype head
  load_finetuned / load_phenotype_model   rebuild a 4a / 4b checkpoint for step 6 (shared checks: rebuild)
  fit               the training loop of 4a and 4b (best-validation checkpoint, --resume)

Every runner exposes: net (the backbone module), dev, hidden_in, hidden_out, trainable (list of
(name, parameter) of the parameters that train, names relative to `net`), run_soft(vectors [T,
hidden_in]) -> hidden states [T, hidden_out], and (evo2, small) run_ids(byte tokens [T]).
"""
import copy
import hashlib
import os
import re
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from haplo_common import chrom_key, die, log, read_graph_individuals

BACKBONES = ("evo2", "bionemo", "small")
HEADS = ("scaled-cosine", "table", "cosine")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class GenomeData:
    """Node embeddings + token sequences per chromosome, in genomic order.

    units "haplotype": example = one haplotype on one chromosome (row of haplotypes.npz), exactly one
    token per haploblock, so token offset = haploblock offset. units "individual": example = one
    individual on one chromosome, every cluster it carries (1 or 2 per haploblock, in file order).
    min_support: clusters carried by fewer than this many training haplotypes (individuals for
    units "individual") of the individuals `support_from` (step 4a's own training individuals) are
    merged into one token per haploblock (embedding = mean of its members).
    Every embedding gets appended: sin/cos of the cluster's genomic start at pos_freqs log-spaced
    wavelengths (10 kb .. the largest start) and the haploblock's log length, standardized."""

    def __init__(self, graph_dir, method, chroms, min_support, support_from, pos_freqs=16, units="haplotype"):
        nodes = pd.read_csv(os.path.join(graph_dir, "nodes.csv.gz"))
        Z = np.load(os.path.join(graph_dir, f"node_emb_{method}.npy")).astype(np.float32)
        if len(Z) != len(nodes):
            die(f"node_emb_{method}.npy has {len(Z)} rows, nodes.csv.gz has {len(nodes)}")
        self.emb_sha1 = hashlib.sha1(np.ascontiguousarray(Z).tobytes()).hexdigest()[:12]
        start = nodes["start"].to_numpy(dtype=np.float64)
        extra = [np.log(np.fmax(nodes["end"].to_numpy(dtype=np.float64) - start, 1))[:, None]]
        if pos_freqs:
            top = np.fmax(np.nanmax(start) if np.isfinite(start).any() else 0, 2e4)
            waves = np.geomspace(1e4, top, pos_freqs) if pos_freqs > 1 else np.array([top])
            angle = start[:, None] / waves * 2 * np.pi
            extra += [np.sin(angle), np.cos(angle)]
        P = np.hstack(extra)
        P = np.nan_to_num((P - np.nanmean(P, 0)) / (np.nanstd(P, 0) + 1e-6))     # unknown start -> 0
        self.Z = np.hstack([(Z - Z.mean(0)) / (Z.std(0) + 1e-6), P]).astype(np.float32)
        self.n_emb = Z.shape[1]                    # columns of the node embedding; the rest say where
        self.units = units

        present = sorted(nodes["chrom"].unique(), key=chrom_key)
        self.chroms = list(chroms or present)
        unknown = set(self.chroms) - set(present)
        if unknown:
            die(f"chromosomes not in this graph: {sorted(unknown)}; graph has {present}")
        self.present, self.n_chrom_slots = present, len(present)
        self.cidx = {c: present.index(c) for c in present}
        code = nodes["chrom"].map(self.cidx).to_numpy()
        self.order = np.lexsort((np.arange(len(nodes)), nodes["start"].fillna(0).to_numpy(), code))
        code_sorted = code[self.order]
        self.bounds = {c: (int(np.searchsorted(code_sorted, i)), int(np.searchsorted(code_sorted, i, "right")))
                       for i, c in enumerate(present)}
        self.region_code, self.regions = pd.factorize(nodes["region"])
        self.chrom_of = nodes["chrom"].to_numpy()
        n_ind = len(read_graph_individuals(graph_dir))
        if units == "haplotype":
            f = os.path.join(graph_dir, "haplotypes.npz")
            if not os.path.exists(f):
                die(f"{f} not found: --units haplotype needs step 1 --haplotypes and a rebuilt graph (step 2); "
                    "or use --units individual")
            h = np.load(f)
            self.H, block_chrom = h["node"], h["block_chrom"]
            if self.H.shape[0] != 2 * n_ind:
                die("haplotypes.npz and individuals.txt disagree; rebuild the graph with step 2")
            self.block_cols = {c: np.where(block_chrom == c)[0] for c in present}
            rows = np.ravel([[2 * i, 2 * i + 1] for i in support_from]).astype(np.int64)
            support = np.bincount(self.H[rows].ravel(), minlength=len(nodes)) if len(rows) else np.zeros(len(nodes))
        else:
            C = sp.load_npz(os.path.join(graph_dir, "carriers.npz")).tocsr()
            if C.shape[1] != n_ind:
                die("individuals.txt and carriers.npz disagree; rebuild the graph with step 2")
            self.by_ind = C[self.order].T.tocsr()                          # [I, N], columns = ranks
            self.by_ind.sort_indices()
            support = np.asarray(C[:, support_from].sum(axis=1)).ravel()
        self.support = support
        self.rare = support < min_support
        self.canon = np.arange(len(nodes))                    # node -> the node standing in for it
        rank = np.empty(len(nodes), np.int64)
        rank[self.order] = np.arange(len(nodes))
        rare = np.where(self.rare)[0]
        if len(rare):
            first = pd.Series(rank[rare]).groupby(self.region_code[rare]).transform("min").to_numpy()
            self.canon[rare] = self.order[first]              # first rare cluster of the haploblock
            reps, inv = np.unique(self.canon[rare], return_inverse=True)
            sums = np.zeros((len(reps), self.Z.shape[1]), np.float64)
            np.add.at(sums, inv, self.Z[rare])
            self.Z[reps] = (sums / np.bincount(inv)[:, None]).astype(np.float32)
        self.local = np.full(len(nodes), -1, np.int64)        # token -> index among its chromosome's tokens
        self.cands = {}
        for c in present:
            s, e = self.bounds[c]
            nd = self.order[s:e]
            self.cands[c] = nd[self.canon[nd] == nd]
            self.local[self.cands[c]] = np.arange(len(self.cands[c]))
        tokens = np.where(self.canon == np.arange(len(nodes)))[0]            # per haploblock, padded
        blk = self.region_code[tokens]
        srt = np.lexsort((tokens, blk))
        tokens, blk = tokens[srt], blk[srt]
        self.block_len = np.bincount(blk, minlength=len(self.regions))
        self.block_pos = np.full(len(nodes), -1, np.int64)                   # token -> index in its haploblock
        self.block_pos[tokens] = np.arange(len(tokens)) - np.repeat(np.cumsum(self.block_len) - self.block_len,
                                                                    self.block_len)
        self.block_pad = np.full((len(self.regions), max(1, self.block_len.max())), -1, np.int64)
        self.block_pad[blk, self.block_pos[tokens]] = tokens
        self.n_nodes = len(nodes)
        self.token_nodes = np.sort(tokens)                                   # rows of a per-token table
        self.tok_row = np.full(len(nodes), -1, np.int64)
        self.tok_row[self.token_nodes] = np.arange(len(self.token_nodes))

    def tokens(self, chrom, ex):
        """Token node indices of example `ex` (haplotype row or individual index) on chrom, genomic order."""
        if self.units == "haplotype":
            return self.canon[self.H[ex, self.block_cols[chrom]]]
        row = self.by_ind.indices[self.by_ind.indptr[ex]:self.by_ind.indptr[ex + 1]]
        s, e = self.bounds[chrom]
        lo, hi = np.searchsorted(row, [s, e])
        nodes = self.canon[self.order[row[lo:hi]]]
        return nodes[np.r_[True, nodes[1:] != nodes[:-1]]]   # two merged clusters of one haploblock = one token

    def examples(self, individuals):
        """(chrom, example) pairs with at least 2 tokens: both haplotype rows of each individual, or the
        individual itself."""
        ex = [2 * int(i) + h for i in individuals for h in (0, 1)] if self.units == "haplotype" \
            else [int(i) for i in individuals]
        return [(c, e) for c in self.chroms for e in ex if len(self.tokens(c, e)) >= 2]


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


def randomize(net, seed):
    """--init random: replace the pretrained weights by a seeded random init of the same architecture.
    Modules with reset_parameters() (linear, embedding, norm layers) are re-initialised by it; every
    other parameter (including e.g. nn.MultiheadAttention's in-projection, which has only a private reset) (custom norm scales, Hyena filter poles and residues, ...) has its values shuffled
    within the tensor, which keeps each tensor's distribution (scales stay near 1, filters stay stable)
    and removes what pretraining learned."""
    import torch
    g = torch.random.fork_rng(devices=[net_device(net)] if net_device(net).type == "cuda" else [])
    with g, torch.no_grad():
        torch.manual_seed(seed)
        done = set()
        for m in net.modules():
            if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
                m.reset_parameters()
                done |= {id(p) for p in m.parameters(recurse=False)}
        for p in net.parameters():
            if id(p) not in done:
                p.copy_(p.flatten()[torch.randperm(p.numel(), device=p.device)].view_as(p))


def net_device(net):
    return next(net.parameters()).device


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

    def __init__(self, evo2_model="evo2_7b_base", layer="blocks.28.mlp.l3", train_blocks=0, embed_module=None,
                 init="pretrained", seed=0):
        import torch
        if not torch.cuda.is_available():
            die("Evo 2 (the evo2 package) needs an NVIDIA GPU with CUDA. To check the pipeline on a CPU, "
                "use --backbone small.")
        try:
            from evo2 import Evo2
        except ImportError as e:
            die(f"could not import evo2 ({e}); install it with: python setup.py --evo2 (see DOCUMENTATION.md)")
        self.torch = torch
        self.evo = Evo2(evo2_model)
        net = self.net = self.evo.model
        self.dev = net_device(net)
        n_conv = convert_inference_tensors(net)
        if n_conv:
            log(f"  converted {n_conv:,} inference-mode tensors from the evo2 loader to trainable tensors")
        if init == "random":
            randomize(net, seed)
            log("  --init random: Evo 2's pretrained weights replaced by a seeded random init")

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
        """ids: int tensor [T] of Evo 2 (byte) tokens -> hidden states at --layer, [T, hidden_out] float32."""
        self._forward(ids.to(self.dev).int().unsqueeze(0))
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

    def __init__(self, ckpt_dir, model_size="7b", layer_index=28, train_blocks=0, fp8=False, pad_multiple=1,
                 init="pretrained", seed=0):
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            die("BioNeMo Evo 2 needs an NVIDIA GPU with CUDA; for a CPU check use --backbone small")
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
        self.dev = net_device(mcore)
        if init == "random":
            randomize(mcore, seed)
            log("  --init random: Evo 2's pretrained weights replaced by a seeded random init")

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


class SmallRunner:
    """A causal transformer (sinusoidal token positions) with randomly initialised weights, trained
    from scratch: the `--backbone small` baseline (at Evo 2's data and objective, it shows how much pretraining adds)
    and a CPU-sized model for checks. Its whole state is saved in the checkpoint."""

    backend = "small"

    def __init__(self, hidden=512, n_layers=6, train_blocks=None, device=None, seed=0):
        import torch
        import torch.nn as nn
        from haplo_common import torch_device
        self.torch = torch
        self.dev = torch_device(device)
        heads = max(1, hidden // 64) if hidden % 64 == 0 else (4 if hidden % 4 == 0 else 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)

            class Net(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.embedding_layer = nn.Embedding(256, hidden)            # byte tokens (run_ids)
                    s.blocks = nn.ModuleList(
                        nn.TransformerEncoderLayer(hidden, nhead=heads, dim_feedforward=4 * hidden, dropout=0.0,
                                                   batch_first=True, norm_first=True) for _ in range(n_layers))

                def forward(s, x):                    # x: [1, T, hidden]
                    T = x.shape[1]
                    i = torch.arange(T, device=x.device, dtype=torch.float32)[:, None]
                    f = torch.exp(torch.arange(0, hidden, 2, device=x.device) * (-np.log(1e4) / hidden))
                    pe = torch.zeros(T, hidden, device=x.device)
                    pe[:, 0::2], pe[:, 1::2] = torch.sin(i * f), torch.cos(i * f)[:, :hidden // 2]
                    x = x + pe                        # standard sinusoidal token-position encoding
                    mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
                    for blk in s.blocks:
                        x = blk(x, src_mask=mask)
                    return x

            self.net = Net().to(self.dev)
        for p in self.net.parameters():
            p.requires_grad_(False)
        n_train = n_layers if train_blocks is None else train_blocks
        self.trainable = unfreeze_last(self.net.blocks, n_layers - 1, n_train, "blocks")
        self.hidden_in = self.hidden_out = hidden
        self.config = {"hidden": hidden, "n_layers": n_layers, "seed": seed}
        log(f"  small causal transformer (trained from scratch): hidden {hidden}, {n_layers} layers, "
            f"{min(n_train, n_layers)} trained ({sum(p.numel() for _, p in self.trainable) / 1e6:.2f}M params)")

    def run_ids(self, ids):
        return self.net(self.net.embedding_layer(ids.long().to(self.dev).unsqueeze(0))).squeeze(0).float()

    def run_soft(self, soft):
        return self.net(soft.unsqueeze(0)).squeeze(0).float()

    def describe(self):
        return {"small_config": self.config}


def add_backbone_args(ap, device=True):
    """Command-line options shared by every script that loads a backbone (device=False: the script
    has its own --device)."""
    b = ap.add_argument_group("backbone")
    b.add_argument("--backbone", choices=BACKBONES, default="evo2",
                   help="evo2 = Evo 2 via the evo2 package (GPU); bionemo = Evo 2 via NVIDIA BioNeMo from a "
                        "NeMo2 checkpoint (--ckpt-dir); small = transformer trained from scratch (CPU or GPU)")
    b.add_argument("--bionemo", action="store_true", help="same as --backbone bionemo")
    b.add_argument("--init", choices=["pretrained", "random"], default="pretrained",
                   help="random = same Evo 2 architecture with its pretrained weights replaced by a seeded "
                        "random init (the control for 'pretraining helps')")
    b.add_argument("--evo2-model", default="evo2_7b_base", help="evo2: evo2_7b_base (8K ctx) or evo2_7b")
    b.add_argument("--layer", default="blocks.28.mlp.l3", help="evo2: read-out layer; forward stops here")
    b.add_argument("--embed-module", help="evo2: name of Evo 2's token embedding module (auto-detected)")
    b.add_argument("--ckpt-dir", help="bionemo: NeMo2 Evo 2 checkpoint folder (weights/ + context/)")
    b.add_argument("--model-size", default="7b", help="bionemo: 7b (8K ctx), 7b_arc_longcontext, ...")
    b.add_argument("--layer-index", type=int, default=28, help="bionemo: read-out decoder layer (negative = from end)")
    b.add_argument("--fp8", action="store_true", help="bionemo: FP8 Hyena projections (Hopper GPU)")
    b.add_argument("--pad-multiple", type=int, default=1, help="bionemo: pad sequences to a multiple of this")
    b.add_argument("--small-hidden", type=int, default=512, help="small: hidden size")
    b.add_argument("--small-layers", type=int, default=6, help="small: number of transformer blocks")
    if device:
        b.add_argument("--device", help="small: cpu / cuda (Evo 2 always uses the GPU)")


def resolve_backbone(args):
    if args.bionemo:
        if args.backbone not in ("evo2", "bionemo"):
            die("--bionemo contradicts --backbone " + args.backbone)
        args.backbone = "bionemo"
    if args.init == "random" and args.backbone == "small":
        die("--init random applies to Evo 2; --backbone small is already trained from scratch")


def make_runner(args, train_blocks):
    resolve_backbone(args)
    if args.backbone == "small":
        return SmallRunner(args.small_hidden, args.small_layers, train_blocks, args.device, args.seed)
    if args.backbone == "bionemo":
        if not args.ckpt_dir:
            die("--backbone bionemo needs --ckpt-dir (a NeMo2 Evo 2 checkpoint; see DOCUMENTATION.md)")
        return BioNeMoEvo2Runner(args.ckpt_dir, args.model_size, args.layer_index, train_blocks,
                                 args.fp8, args.pad_multiple, args.init, args.seed)
    return Evo2Runner(args.evo2_model, args.layer, train_blocks, args.embed_module, args.init, args.seed)


# ---------------------------------------------------------------------------
# New layers, objectives and loss
# ---------------------------------------------------------------------------

class HaploModel:
    """Input projection (embedding -> backbone hidden) + per-chromosome vector + head.

    objective "next": predict token t+1 from the backbone state at t (causal, like Evo 2's own
      pretraining). objective "masked": replace spans of 1..mask_max tokens (mask_frac of the tokens)
      by a learned mask vector plus the haploblock's position and length (the cluster's embedding is
      hidden) and predict them from both sides: the backbone runs on the sequence and on its
      reverse, and the head reads the two states of each masked position concatenated.
    head: "cosine" = cosine similarity to the candidate's embedding / temperature (fixed);
      "scaled-cosine" = the same with a learned logit scale (log scale initialised at log 10,
      clamped to [0, log 100]); "table" = a learned output vector and bias per token, initialised
      from the embeddings.
    softmax: "all" = over every token of the chromosome; "block" = over the tokens of the target's
      haploblock only (exact for haplotype units, where every position is one haploblock), which
      leaves the question "which cluster" without "where"."""

    def __init__(self, rnr, data, head="scaled-cosine", softmax="block", objective="next", temperature=0.1,
                 max_len=1024, mask_frac=0.15, mask_max=3):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        if softmax == "block" and data.units != "haplotype" and objective == "next":
            die("--softmax block needs --units haplotype (with individual units the next token may be in "
                "the same haploblock)")
        if objective == "masked" and data.units != "haplotype":
            die("--objective masked needs --units haplotype")
        self.torch, self.F = torch, F
        self.rnr, self.data = rnr, data
        self.head_kind, self.softmax, self.objective = head, softmax, objective
        self.temperature, self.max_len, self.mask_frac, self.mask_max = temperature, max_len, mask_frac, mask_max
        dev, dim = rnr.dev, data.Z.shape[1]
        self.proj_in = nn.Sequential(nn.Linear(dim, rnr.hidden_in), nn.LayerNorm(rnr.hidden_in)).to(dev)
        self.chrom_emb = nn.Embedding(data.n_chrom_slots, rnr.hidden_in).to(dev)
        nn.init.normal_(self.chrom_emb.weight, std=0.02)
        width = rnr.hidden_out * (2 if objective == "masked" else 1)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim)).to(dev)
        self.extra = nn.ParameterDict()
        if objective == "masked":
            self.extra["mask"] = nn.Parameter(torch.randn(rnr.hidden_in, device=dev) * 0.02)
        if head == "scaled-cosine":
            self.extra["log_scale"] = nn.Parameter(torch.tensor(float(np.log(10.0)), device=dev))
        self.Z_t = torch.from_numpy(data.Z).to(dev)
        self.keys = F.normalize(self.Z_t, dim=-1)
        self.tok_row = torch.from_numpy(data.tok_row).to(dev)
        if head == "table":
            self.extra["table"] = nn.Parameter(self.keys[torch.from_numpy(data.token_nodes).to(dev)].clone())
            self.extra["bias"] = nn.Parameter(torch.zeros(len(data.token_nodes), device=dev))
        self.modules = [self.proj_in, self.chrom_emb, self.head, self.extra]
        self.pad_t = torch.from_numpy(data.block_pad).to(dev)

    def window(self, n_tok, rng=None):
        """Training: random window of max_len tokens; evaluation (rng=None): the central one."""
        if n_tok <= self.max_len:
            return 0, n_tok
        s = int(rng.integers(0, n_tok - self.max_len + 1)) if rng is not None else (n_tok - self.max_len) // 2
        return s, s + self.max_len

    def sample_mask(self, T, rng):
        """Positions of masked spans (1..mask_max tokens each) covering ~mask_frac of T tokens."""
        want, out = max(1, int(round(self.mask_frac * T))), set()
        while len(out) < want:
            n = int(rng.integers(1, self.mask_max + 1))
            s = int(rng.integers(0, max(1, T - n + 1)))
            out.update(range(s, min(T, s + n)))
        return np.array(sorted(out))

    def scale(self):
        if self.head_kind == "cosine":
            return 1 / self.temperature
        return self.extra["log_scale"].clamp(0, float(np.log(100.0))).exp()

    def logits(self, pred, targets, chrom, chunk=256):
        """pred [P, dim] -> (logits [P, K] over the candidate tokens, target index [P], candidate nodes [P, K]
        (-1 = padding))."""
        torch = self.torch
        table = self.head_kind == "table"
        q = pred if table else self.F.normalize(pred, dim=-1) * self.scale()
        if self.softmax == "all":
            nodes = torch.from_numpy(self.data.cands[chrom]).to(self.rnr.dev)
            if table:
                rows = self.tok_row[nodes]
                lg = q @ self.extra["table"][rows].T + self.extra["bias"][rows]
            else:
                lg = q @ self.keys[nodes].T
            cand = nodes.expand(len(targets), -1)
        else:
            cand = self.pad_t[torch.from_numpy(self.data.region_code[targets]).to(self.rnr.dev)]
            safe, parts = cand.clamp(min=0), []
            for c in range(0, len(targets), chunk):          # [chunk, K, dim] at a time
                sl = slice(c, c + chunk)
                if table:
                    rows = self.tok_row[safe[sl]]
                    parts.append(torch.einsum("pd,pkd->pk", q[sl], self.extra["table"][rows]) + self.extra["bias"][rows])
                else:
                    parts.append(torch.einsum("pd,pkd->pk", q[sl], self.keys[safe[sl]]))
            lg = torch.cat(parts).masked_fill(cand < 0, float("-inf"))
        local = self.data.local if self.softmax == "all" else self.data.block_pos
        return lg, torch.from_numpy(local[targets]).to(self.rnr.dev), cand

    def forward(self, chrom, ex, rng=None, mask=None):
        """-> (logits [P, K], target index [P], candidates [P, K], target nodes [P], positions [P]).
        Positions are token offsets in the example; mask (masked objective) = token offsets to mask
        (default: sampled with rng, or fixed by evaluation_mask)."""
        torch = self.torch
        nodes = self.data.tokens(chrom, ex)
        s, e = self.window(len(nodes), rng)
        tok = torch.from_numpy(nodes[s:e]).to(self.rnr.dev)
        x = self.proj_in(self.Z_t[tok]) + self.chrom_emb.weight[self.data.cidx[chrom]]
        if self.objective == "next":
            h = self.rnr.run_soft(x)[:-1]
            pos = np.arange(s + 1, e)
        else:
            pos = np.asarray(mask if mask is not None else
                             (self.sample_mask(e - s, rng) + s if rng is not None else self.evaluation_mask(chrom, ex)))
            pos = pos[(pos >= s) & (pos < e)]
            local = torch.from_numpy(pos - s).to(self.rnr.dev)
            where = self.Z_t[tok[local]].clone()
            where[:, :self.data.n_emb] = 0                    # hide the cluster, keep which haploblock it is
            x = x.index_put((local,), self.proj_in(where) + self.extra["mask"]
                            + self.chrom_emb.weight[self.data.cidx[chrom]])
            h = torch.cat([self.rnr.run_soft(x), self.rnr.run_soft(x.flip(0)).flip(0)], -1)[local]
        lg, target, cand = self.logits(self.head(h), nodes[pos], chrom)
        return lg, target, cand, nodes[pos], pos

    def evaluation_mask(self, chrom, ex):
        """Fixed masked positions for evaluating a masked-objective model on one example."""
        T = len(self.data.tokens(chrom, ex))
        return self.sample_mask(T, np.random.default_rng([ex, self.data.cidx[chrom]]))

    def loss(self, chrom, ex, rng=None, mask=None):
        """-> (mean cross-entropy, (n correct top-1, n predictions))."""
        lg, target, *_ = self.forward(chrom, ex, rng, mask)
        return self.F.cross_entropy(lg, target), ((lg.argmax(-1) == target).sum().item(), len(target))

    def score(self, chrom, ex, mask=None, probs=False):
        """No gradient. -> dict of per-prediction arrays: pos, target (node), pred (node), logp (of the
        target), top1, top5, entropy; with probs, also cand [P, K] (candidate nodes, -1 = padding) and
        p [P, K] (their probabilities)."""
        torch = self.torch
        with torch.no_grad():
            lg, target, cand, tnodes, pos = self.forward(chrom, ex, mask=mask)
            lp = torch.log_softmax(lg, -1)
            k = min(5, lp.shape[1])
            top = lp.topk(k, -1).indices
            p = lp.exp()
            ent = -(p * lp.masked_fill(p == 0, 0)).sum(-1)
            out = {"pos": pos, "target": tnodes, "pred": cand.gather(1, top[:, :1]).squeeze(1).cpu().numpy(),
                   "logp": lp.gather(1, target[:, None]).squeeze(1).cpu().numpy(),
                   "top1": (top[:, 0] == target).cpu().numpy(), "top5": (top == target[:, None]).any(1).cpu().numpy(),
                   "entropy": ent.cpu().numpy()}
            if probs:
                out.update(cand=cand.cpu().numpy(), p=p.cpu().numpy())
        return out

    def parameters(self):
        return [p for m in self.modules for p in m.parameters()]

    def train(self, mode=True):
        self.rnr.net.train(mode)
        for m in self.modules:
            m.train(mode)

    def state(self):
        return {"proj_in": self.proj_in.state_dict(), "chrom_emb": self.chrom_emb.state_dict(),
                "head_layers": self.head.state_dict(), "extra": self.extra.state_dict()}

    def load(self, meta):
        for name, m in (("proj_in", self.proj_in), ("chrom_emb", self.chrom_emb), ("head_layers", self.head),
                        ("extra", self.extra)):
            m.load_state_dict(meta[name])


class PhenotypeModel:
    """A phenotype from an individual's genome, for 4b. input "tokens": both haplotypes of every
    chromosome as step-4a tokens (one per haploblock), each run through the backbone; the states of
    the two haplotypes of a chromosome are attention-pooled into one vector, the chromosome vectors
    attention-pooled into one genome vector, and a head predicts the phenotype. input "hash": a 0/1
    genomic hash cut into tokens of hash_bits bits (+-1 vectors, one per haploblock hash), so linkage
    between blocks is not given; same pooling over the tokens and head."""

    def __init__(self, rnr, n_out, data=None, hash_bits=20, max_len=1024):
        import torch
        import torch.nn as nn
        self.torch, self.rnr, self.data, self.hash_bits, self.max_len = torch, rnr, data, hash_bits, max_len
        dev, H = rnr.dev, rnr.hidden_in
        self.proj_in = (nn.Sequential(nn.Linear(data.Z.shape[1], H), nn.LayerNorm(H)) if data is not None
                        else nn.Sequential(nn.Linear(hash_bits, H), nn.LayerNorm(H))).to(dev)
        self.chrom_emb = nn.Embedding(data.n_chrom_slots if data is not None else 1, H).to(dev)
        nn.init.normal_(self.chrom_emb.weight, std=0.02)
        self.query = nn.Parameter(torch.zeros(2, rnr.hidden_out, device=dev))     # chromosome, genome pooling
        self.head = nn.Sequential(nn.LayerNorm(rnr.hidden_out), nn.Linear(rnr.hidden_out, n_out)).to(dev)
        self.modules = [self.proj_in, self.chrom_emb, self.head]
        self.Z_t = torch.from_numpy(data.Z).to(dev) if data is not None else None

    def pool(self, h, which):
        w = self.torch.softmax(h @ self.query[which] / h.shape[-1] ** 0.5, 0)
        return w @ h, w

    def encode(self, example):
        """example: individual index (tokens) or hash string -> (genome vector, {chrom: (token nodes, weights)})."""
        torch, rnr = self.torch, self.rnr
        windows = lambda n: range(0, n, self.max_len)          # every token, --max-len per backbone pass
        if self.data is None:
            bits = np.frombuffer(example.encode(), np.uint8) - 48
            bits = np.pad(bits, (0, (-len(bits)) % self.hash_bits)).reshape(-1, self.hash_bits)
            x = self.proj_in(torch.from_numpy(bits * 2.0 - 1).float().to(rnr.dev)) + self.chrom_emb.weight[0]
            return self.pool(torch.cat([rnr.run_soft(x[s:s + self.max_len]) for s in windows(len(x))]), 0)[0], {}
        vecs, attn = [], {}
        for chrom in self.data.chroms:
            hs, toks = [], []
            for row in (2 * example, 2 * example + 1):
                t = self.data.tokens(chrom, row)
                for s in windows(len(t)):
                    tt = t[s:s + self.max_len]
                    x = self.proj_in(self.Z_t[torch.from_numpy(tt).to(rnr.dev)]) + self.chrom_emb.weight[self.data.cidx[chrom]]
                    hs.append(rnr.run_soft(x)); toks.append(tt)
            v, w = self.pool(torch.cat(hs), 0)
            vecs.append(v)
            attn[chrom] = (np.concatenate(toks), w.detach().cpu().numpy())
        return self.pool(torch.stack(vecs), 1)[0], attn

    def predict(self, example):
        return self.head(self.encode(example)[0])

    def parameters(self):
        return [p for m in self.modules for p in m.parameters()] + [self.query]

    def train(self, mode=True):
        self.rnr.net.train(mode)
        for m in self.modules:
            m.train(mode)

    def state(self):
        return {"proj_in": self.proj_in.state_dict(), "chrom_emb": self.chrom_emb.state_dict(),
                "head_layers": self.head.state_dict(), "query": self.query.detach().cpu()}

    def load(self, meta):
        for name, m in (("proj_in", self.proj_in), ("chrom_emb", self.chrom_emb), ("head_layers", self.head)):
            m.load_state_dict(meta[name])
        with self.torch.no_grad():
            self.query.copy_(meta["query"].to(self.query.device))


def backbone_state(rnr):
    """What a checkpoint stores of the backbone: everything for small, the unfrozen blocks otherwise."""
    return rnr.net.state_dict() if rnr.backend == "small" else dict(rnr.trainable)


def backbone_fingerprint(net):
    """Short id of a backbone's weights (the first 64 values of every tensor), stored before training:
    step 6 checks that it rebuilds the same base model (and, with --init random, the same random init)."""
    h = hashlib.sha1()
    for name, t in sorted(list(net.named_parameters()) + list(net.named_buffers()), key=lambda x: x[0]):
        if name.endswith("_extra_state"):                      # FP8 runtime metadata, not weights
            continue
        h.update(name.encode())
        h.update(t.detach().reshape(-1)[:64].float().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def rebuild(meta, graph_dir, units, device=None, ckpt_dir=None):
    """GenomeData and backbone (with the checkpoint's trained parameters) of a 4a / 4b checkpoint,
    after checking that the graph build, the embeddings and the base model are the ones it was trained on."""
    import argparse
    import torch
    from haplo_common import graph_id
    cfg = meta["config"]
    pos = {x: i for i, x in enumerate(read_graph_individuals(graph_dir))}
    data = GenomeData(graph_dir, meta["graph_method"], meta["chroms"], cfg["min_support"],
                      [pos[x] for x in meta["train_individuals"] if x in pos], cfg["pos_freqs"], units)
    if meta["graph_id"] != graph_id(graph_dir):
        raise ValueError(f"the model was trained on another build of this graph (graph_id {meta['graph_id']}, "
                         f"now {graph_id(graph_dir)}); rerun steps 3-4 on the current graph")
    if meta["embeddings_sha1"] != data.emb_sha1:
        raise ValueError(f"node_emb_{meta['graph_method']}.npy changed since the model was trained (different "
                         "checksum); rerun step 4 on the current embeddings")
    if data.n_chrom_slots != meta["n_chrom_slots"]:
        raise ValueError("this graph has a different set of chromosomes than the one the model was trained on")
    a = argparse.Namespace(**{**cfg, "backbone": meta["backend"], "bionemo": False, "device": device,
                              "ckpt_dir": ckpt_dir or cfg.get("ckpt_dir")})
    rnr = make_runner(a, train_blocks=0)
    if meta["backend"] != "small" and backbone_fingerprint(rnr.net) != meta["backbone_fingerprint"]:
        raise ValueError("the backbone rebuilt here is not the one the model was trained from (other base model, "
                         "Evo 2 version or random init)")
    tb = {k: v.to(rnr.dev) for k, v in meta["trained_blocks"].items()}
    missing = set(tb) - set(rnr.net.state_dict())
    if missing:
        die(f"{len(missing)} fine-tuned parameters do not exist in the backbone (e.g. {sorted(missing)[:2]})")
    with torch.no_grad():
        rnr.net.load_state_dict(tb, strict=False)
    return data, rnr


def load_finetuned(meta, graph_dir, device=None, ckpt_dir=None):
    """Rebuild a step-4a model (checkpoint dict `meta`) on the graph in graph_dir.
    Returns (HaploModel in eval mode, GenomeData). Evo 2 / BioNeMo backbones need the GPU and the
    same base model as in training (with --init random, the same seed); the small backbone is
    stored whole in the checkpoint."""
    if meta.get("mode") != "embeddings":
        die("only models from 4a_finetune_evo2_embeddings.py can be queried here")
    cfg = meta["config"]
    data, rnr = rebuild(meta, graph_dir, meta["units"], device, ckpt_dir)
    model = HaploModel(rnr, data, meta["head"], meta["softmax"], meta["objective"], cfg["temperature"],
                       cfg["max_len"], cfg["mask_frac"], cfg["mask_max"])
    model.load(meta)
    model.train(False)
    return model, data


def load_phenotype_model(meta, graph_dir, device=None, ckpt_dir=None):
    """Rebuild a 4b checkpoint (token input) on graph_dir -> (PhenotypeModel in eval mode, GenomeData)."""
    data, rnr = rebuild(meta, graph_dir, "haplotype", device, ckpt_dir)
    model = PhenotypeModel(rnr, meta["n_out"], data, max_len=meta["config"]["max_len"])
    model.load(meta)
    model.train(False)
    return model, data


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


RESUMABLE = ("epochs", "resume", "log_every", "eval_every", "out", "device", "test", "smoke")


def fit(args, blocks, new_params, n_examples, example_loss, evaluate, describe, set_train, grad_check,
        resume_path=None, rng=None):
    """AdamW on the trainable backbone parameters `blocks` (--lr, through fp32 master weights) and the
    new layers' parameters `new_params` (--head-lr); gradient accumulation over --batch examples,
    gradient clipping at 1; --epochs passes in an order fixed by --seed and the epoch.
      example_loss(j) -> loss tensor of training example j (it may draw from `rng`);  evaluate() ->
      metrics dict with "loss" (run without gradients in eval mode, every --eval-every steps and at
      the end of every epoch);  set_train(mode) switches every module;  grad_check = (description,
      parameters that must receive a gradient on the first step).
    The parameters of the best evaluation (lowest validation loss; the untrained parameters count too)
    are restored at the end. With resume_path, the training state (fp32 master weights, optimizer,
    best parameters, the state of `rng`, the settings) is saved there after every evaluation, and
    --resume continues from it; settings other than --epochs and logging must not change.
    Returns {"before", "best", "last", "best_step", "history"}."""
    import torch
    master = MasterWeights(blocks)
    if master.pairs:
        log(f"  {len(master.pairs)} low-precision trainable tensors: optimizing fp32 master copies")
    trained = master.params + list(new_params)               # what the optimizer updates (fp32)
    opt = torch.optim.AdamW([{"params": master.params, "lr": args.lr}, {"params": list(new_params), "lr": args.head_lr}],
                            weight_decay=0.01)
    snapshot = lambda: [p.detach().cpu().clone() for p in trained]
    settings = hashlib.sha1(json_dumps({k: v for k, v in config_of(args).items() if k not in RESUMABLE})).hexdigest()

    def restore(state):
        with torch.no_grad():
            for p, v in zip(trained, state):
                p.copy_(v.to(p.device))
        master.master_to_model()

    def run_eval():
        set_train(False)
        with torch.no_grad():
            res = evaluate()
        set_train(True)
        return res

    batch = min(args.batch, n_examples)
    if batch < 1:
        die("no training examples")
    st = {"step": 0, "epoch": 0, "done": 0, "history": [], "best": None, "best_step": 0}
    if getattr(args, "resume", False) and resume_path and os.path.exists(resume_path):
        r = torch.load(resume_path, map_location="cpu", weights_only=True)
        if r["settings"] != settings:
            die(f"{resume_path} was written with other settings; rerun without --resume (or with the same settings)")
        restore(r["params"])
        best_params = r["best_params"]
        opt.load_state_dict(r["optimizer"])
        st, before = {k: r[k] for k in st}, r["before"]
        if rng is not None:
            rng.bit_generator.state = r["rng"]
        log(f"  resumed from {resume_path}: epoch {st['epoch'] + 1}, step {st['step']}")
    else:
        before = run_eval()
        st["best"], best_params = before, snapshot()
        log(f"  before training: {describe(before)}")
    set_train(True)
    t0, steps_here = time.time(), 0

    def evaluation():
        nonlocal best_params
        res = run_eval()
        st["history"].append({"step": st["step"], **res})
        if res["loss"] < st["best"]["loss"]:
            st["best"], st["best_step"], best_params = res, st["step"], snapshot()
        log(f"  [eval step {st['step']}] {describe(res)}" + ("  (best)" if st["best_step"] == st["step"] else ""))
        if resume_path:
            torch.save({"params": snapshot(), "best_params": best_params, "optimizer": opt.state_dict(),
                        "before": before, "settings": settings,
                        "rng": rng.bit_generator.state if rng is not None else None, **st}, resume_path)

    for epoch in range(st["epoch"], args.epochs):
        order = np.random.default_rng([args.seed, epoch]).permutation(n_examples)
        for b in range(st["done"] if epoch == st["epoch"] else 0, n_examples, batch):
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
                die(f"loss became {batch_loss} at step {st['step']}; lower --lr / --head-lr")
            if st["step"] == 0:
                what, check = grad_check
                if all(p.grad is None for p in check):
                    die(f"no gradient reached {what}: this backbone's forward pass is not differentiable here")
                if blocks and all(p.grad is None for p in blocks):
                    die("no gradient reached the trainable backbone blocks; try --train-blocks 0 "
                        "(trains only the new layers through the frozen backbone)")
            master.grads_to_master()
            torch.nn.utils.clip_grad_norm_(trained, 1.0)
            opt.step()
            master.master_to_model()
            st["step"] += 1
            st["done"] = b + len(chunk)
            steps_here += 1
            if st["step"] % args.log_every == 0:
                log(f"  epoch {epoch + 1} step {st['step']}: loss {batch_loss:.4f}  "
                    f"{(time.time() - t0) / steps_here:.2f}s/step")
            if st["step"] % args.eval_every == 0:
                evaluation()
        st["epoch"], st["done"] = epoch + 1, 0
        if not st["history"] or st["history"][-1]["step"] != st["step"]:
            evaluation()
    restore(best_params)
    log(f"  best: {describe(st['best'])} at step {st['best_step']} of {st['step']}")
    log(f"  (before training: {describe(before)})")
    return {"before": before, "best": st["best"], "last": st["history"][-1] if st["history"] else None,
            "best_step": st["best_step"], "history": st["history"]}


def json_dumps(d):
    import json
    return json.dumps(d, sort_keys=True, default=str).encode()


def config_of(args):
    return {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, list, bool, type(None)))}

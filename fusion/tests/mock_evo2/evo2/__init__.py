"""A tiny stand-in for the `evo2` package, ONLY for testing the plumbing of step 4a and the per-patient export
on a CPU (no GPU, no weights download). Put this folder first on PYTHONPATH:

    PYTHONPATH=fusion/tests/mock_evo2 python 4a_finetune_evo2_embeddings.py ... --epochs 1

It mimics what step 4a uses: Evo2(name).model with a token embedding module named `embedding_layer`, a
`blocks` list and a module `blocks.28.mlp.l3`. Its numbers mean nothing; never use its outputs as results.
"""

import torch
from torch import nn

IS_MOCK = True      # checked by fusion/run_genome_folds.sh and genome_patient_embeddings.py
HIDDEN = 16
N_BLOCKS = 29


class _MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l1 = nn.Linear(d, 2 * d)
        self.l3 = nn.Linear(2 * d, d)

    def forward(self, x):
        return self.l3(torch.relu(self.l1(x)))


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = _MLP(d)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(512, HIDDEN)
        self.blocks = nn.ModuleList([_Block(HIDDEN) for _ in range(N_BLOCKS)])

    def forward(self, ids):
        x = self.embedding_layer(ids)
        for b in self.blocks:
            x = b(x)
        return x


class Evo2:
    def __init__(self, name="mock"):
        torch.manual_seed(0)
        self.name = name
        self.model = _Net()

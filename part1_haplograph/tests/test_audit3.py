"""
Regression tests from the third audit (iteration 3: --pos-freqs genomic-position encoding).
Run:  python -m pytest part1_haplograph/tests -q
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _tiny_graph(path, starts):
    """Minimal graph folder GenomeData can read: one chromosome, one cluster per region."""
    n = len(starts)
    pd.DataFrame({"node": range(n), "chrom": "chr21", "region": [f"r{i}" for i in range(n)],
                  "start": starts}).to_csv(path / "nodes.csv.gz", index=False)
    np.save(path / "node_emb_gnn.npy", np.random.default_rng(0).normal(size=(n, 4)).astype(np.float32))
    sp.save_npz(path / "carriers.npz", sp.csr_matrix(np.ones((n, 2), np.int8)))
    (path / "individuals.txt").write_text("A\nB\n")
    return str(path)


def test_pos_freqs_encoding_unique_on_chr21_like_graph(tmp_path):
    """Guard (passes): steps 4 and 5 rely on the default encoding telling regions apart, including
    regions whose starts differ by a multiple of the shortest (10 kb) wavelength."""
    from evo2_model import GenomeData
    starts = [14_000_000, 14_010_000, 14_020_000, 30_000_000, 46_000_000]
    d = GenomeData(_tiny_graph(tmp_path, starts), "gnn", None, 0, [], 16)
    assert len(np.unique(d.Z[:, -32:].round(4), axis=0)) == len(starts)


def test_pos_freqs_1_encodes_coarse_position(tmp_path):
    """DOCUMENTATION step 4: wavelengths 'from 10 kb to the longest chromosome span'. A single
    wavelength must still say where on the chromosome a cluster is (it is the coarse one that does)."""
    from evo2_model import GenomeData
    starts = [10_000_000, 20_000_000, 30_000_000, 40_000_000]
    d = GenomeData(_tiny_graph(tmp_path, starts), "gnn", None, 0, [], 1)
    assert len(np.unique(d.Z[:, -2:].round(4), axis=0)) == len(starts)

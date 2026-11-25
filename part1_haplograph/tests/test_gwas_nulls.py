"""
gwas_enrichment in step 6: calibration of the nulls and edge cases.
Run:  python -m pytest part1_haplograph/tests -q
"""
import re
import types

import numpy as np
import pandas as pd
import pytest

from conftest import script

V = script("6_validate_evo2")


class Ctx:
    def __init__(self, regions, out_dir, seed=0):
        self.graph = types.SimpleNamespace(regions=regions)
        self.rng = np.random.default_rng(seed)
        self.out_dir = str(out_dir)

    def out(self, name):
        return f"{self.out_dir}/{name}"


def tiled(lengths, chrom="chr21", start=1_000_000):
    st = start + np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return pd.DataFrame({"region": [f"{chrom}_{s}-{s + n}" for s, n in zip(st, lengths)],
                         "chrom": chrom, "start": st, "end": st + np.asarray(lengths)})


def run(tmp_path, regions, df, seed=0, **kw):
    path = tmp_path / "g.tsv"
    df.to_csv(path, sep="\t", index=False)
    return V.gwas_enrichment(Ctx(regions, tmp_path, seed), gwas=str(path), **kw)


def frame(rows):
    return pd.DataFrame(rows, columns=["CHR", "POS", "SNP", "P"])


def test_empirical_fdr_reflects_under_clumped_cluster(tmp_path):
    """One LD signal (30 hits within 1.5 kb) counted hit by hit (clump=0) gives 1 enriched haploblock
    that is false by construction. The rotation null reproduces it (~1.0), but empirical_fdr is 0.0
    and the note says 'empirical FDR 0%' (real chr21 height, clump=0: 8 calls, FDR 0.001, rotation
    7.9). In LD-like simulations the reported FDR was 0-3% where the true FDP was 100%. Some FDR
    field must account for the rotation null."""
    reg = tiled([40_000] * 50)
    rows = [("chr21", 1_400_010 + 50 * i, f"a{i}", 1e-9) for i in range(30)]
    rows += [("chr21", int(x), f"b{i}", 1e-9) for i, x in enumerate(np.linspace(1_010_000, 2_990_000, 20))]
    s = run(tmp_path, reg, frame(rows), clump=0, permutations=200)["summary"]
    assert s["regions_enriched_q<0.05"] >= 1
    assert s["regions_q<0.05_under_rotation"] >= 0.8             # the diagnostic itself works
    fdrs = [v for k, v in s.items() if "fdr" in k.lower() and isinstance(v, (int, float))]
    assert max(fdrs) >= 0.5, s


def test_summary_mode_rotation_tracks_false_calls(tmp_path):
    """Summary statistics: one 60 kb LD signal (every SNP in it a hit) at a random place, clump 5 kb,
    so its several 'loci' are one signal and every call it causes is false. Over random placements it
    gives ~2.3 calls; hits mode on the same hits gives rotation ~2.0, but summary mode gives ~0.1
    (by_chance ~0.06): moved hits are added on top of the destination's tested SNPs (n_other fixed),
    so the expected count there roughly doubles. Null draws must keep per-haploblock SNP counts
    (e.g. rotate the hit labels / p-values over the SNP order instead of moving positions)."""
    R, W = 60, 20_000
    reg = tiled([W] * R)
    pos = np.arange(1_000_050, 1_000_000 + R * W, 100)
    rng = np.random.default_rng(0)
    p0 = rng.uniform(0.01, 1, len(pos))
    p0[rng.choice(len(pos), 15, replace=False)] = 1e-9
    calls = []
    for c in np.linspace(1_100_000, 2_100_000, 9).astype(int):
        p = p0.copy()
        p[(pos >= c) & (pos < c + 60_000)] = 1e-12
        df = pd.DataFrame({"CHR": "chr21", "POS": pos, "SNP": [f"s{i}" for i in range(len(pos))], "P": p})
        calls.append(run(tmp_path, reg, df, mode="summary", clump=5000, permutations=0)
                     ["summary"]["regions_enriched_q<0.05"])
    s = run(tmp_path, reg, df, mode="summary", clump=5000, permutations=200)["summary"]
    assert np.mean(calls) >= 1.5, calls
    assert s["regions_q<0.05_under_rotation"] >= 0.5 * np.mean(calls), (s["regions_q<0.05_under_rotation"], calls)


def test_window_coverage_at_most_100_percent(tmp_path):
    """window=50000 on chr21 printed 'haploblocks cover 286% of each chromosome span'. Coverage of a
    span cannot exceed 100%; compute it from the union of the (extended) haploblocks."""
    reg = tiled([20_000] * 20)
    rows = [("chr21", 1_000_000 + 5_000 + 20_000 * i, f"a{i}", 1e-9) for i in range(20)]
    s = run(tmp_path, reg, frame(rows), window=50_000, permutations=0)["summary"]
    assert float(s["haploblock_coverage_of_span"].rstrip("%")) <= 100.0, s["haploblock_coverage_of_span"]


def test_no_top_region_without_loci_in_haploblocks(tmp_path):
    """The only hit lies in the gap between two haploblocks (a non-hit SNP is inside one, so the
    task runs): no haploblock has a locus, yet top_region = the first haploblock (q = 1)."""
    reg = pd.DataFrame({"region": ["a", "b"], "chrom": "chr21", "start": [1000, 5000], "end": [2000, 6000]})
    s = run(tmp_path, reg, frame([("chr21", 3000, "g", 1e-9), ("chr21", 1500, "h", 0.5)]),
            permutations=0)["summary"]
    assert s["regions_with_loci"] == 0
    assert s["top_region"] is None, s["top_region"]


def test_summary_auto_clump_warns_when_no_power(tmp_path):
    """On chr21 (longest haploblock 341 kb, median 41 kb) a 250 kb clump allows at most 2 loci per
    haploblock, so 0% of haploblocks can reach p < 0.05/597 (25 kb hits: 7%, 10 kb: 67%). The
    per-haploblock test is then structurally unable to call anything; the output should say so."""
    reg = tiled([40_000] * 100)
    pos = np.arange(1_000_050, 5_000_000, 200)
    p = np.random.default_rng(0).uniform(0.01, 1, len(pos))
    for c in range(1_200_000, 4_800_000, 400_000):
        p[(pos >= c) & (pos < c + 20_000)] = 1e-10
    df = pd.DataFrame({"CHR": "chr21", "POS": pos, "SNP": [f"s{i}" for i in range(len(pos))], "P": p})
    r = run(tmp_path, reg, df, permutations=0)
    assert r["summary"]["clump_bp"] == 250_000
    assert any(re.search(r"power|cannot reach|can't reach|too large", n) for n in r["notes"]), r["notes"]

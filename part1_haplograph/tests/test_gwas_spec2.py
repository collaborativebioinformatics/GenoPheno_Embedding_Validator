"""
Spec-test regressions (round 2) for gwas_enrichment in step 6: calibration and edge cases.
Each test guards a confirmed issue and passes once it is fixed. Synthetic regions, no network.
Run:  python -m pytest part1_haplograph/tests -q
"""
import json
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
    """Haploblocks tiling a chromosome; adjacent blocks share the end/start coordinate like the
    real haplograph (chr21_14215892-14284114 is followed by chr21_14284114-...)."""
    st = start + np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return pd.DataFrame({"region": [f"{chrom}_{s}-{s + n}" for s, n in zip(st, lengths)],
                         "chrom": chrom, "start": st, "end": st + np.asarray(lengths)})


def run(tmp_path, regions, rows, **kw):
    path = tmp_path / "g.tsv"
    pd.DataFrame(rows, columns=["CHR", "POS", "SNP", "P"]).to_csv(path, sep="\t", index=False)
    return V.gwas_enrichment(Ctx(regions, tmp_path), gwas=str(path), **kw)


def test_permutations_zero_reports_no_null_based_numbers(tmp_path):
    """With permutations=0 there is no null, yet the summary reported empirical_fdr = 1.0 (min(1, nan))
    and p = 1.0 for the same-haploblock and inside tests, and wrote NaN (invalid JSON) for the
    chance level. Null-based numbers must be None when no rotations were run."""
    reg = tiled([50_000] * 40)
    rows = [("chr21", 1_100_000 + i * 100, f"rs{i}", 1e-9) for i in range(10)] + \
           [("chr21", 1_000_000 + 47_000 * i, f"bg{i}", 1e-9) for i in range(1, 40)]
    s = run(tmp_path, reg, rows, clump=0, permutations=0)["summary"]
    assert s["regions_enriched_q<0.05"] >= 1
    for k in ("empirical_fdr", "regions_q<0.05_by_chance", "hit_pairs_same_haploblock_p", "hits_in_haploblocks_p"):
        assert s[k] is None, (k, s[k])
    json.dumps(s, allow_nan=False)


@pytest.mark.parametrize("param", [{"permutations": -5}, {"clump": -1}, {"window": -100}])
def test_negative_parameters_are_rejected_with_a_message(tmp_path, param):
    """Negative permutations crashed with numpy's 'negative dimensions are not allowed'; negative
    clump or window were silently accepted (clump=-1 keeps every hit, window<0 shrinks blocks).
    They must be refused with a message naming the parameter."""
    reg = tiled([50_000] * 10)
    rows = [("chr21", 1_010_000 + i * 30_000, f"rs{i}", 1e-9) for i in range(10)]
    name = next(iter(param))
    with pytest.raises(ValueError, match=name):
        run(tmp_path, reg, rows, **{"permutations": 5, **param})


def test_hit_on_a_shared_boundary_counts_once_without_window(tmp_path):
    """Adjacent haploblocks share their boundary coordinate and region_pairs uses closed intervals,
    so with window=0 a hit exactly on a boundary was counted in both haploblocks (1 locus -> 2
    region counts, and an extra same-haploblock pair). Only window > 0 may count a SNP twice."""
    reg = tiled([50_000] * 10)
    boundary = int(reg["end"].iloc[3])
    rows = [("chr21", boundary, "rsB", 1e-9), ("chr21", 1_010_000, "rs1", 1e-9)]
    out = run(tmp_path, reg, rows, clump=0, permutations=5)
    table = pd.read_csv(out["files"][0])
    assert table["n_loci"].sum() == out["summary"]["loci"] == 2


def test_snps_only_in_haploblock_gaps_fail_as_documented(tmp_path):
    """DOCUMENTATION.md: 'a file with no SNP inside the haploblocks fails'. SNPs inside the span but
    all in gaps between haploblocks were accepted, and a haploblock with 0 loci was reported as
    top_region."""
    reg = pd.DataFrame({"region": ["a", "b"], "chrom": "chr21", "start": [1000, 5000], "end": [2000, 6000]})
    rows = [("chr21", 3000 + 100 * i, f"rs{i}", 1e-9) for i in range(5)]
    try:
        s = run(tmp_path, reg, rows, permutations=5)["summary"]
    except ValueError:
        return
    assert s["top_region"] is None


def test_independent_loci_in_one_haploblock_are_not_called_chance(tmp_path):
    """The rotation null moves all hits together, so it also carries the true enrichment along: 10
    loci 12 kb apart (clump=10 kb, so each is its own locus) planted in one 200 kb haploblock are
    detected, but rotations land the whole set inside another 200 kb haploblock most of the time,
    giving an empirical FDR of ~1.0 and the note 'raise clump' (on chr21 blocks: 0.72-0.77). The
    chance level should keep LD clustering (hits within `clump`) but not the co-location of
    independent loci, e.g. place each clumped locus (with its satellite hits) independently at
    random in the null."""
    reg = tiled([200_000] * 60)
    rng = np.random.default_rng(1)
    span = int(reg["end"].iloc[-1])
    rows = [("chr21", int(x), f"bg{i}", 1e-9) for i, x in enumerate(rng.integers(1_000_000, span, 60))]
    a = int(reg["start"].iloc[30])
    rows += [("chr21", a + 10_000 + 12_000 * j, f"pl{j}", 1e-12) for j in range(10)]
    s = run(tmp_path, reg, rows, clump=10_000, permutations=200)["summary"]
    assert s["top_region"] == reg["region"].iloc[30] and s["regions_enriched_q<0.05"] >= 1
    assert s["empirical_fdr"] < 0.2, s["empirical_fdr"]

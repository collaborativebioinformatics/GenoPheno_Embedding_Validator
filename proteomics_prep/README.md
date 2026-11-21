# Proteins into haploblocks: paired tumor/normal proteomics as a graph

Version v0.1.0 (see `CHANGELOG.md`). License: MIT (see `LICENSE` at the top of this repository).

Small R scripts that prepare paired tumor/normal proteomics data for graph and embedding methods built on
haploblocks. They take a proteomics table (log2 values, proteins × samples, with paired tumor and
adjacent-normal samples from the same patients), and:

1. place each eligible protein in the haploblock that holds the transcription start site (TSS) of its gene;
2. compute tumor minus normal (T-N) per protein and patient, with QC;
3. write node and edge CSVs and a protein × patient T-N matrix.

The CSVs can be loaded into graph tools (for example PyTorch Geometric, NetworkX or Neo4j) for early fusion, with
proteins attached to their blocks. The matrix can be embedded on its own for late fusion.

The approach follows the ProGenome idea of labeling each haploblock with its proteins
(https://github.com/collaborativebioinformatics/ProGenome), but it is a separate implementation. One difference:
ProGenome attaches a protein to every block its gene overlaps. Here a protein goes to the one block that contains
its gene's TSS. Only eligible proteins get a block (one gene, found, TSS inside a block, symbol not ambiguous);
all other proteins stay unassigned, and `map_status` says why.

The node, edge and matrix files use the headers listed in the schema table below. Matching headers is all that is
claimed: whether a given downstream loader accepts the files has to be checked with that loader.

**Column names.** A few column names keep the British spelling of the published interface and are not renamed:
`prot_tumour`, `prot_normal`, `rna_tumour`, `rna_normal`, `centring_set`, `tumour_signature` and
`tumour_signature_r`. The rest of the text uses American spelling.

## Steps

| Step | Script | What it does | Main output |
|---|---|---|---|
| 1 | `bin/block_partition.R` | Reads the block partition, converts chromosome names to `chr` style, adds stable block ids and chromosome arms | `blocks.tsv` |
| 2 | `bin/gene_models.R` | Reads gene records from a GENCODE GTF, with 0-based coordinates and the TSS | `genes.tsv` |
| 3 | `bin/feature_map.R` | Places each gene in the block holding its TSS, then maps proteins to genes by gene symbol | `features.tsv` (one row per protein, with `block_id`, `cis_ok`, `map_status`) |
| 4 | `bin/paired_tn.R` | log2 T − log2 N per protein and patient, plus two centered versions and QC | `protein_tn.tsv.gz` |
| 5 | `bin/protein_graph.R` | Node and edge CSVs and the T-N matrix | `graph/`, `proteins/` |

`bin/utils.R` holds shared helpers and is loaded by every script.

Run all five on a small made-up example (16 fake genes, 17 fake proteins, 6 fake patients). The script then runs
step 5 a second time on `example/protein_tn_long.tsv`, a long table of T-N values, which is the route that skips
step 4:

```bash
bash run_example.sh                 # or: bash run_example.sh /path/to/Rscript
```

Run the tests (testthat, tiny fixtures, about two minutes):

```bash
bash run_tests.sh                   # or: bash run_tests.sh /path/to/Rscript
```

## Requirements

R 4.3 or later with tidyverse (dplyr ≥ 1.1 for `join_by()`, readr ≥ 2), optparse and cowplot (cairo graphics for the
PDF plots); testthat for the tests. Create and activate an environment, for example:

```bash
mamba create -n protprep -c conda-forge "r-base>=4.3" r-tidyverse r-optparse r-cowplot r-testthat
mamba activate protprep             # or: conda activate protprep
```

Without activation, pass the environment's `Rscript` to the helper scripts, for example
`bash run_example.sh ~/miniforge3/envs/protprep/bin/Rscript`.

## Inputs for real data

- **Block partition:** `assets/haploblocks_org_blocks.tsv` is the haploblocks.org partition (GRCh38); see
  "Provenance of bundled files". Any TSV with `chrom`, `start`, `end` (0-based, half-open, upper or lower case)
  also works.
- **Centromeres:** `assets/centromeres_hg38.tsv` (GRCh38 centromere bands), used for arm labels.
- **Gene annotation:** a GENCODE GTF for GRCh38 (for example `gencode.v44.annotation.gtf.gz`, plain or gzipped).
- **Chromosome names and build:** steps 1 and 2 convert names to `chr` style by default (`1` → `chr1`,
  `MT` → `chrM`; `--chrom_style asis` keeps them). Contigs other than chr1-22, X, Y (and M for genes) are dropped
  and reported, and an empty result stops the run. `--assembly` (default `GRCh38`) is a label stored in the
  `genome_build` column; step 3 stops if the two tables carry different labels. The label does not check the
  coordinates: use a partition and a GTF of the same build. Step 3 reports how many genes fall in no block and
  warns loudly if most do (a typical sign of a build or naming mismatch).
- **Protein list** (TSV): `feature_id`, `gene_symbol`. Use `|` between symbols when one assay targets several
  genes. Other columns (for example UniProt) are kept; a column whose name is also computed by the scripts (such as
  `chrom`, `map_status` or `cis_ok`) is kept as `input_<name>` so that it cannot overwrite the computed one.
  `feature_id` must be unique.
- **Protein matrix** (TSV or CSV): first column = `feature_id`, then one column per sample, log2 values. Ids are read
  as text (`00123` stays `00123`); all other columns must be numbers (empty or `NA` = missing). The scripts stop on
  values that cannot be read, on missing or duplicated ids and on duplicated sample names. If your values are on a
  linear scale, take log2 first. The scripts subtract tumor minus normal themselves.
- **Sample sheet** (TSV): `patient`, `prot_tumour`, `prot_normal` (sample names = matrix column names, `;` between
  replicate pairs, so a sample name may not contain `;`). Rules:
  - one row per patient; a patient listed twice is an error;
  - the tumor and normal lists of a patient must have the same number of names, and pair up by position;
  - the same tumor/normal pair twice, or a sample name used both as tumor and as normal, is an error;
  - a pair with a sample name that is not a matrix column is skipped; patients with no usable pair are left out.
    Both are listed in `protein_excluded.tsv`.

  Optional `has_wgs` (TRUE/FALSE) only colors the QC plot.
- **Skipping step 4:** if you already have one T-N value per protein and patient, give step 5 a long table with the
  columns `feature_id`, `patient` and the value column named by `--value` (default `delta`); see
  `example/protein_tn_long.tsv`. Keys must not be missing, each (`feature_id`, `patient`) pair may appear once, and
  values must be numbers. Duplicated protein rows in `features.tsv` are also an error.

## Step 4: formulas

For protein `f` and patient `p`, with log2 values `T` and `N` for each usable replicate pair `r` of the patient:

- `delta[f,p]` = mean over the pairs `r` that are non-missing for `f` of (`T[f,r]` − `N[f,r]`). A replicate that is
  missing for one protein is therefore skipped for that protein, not for the whole patient. `n_pairs_used` in the
  output table gives the number of pairs averaged; `n_pairs` is the patient's number of usable pairs. `delta` is
  `NA` when no pair is non-missing.
- **Centering set:** the proteins that are expressed (mean log2 level ≥ `--min_ave`, if given) and do not show a
  strong cohort-wide T-N change. A strong change means Benjamini-Hochberg q < `--strong_q` (default 0.01) from a
  one-sample t-test of `delta` over patients, and |mean `delta`| > `--strong_lfc` (default 1). The set is found
  from `delta` and is the same for all patients (column `centring_set` in `protein_feature_stats.tsv`).
- `delta_pc[f,p]` = `delta[f,p]` − median over the centering set of `delta[·,p]` (missing values ignored).
- `delta_dc[f,p]` = `dc[f,p]` − median over the centering set of `dc[·,p]`, where
  `dc[f,p]` = `delta[f,p]` − mean over patients of `delta[f,·]`. The protein mean is subtracted first, over all
  patients with a value; the patient median, taken over the centering set, is calculated afterwards on the already
  protein-centered values.
- `--clamp` limits `delta_pc` and `delta_dc` to ± the given value. If the centering set is empty, `delta_pc` and
  `delta_dc` are `NA` and a warning is printed.

`delta_dc` removes each protein's average tumor effect in the cohort, including any systematic tumor-specific
effect that all patients share, so choose the version to match the question.

The T-N table (`protein_tn.tsv.gz`) and `protein_feature_stats.tsv` are written before the QC and the plots; a
plotting failure gives a warning and does not stop the run.

**Cohort-signature resemblance** (`tumour_signature_r` in `protein_patients.tsv`). For each patient: the Pearson
correlation of the patient's `delta` with the cohort mean `delta`, over the signature proteins (expressed and q <
0.01; if there are fewer than 20, all expressed proteins). Two limits: it is **scale-invariant** (a patient whose
tumor signal is weak, but has the same pattern, still scores close to 1, so it is not a measure of tumor content) and
**self-inclusive** (the patient is part of the cohort mean, which inflates r in small cohorts). It is `NA` when the
signature set is empty or a patient has fewer than 3 complete observations.

## Outputs of step 5 (schema)

| File | Columns |
|---|---|
| `graph/nodes_patient.csv` | `id`, `label` |
| `graph/nodes_block.csv` | `id`, `label`, `chrom`, `start`, `end`, `arm`, `haplograph_block_id` |
| `graph/nodes_protein.csv` | `id`, `label`, `gene_names` |
| `graph/edges_protein_on_block.csv` | `source` (protein), `target` (block), `type` = `ON_BLOCK` |
| `proteins/protein_features.tsv` | `protein_id`, `chrom`, `block_id`, `haplograph_block_id`, `pos` (TSS), `gene_names`, `gene_ids`, `symbol_ambiguous`, `map_status` |
| `proteins/protein_tn_matrix.tsv.gz` | `protein_id`, then one column per patient (T-N, log2) |

- Only eligible proteins get an `ON_BLOCK` edge: exactly one gene, found, with its TSS in exactly one block
  (`cis_ok`), and, by default, a gene symbol that is not ambiguous. The matrix keeps every protein, for late fusion.
  If no protein is eligible (for example all symbols are unmapped), all nodes and the matrix are still written and
  the edge file has only its header.
- `map_status` gives the reason when a protein has no edge:

  | `map_status` | Meaning |
  |---|---|
  | `mapped` | has an `ON_BLOCK` edge |
  | `unmapped` | no targeted gene symbol was found in the GTF |
  | `partly_unmapped` | some, not all, of several targeted symbols were found |
  | `no_block` | the gene was found, but its TSS is in no block |
  | `multi_block` | several targeted genes lie in different blocks |
  | `multi_gene` | several targeted genes, all in one block (no single locus) |
  | `ambiguous_symbol` | the symbol matched several equally ranked loci; no edge unless `--drop_ambiguous FALSE` |
  | `not_in_feature_list` | the protein is in the T-N table but not in `features.tsv` |

- `haplograph_block_id` is the block in the naming of the 1000 Genomes haplograph (`chr21_14215892-14284114`).
  Haplograph cluster node ids, as written by `1_build_graph.py` at the top of this repository, are this id plus
  `_cluster<N>`, so proteins can be joined to clusters by prefix. The ids match exactly when the bundled
  haploblocks.org partition is used in step 1.
- `symbol_ambiguous` and the selected `gene_ids` are kept in `protein_features.tsv`, so the choice of locus can be
  checked. With `--drop_ambiguous FALSE` the edge of such a protein is kept, and `symbol_ambiguous` still marks it.
  Step 3 has the same option (default `TRUE`): `block_features.tsv` then leaves ambiguous proteins out of its counts.
- `--value` chooses which T-N goes into the matrix: `delta` (default), `delta_pc` or `delta_dc` (formulas above).
- `--digits` (default 4) is the number of decimal places kept in the matrix; use a large value such as 15 to
  effectively not round.
- Block ids look like `chr1:5854228-6003595`. They match the block ids produced from the same partition.

## Notes

- Gene symbols must match the GTF's `gene_name`. Renamed symbols (old HGNC names) will be `unmapped`. Update
  them first, or use a GTF of the same release as your assay annotation.
- When several genes share a symbol, one locus is chosen: protein-coding first, then not the chrY PAR copy, then
  the first chromosome. Such proteins are flagged `symbol_ambiguous` when more than one locus remains equally
  ranked, and they get no edge by default (see above).
- `gene_models.R` reads the whole GTF (plain or gzipped) but keeps only the gene records as it goes; the memory use
  is small, the run time grows with the size of the file.
- The scripts write a few QC plots (`plots/`) next to their outputs.

## Provenance of bundled files

| File | Source | Assembly | MD5 checksum | Terms of reuse |
|---|---|---|---|---|
| `assets/haploblocks_org_blocks.tsv` | haploblocks.org (1000 Genomes haploblock partition and per-block cluster statistics), https://data.haploblocks.org/ | GRCh38 | `20e523b87450c074960ad668301c57ca` | CC0 1.0 (https://data.haploblocks.org/LICENSE.txt). Exact upstream file and version: to be confirmed |
| `assets/centromeres_hg38.tsv` | UCSC Genome Browser hg38 cytoBand table, `acen` bands merged per chromosome | GRCh38 | `03261f5afa79bf2314d7897feec0fdb3` | UCSC data use terms: to be confirmed |

## How to cite

To be added (software citation for this bundle, and the citations of the upstream data sources above).

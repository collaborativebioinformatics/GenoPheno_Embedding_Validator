# Changelog

## v0.1.0

First tagged version, after an internal review. Changes relative to the untagged draft:

- `paired_tn.R`: feature ids are read as text and sample values as numbers, with clear errors for parsing
  problems, missing or duplicated ids and infinite values. Replicate lists must have equal length per patient;
  duplicated patient rows, duplicated pairs and samples used as both tumor and normal are rejected. Patients or
  pairs that cannot be used are written to `<assay>_excluded.tsv`.
- `paired_tn.R`: a replicate that is missing for one feature is skipped for that feature (mean over the remaining
  pairs); `n_pairs_used` records how many pairs were averaged. Empty centering or signature sets give `NA` and a
  warning. The T-N table and feature statistics are written before the QC and the plots; a plotting failure only
  warns. The per-patient QC (`tumour_signature_r`) is described as cohort-signature resemblance.
- `feature_map.R`, `protein_graph.R`: explicit column types; an all-unmapped panel exports nodes and the matrix
  with zero edges. New `multi_gene` status, `symbol_ambiguous` and `gene_ids` carried into
  `protein_features.tsv`, `--drop_ambiguous` (default `TRUE`), `input_` prefix for clashing extra columns,
  `--digits` (default 4).
- `protein_graph.R`: the long-table route checks keys, duplicates and numeric values; an example long table and a
  step in `run_example.sh` exercise it.
- `block_partition.R`, `gene_models.R`: `--chrom_style` (default converts to `chr` style), dropped contigs are
  reported, an empty partition stops the run, `--assembly` label stored as `genome_build`; `feature_map.R` stops
  when the labels differ and warns when most genes fall in no block.
- `protein_graph.R`: `haplograph_block_id` (`chr21_14215892-14284114`) in `nodes_block.csv` and
  `protein_features.tsv`, to join proteins to 1000 Genomes haplograph cluster nodes by prefix.
- `tests/` (testthat, `bash run_tests.sh`), quoted paths in `run_example.sh`, README: environment, minimum
  versions, schema table, formulas, provenance and citation placeholders.

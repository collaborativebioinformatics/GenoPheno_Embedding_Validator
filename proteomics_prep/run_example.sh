#!/bin/bash
# Run the five steps on the made-up example in example/ (about 30 s), then the
# shortcut that skips step 4 (a long table of T-N values given to step 5).
# Usage: bash run_example.sh [Rscript]      outputs go to example_out/
set -euo pipefail
R=${1:-Rscript}
cd "$(dirname "$0")"
OUT=example_out
mkdir -p "$OUT"

# 1. Block partition (the public haploblocks.org table, with chromosome arms)
"$R" bin/block_partition.R --blocks assets/haploblocks_org_blocks.tsv \
  --centromeres assets/centromeres_hg38.tsv --out "$OUT/blocks.tsv"

# 2. Gene models (use a GENCODE GTF for real data, e.g. gencode.v44.annotation.gtf.gz)
"$R" bin/gene_models.R --gtf_genes example/genes.gtf --out "$OUT/genes.tsv"

# 3. Proteins into blocks (via the TSS of their gene)
"$R" bin/feature_map.R --genes "$OUT/genes.tsv" --blocks "$OUT/blocks.tsv" \
  --protein_features example/protein_features.tsv --outdir "$OUT/feature_map"

# 4. Tumor - normal per protein and patient
"$R" bin/paired_tn.R --assay protein --matrix example/protein_matrix.tsv \
  --samplesheet example/samplesheet.tsv --outdir "$OUT/paired"

# 5. Node and edge CSVs and the T-N matrix
"$R" bin/protein_graph.R --blocks "$OUT/blocks.tsv" --features "$OUT/feature_map/features.tsv" \
  --tn "$OUT/paired/protein_tn.tsv.gz" --outdir "$OUT/graph_export"

# 5b. Skip step 4: give step 5 your own long table (feature_id, patient, delta)
"$R" bin/protein_graph.R --blocks "$OUT/blocks.tsv" --features "$OUT/feature_map/features.tsv" \
  --tn example/protein_tn_long.tsv --outdir "$OUT/graph_export_long"

# The two routes must give the same graph edges (the example table holds the same T-N values).
diff "$OUT/graph_export/graph/edges_protein_on_block.csv" \
     "$OUT/graph_export_long/graph/edges_protein_on_block.csv"

echo "done: see $OUT/graph_export/ and $OUT/graph_export_long/"

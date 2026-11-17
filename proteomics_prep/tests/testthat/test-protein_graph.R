# Tests for bin/protein_graph.R

panel_features <- function() {
  panel.df <- tibble(feature_id = c("PA", "PAMB", "PX"), gene_symbol = c("GENE_A", "AMB", "NOPE"))
  map_panel(panel.df)$features
}

test_that("an all-unmapped panel still exports nodes and the matrix, with zero edges", {
  mapped <- map_panel(tibble(feature_id = c("U1", "U2"), gene_symbol = c("NOPE1", "NOPE2")))
  expect_script_ok(mapped$result)
  exported <- export_graph(mapped$features, long_tn(c("U1", "U2")))
  expect_script_ok(exported$result)
  edges.df <- read_text_csv(file.path(exported$outdir, "graph", "edges_protein_on_block.csv"))
  expect_equal(nrow(edges.df), 0)
  expect_equal(names(edges.df), c("source", "target", "type"))
  expect_equal(nrow(read_text_csv(file.path(exported$outdir, "graph", "nodes_protein.csv"))), 2)
  expect_equal(nrow(read_text_tsv(file.path(exported$outdir, "proteins", "protein_tn_matrix.tsv.gz"))), 2)
  status <- read_text_tsv(file.path(exported$outdir, "proteins", "protein_features.tsv"))$map_status
  expect_equal(status, c("unmapped", "unmapped"))
})

test_that("a symbol-ambiguous protein gets no edge by default, and one with --drop_ambiguous FALSE", {
  features <- panel_features()
  default_run <- export_graph(features, long_tn(c("PA", "PAMB", "PX")))
  expect_script_ok(default_run$result)
  edges.df <- read_text_csv(file.path(default_run$outdir, "graph", "edges_protein_on_block.csv"))
  expect_equal(edges.df$source, "PA")
  protein.df <- read_text_tsv(file.path(default_run$outdir, "proteins", "protein_features.tsv"))
  expect_equal(protein.df$map_status[protein.df$protein_id == "PAMB"], "ambiguous_symbol")
  expect_equal(protein.df$symbol_ambiguous[protein.df$protein_id == "PAMB"], "TRUE")
  expect_true(!is.na(protein.df$gene_ids[protein.df$protein_id == "PAMB"]))

  keep_run <- export_graph(features, long_tn(c("PA", "PAMB", "PX")), "--drop_ambiguous", "FALSE")
  expect_script_ok(keep_run$result)
  kept.df <- read_text_csv(file.path(keep_run$outdir, "graph", "edges_protein_on_block.csv"))
  expect_setequal(kept.df$source, c("PA", "PAMB"))
})

test_that("a long table with a duplicated (feature_id, patient) row is rejected", {
  duplicated_key.df <- bind_rows(long_tn(c("PA", "PX")), long_tn("PA")[1, ])
  exported <- export_graph(panel_features(), duplicated_key.df)
  expect_script_fails(exported$result, "duplicated (feature_id, patient)")
})

test_that("a long table with a non-numeric value or a missing key is rejected", {
  text_value.df <- long_tn(c("PA", "PX")) %>% mutate(delta = as.character(delta))
  text_value.df$delta[2] <- "high"
  expect_script_fails(export_graph(panel_features(), text_value.df)$result, "could not be parsed")

  missing_key.df <- long_tn(c("PA", "PX"))
  missing_key.df$patient[1] <- NA
  expect_script_fails(export_graph(panel_features(), missing_key.df)$result, "missing patient")
})

test_that("a long table without the chosen value column is rejected", {
  wrong_name.df <- long_tn("PA") %>% rename(log2_diff = delta)
  expect_script_fails(export_graph(panel_features(), wrong_name.df)$result, "lacks column(s)")
})

test_that("--digits sets the rounding of the matrix", {
  tn.df <- tibble(feature_id = "PA", patient = c("PA1", "PA2"), delta = c(1.23456789, -0.98765432))
  exported <- export_graph(panel_features(), tn.df, "--digits", "2")
  expect_script_ok(exported$result)
  matrix.df <- read_text_tsv(file.path(exported$outdir, "proteins", "protein_tn_matrix.tsv.gz"))
  expect_equal(matrix.df$PA1, "1.23")
  expect_equal(matrix.df$PA2, "-0.99")
})

test_that("blocks and proteins carry the haplograph block id (chrom_start-end)", {
  exported <- export_graph(panel_features(), long_tn(c("PA", "PX")))
  expect_script_ok(exported$result)
  blocks.df <- read_text_csv(file.path(exported$outdir, "graph", "nodes_block.csv"))
  expect_equal(blocks.df$haplograph_block_id, sub(":", "_", blocks.df$id, fixed = TRUE))
  expect_true(all(grepl("^chr[^_:]+_[0-9]+-[0-9]+$", blocks.df$haplograph_block_id)))
  protein.df <- read_text_tsv(file.path(exported$outdir, "proteins", "protein_features.tsv"))
  expect_equal(protein.df$haplograph_block_id[protein.df$protein_id == "PA"],
               sub(":", "_", protein.df$block_id[protein.df$protein_id == "PA"], fixed = TRUE))
  expect_true(is.na(protein.df$haplograph_block_id[protein.df$protein_id == "PX"]))   # unmapped: no block
})

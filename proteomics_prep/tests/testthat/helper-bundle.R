# Helpers shared by all test files: run a script of the bundle, write tiny
# input tables, and build the block and gene tables from the fixtures once.

library(testthat)
suppressPackageStartupMessages(library(tidyverse))

bundle_dir <- normalizePath(test_path("..", ".."))
fixture <- function(name) file.path(bundle_dir, "tests", "fixtures", name)
rscript <- file.path(R.home("bin"), "Rscript")

# Run bin/<script> with a character vector of arguments. Returns the exit
# status and everything the script printed (messages and warnings included).
run_script <- function(script, args) {
  output <- suppressWarnings(
    system2(rscript, c(shQuote(file.path(bundle_dir, "bin", script)), shQuote(args)),
            stdout = TRUE, stderr = TRUE))
  list(status = attr(output, "status") %||% 0L, output = paste(output, collapse = "\n"))
}

expect_script_ok <- function(result) {
  expect_equal(result$status, 0L, info = result$output)
}

expect_script_fails <- function(result, pattern) {
  expect_gt(result$status, 0L)
  expect_match(result$output, pattern, fixed = TRUE)
}

# Write a data frame as a TSV; returns the path.
write_input <- function(data.df, name, dir = tempdir()) {
  path <- file.path(dir, name)
  write_tsv(data.df, path, na = "")
  path
}

# Read every column as text, so tests see exactly what was written.
read_text_tsv <- function(path) {
  read_tsv(path, col_types = cols(.default = "c"), show_col_types = FALSE)
}

# Same for a CSV file.
read_text_csv <- function(path) {
  read_csv(path, col_types = cols(.default = "c"), show_col_types = FALSE)
}

# Blocks and genes built once from the fixtures (chr style, GRCh38 label).
reference <- local({
  cache <- NULL
  function() {
    if (is.null(cache)) {
      dir <- file.path(tempdir(), "reference")
      dir.create(dir, showWarnings = FALSE)
      blocks_result <- run_script("block_partition.R", c(
        "--blocks", fixture("blocks_chr.tsv"), "--centromeres", fixture("centromeres_tiny.tsv"),
        "--out", file.path(dir, "blocks.tsv")))
      genes_result <- run_script("gene_models.R", c(
        "--gtf_genes", fixture("genes_chr.gtf"), "--out", file.path(dir, "genes.tsv")))
      expect_script_ok(blocks_result)
      expect_script_ok(genes_result)
      cache <<- list(blocks = file.path(dir, "blocks.tsv"), genes = file.path(dir, "genes.tsv"))
    }
    cache
  }
})

# Run step 3 on a protein panel (feature_id, gene_symbol, ...) with the
# reference tables; returns the result and the output directory.
map_panel <- function(panel.df, genes = reference()$genes, blocks = reference()$blocks) {
  outdir <- tempfile("feature_map_")
  panel_file <- write_input(panel.df, basename(tempfile("panel_", fileext = ".tsv")))
  result <- run_script("feature_map.R", c("--genes", genes, "--blocks", blocks,
                                           "--protein_features", panel_file, "--outdir", outdir))
  list(result = result, outdir = outdir, features = file.path(outdir, "features.tsv"))
}

# Run step 5 on a features table and a long T-N table (feature_id, patient, delta).
export_graph <- function(features, tn.df, ...) {
  outdir <- tempfile("graph_")
  tn_file <- write_input(tn.df, basename(tempfile("tn_", fileext = ".tsv")))
  result <- run_script("protein_graph.R", c("--blocks", reference()$blocks, "--features", features,
                                             "--tn", tn_file, "--outdir", outdir, ...))
  list(result = result, outdir = outdir)
}

# A long T-N table with one value per protein and patient.
long_tn <- function(feature_ids, patients = c("PA", "PB", "PC")) {
  expand_grid(feature_id = feature_ids, patient = patients) %>%
    mutate(delta = round(seq_along(feature_id) / 7, 3))
}

# Tests for bin/feature_map.R

test_that("a position on a block boundary belongs to the block that starts there", {
  panel.df <- tibble(feature_id = c("PC", "PD", "PB"), gene_symbol = c("GENE_C", "GENE_D", "GENE_B"))
  mapped <- map_panel(panel.df)
  expect_script_ok(mapped$result)
  features.df <- read_text_tsv(mapped$features)
  block_of <- function(id) features.df$block_id[features.df$feature_id == id]
  expect_equal(block_of("PC"), "chr1:1000-2000")   # TSS 1000: first base of the second block
  expect_equal(block_of("PD"), "chr1:0-1000")      # TSS 999: last base of the first block
  expect_equal(block_of("PB"), "chr1:1000-2000")   # minus strand: TSS = end - 1 = 1099, not start 800
})

test_that("feature ids keep their exact spelling", {
  ids <- c("10000-28", "00123", "12345678901234567890")
  mapped <- map_panel(tibble(feature_id = ids, gene_symbol = c("GENE_A", "GENE_C", "GENE_E")))
  expect_script_ok(mapped$result)
  expect_setequal(read_text_tsv(mapped$features)$feature_id, ids)
  exported <- export_graph(mapped$features, long_tn(ids))
  expect_script_ok(exported$result)
  matrix.df <- read_text_tsv(file.path(exported$outdir, "proteins", "protein_tn_matrix.tsv.gz"))
  expect_equal(matrix.df$protein_id, ids)
  expect_equal(read_text_csv(file.path(exported$outdir, "graph", "nodes_protein.csv"))$id, ids)
})

test_that("every protein without an edge has a reason, including multi_gene", {
  panel.df <- tibble(feature_id = c("ok", "multi_gene", "multi_block", "unmapped", "partly"),
                     gene_symbol = c("GENE_A", "GENE_A|GENE_D", "GENE_A|GENE_C", "NOPE", "GENE_A|NOPE"))
  mapped <- map_panel(panel.df)
  expect_script_ok(mapped$result)
  status <- read_text_tsv(mapped$features) %>% select(feature_id, map_status) %>% deframe()
  expect_equal(status[["ok"]], "mapped")
  expect_equal(status[["multi_gene"]], "multi_gene")
  expect_equal(status[["multi_block"]], "multi_block")
  expect_equal(status[["unmapped"]], "unmapped")
  expect_equal(status[["partly"]], "partly_unmapped")
})

test_that("a symbol shared by two equally ranked loci is flagged as ambiguous", {
  mapped <- map_panel(tibble(feature_id = c("PAMB", "PA"), gene_symbol = c("AMB", "GENE_A")))
  features.df <- read_text_tsv(mapped$features)
  expect_equal(features.df$symbol_ambiguous[features.df$feature_id == "PAMB"], "TRUE")
  expect_equal(features.df$symbol_ambiguous[features.df$feature_id == "PA"], "FALSE")
})

test_that("extra input columns that clash with computed columns are prefixed", {
  panel.df <- tibble(feature_id = "PA", gene_symbol = "GENE_A", chrom = "my_chrom",
                     map_status = "my_status", uniprot = "Q00000")
  mapped <- map_panel(panel.df)
  expect_script_ok(mapped$result)
  features.df <- read_text_tsv(mapped$features)
  expect_equal(features.df$input_chrom, "my_chrom")
  expect_equal(features.df$input_map_status, "my_status")
  expect_equal(features.df$uniprot, "Q00000")
  expect_equal(features.df$chrom, "chr1")
  expect_equal(features.df$map_status, "mapped")
})

test_that("genes and blocks with different assembly labels stop the run", {
  other_genes <- tempfile(fileext = ".tsv")
  expect_script_ok(run_script("gene_models.R", c("--gtf_genes", fixture("genes_chr.gtf"),
                                                 "--assembly", "GRCh37", "--out", other_genes)))
  mapped <- map_panel(tibble(feature_id = "PA", gene_symbol = "GENE_A"), genes = other_genes)
  expect_script_fails(mapped$result, "genome build differs")
})

test_that("genes that fall in no block give a loud warning", {
  elsewhere_gtf <- tempfile(fileext = ".gtf")
  read_lines(fixture("genes_chr.gtf")) %>% str_replace("^chr[12]\t", "chr3\t") %>% write_lines(elsewhere_gtf)
  elsewhere_genes <- tempfile(fileext = ".tsv")
  expect_script_ok(run_script("gene_models.R", c("--gtf_genes", elsewhere_gtf, "--out", elsewhere_genes)))
  mapped <- map_panel(tibble(feature_id = "PA", gene_symbol = "GENE_A"), genes = elsewhere_genes)
  expect_script_ok(mapped$result)
  expect_match(mapped$result$output, "MOST GENES", fixed = TRUE)
})

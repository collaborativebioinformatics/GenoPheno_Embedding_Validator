# Tests for bin/gene_models.R

test_that("the TSS is the start on the plus strand and end - 1 on the minus strand", {
  genes.df <- read_text_tsv(reference()$genes)
  tss_of <- function(name) as.numeric(genes.df$tss[genes.df$gene_name == name])
  expect_equal(tss_of("GENE_A"), 100)    # + strand, 1-based start 101
  expect_equal(tss_of("GENE_B"), 1099)   # - strand, end 1100, so TSS = 1099
  expect_equal(tss_of("GENE_D"), 999)    # - strand, end 1000, so TSS = 999
})

test_that("a GTF with Ensembl-style names (1, MT) gives chr-style genes", {
  out <- tempfile(fileext = ".tsv")
  result <- run_script("gene_models.R", c("--gtf_genes", fixture("genes_nochr.gtf"), "--out", out))
  expect_script_ok(result)
  genes.df <- read_text_tsv(out)
  expect_setequal(unique(genes.df$chrom), c("chr1", "chr2", "chrM"))
  expect_match(result$output, "KI270706.1", fixed = TRUE)   # the dropped contig is reported
  expect_equal(unique(genes.df$genome_build), "GRCh38")
})

test_that("a GTF with no usable chromosome stops", {
  out <- tempfile(fileext = ".tsv")
  result <- run_script("gene_models.R", c("--gtf_genes", fixture("genes_nochr.gtf"),
                                          "--chrom_style", "asis", "--out", out))
  expect_script_fails(result, "no gene is left")   # asis keeps 1, 2, MT, which are not chr names
})

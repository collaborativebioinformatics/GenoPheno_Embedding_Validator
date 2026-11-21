# Tests for bin/block_partition.R

partition_blocks <- function(blocks_file, ...) {
  out <- tempfile(fileext = ".tsv")
  result <- run_script("block_partition.R", c("--blocks", blocks_file,
                                              "--centromeres", fixture("centromeres_tiny.tsv"),
                                              "--out", out, ...))
  list(result = result, out = out)
}

test_that("a partition written as 1, 2 gives the same blocks as chr1, chr2", {
  chr_style <- partition_blocks(fixture("blocks_chr.tsv"))
  plain_style <- partition_blocks(fixture("blocks_nochr.tsv"))
  expect_script_ok(chr_style$result)
  expect_script_ok(plain_style$result)
  expect_equal(read_text_tsv(plain_style$out)$block_id, read_text_tsv(chr_style$out)$block_id)
  expect_equal(read_text_tsv(plain_style$out)$block_id,
               c("chr1:0-1000", "chr1:1000-2000", "chr2:0-1000"))
})

test_that("dropped contigs are reported", {
  plain_style <- partition_blocks(fixture("blocks_nochr.tsv"))
  expect_match(plain_style$result$output, "GL000220.1", fixed = TRUE)
  expect_match(plain_style$result$output, "chrM", fixed = TRUE)
})

test_that("--chrom_style asis on a partition without chr stops with an empty partition", {
  asis <- partition_blocks(fixture("blocks_nochr.tsv"), "--chrom_style", "asis")
  expect_script_fails(asis$result, "no block is left")
})

test_that("the assembly label is written to the blocks table", {
  labeled <- partition_blocks(fixture("blocks_chr.tsv"), "--assembly", "TESTBUILD")
  expect_script_ok(labeled$result)
  expect_equal(unique(read_text_tsv(labeled$out)$genome_build), "TESTBUILD")
})

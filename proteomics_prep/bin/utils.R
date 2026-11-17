#################################################
# Title: Shared R helpers                       #
# Date: November 10, 2026                       #
# Notes: Sourced by every script in bin/.       #
#   Holds the plot style (firebrick / grey,     #
#   theme_bw, 7 pt text, PDF in mm), the        #
#   command-line helper, input checks, the      #
#   chromosome-name helpers and the small       #
#   functions that place genes and proteins in  #
#   blocks.                                     #
#################################################

#############
# Libraries #
#############

suppressPackageStartupMessages({
  library(tidyverse)
  library(optparse)
})

###############
# Plot style  #
###############

# One focus color (firebrick) and greys for everything else.
col_focus <- "firebrick"
col_other <- "grey70"
col_dark  <- "grey30"

theme_simple <- function(base_size = 7) {
  theme_bw() +
    theme(text = element_text(size = base_size),
          panel.grid.minor = element_blank(),
          strip.background = element_rect(fill = "grey95"))
}

# Save a plot as PDF, with width and height in mm.
save_pdf <- function(plot, filename, width = 100, height = 70) {
  ggsave(filename = filename, plot = plot, width = width, height = height,
         units = "mm", device = cairo_pdf)
}

#####################
# Command line      #
#####################

# Parse options and stop with a clear message if a required one is missing.
parse_args_required <- function(option_list, required = character()) {
  opt <- parse_args(OptionParser(option_list = option_list))
  missing <- required[vapply(required, function(x) is.null(opt[[x]]), logical(1))]
  if (length(missing) > 0) {
    stop("missing required option(s): ", paste0("--", missing, collapse = ", "))
  }
  opt
}

# Read a TSV in which missing values may be written as "." as well as "" or NA.
read_tsv_dot <- function(file, ...) {
  read_tsv(file, na = c("", "NA", "."), show_col_types = FALSE, progress = FALSE, ...)
}

#############
# Functions #
#############

# Stop with a readable message when readr reported parsing problems (for
# example a non-numeric value in a numeric column). Call it right after the
# read: the problems are stored on the table that read_tsv() returned.
stop_on_parse_problems <- function(data.df, what) {
  problems.df <- readr::problems(data.df)
  if (nrow(problems.df) > 0) {
    first.df <- head(problems.df, 3)
    examples <- sprintf("row %s, column %s: expected %s, found '%s'",
                        first.df$row, first.df$col, first.df$expected, first.df$actual)
    stop(what, ": ", nrow(problems.df), " value(s) could not be parsed. First: ",
         paste(examples, collapse = " | "), call. = FALSE)
  }
  invisible(data.df)
}

# Write a plot, but only warn when drawing or saving fails, so that a
# plotting problem never prevents the tables from being produced.
# make_plot is a function without arguments that returns the plot.
save_plot_safely <- function(make_plot, filename, ...) {
  tryCatch(
    save_pdf(make_plot(), filename, ...),
    error = function(e) {
      warning("could not write ", basename(filename), ": ", conditionMessage(e), call. = FALSE)
    })
  invisible(file.exists(filename))
}

# Chromosome names in "chr" style: 1 -> chr1, X -> chrX, MT or M -> chrM.
# Names that already start with "chr" keep it. Anything that is not a
# numbered chromosome, X, Y or the mitochondrion (for example a scaffold such
# as KI270706.1) is returned unchanged, so that it can be reported and dropped.
normalize_chrom <- function(chrom_values) {
  bare <- toupper(sub("^chr", "", chrom_values, ignore.case = TRUE))
  is_standard <- !is.na(bare) & grepl("^([0-9]+|X|Y|M|MT)$", bare)
  bare <- if_else(bare == "MT", "M", bare)
  if_else(is_standard, paste0("chr", bare), chrom_values)
}

# Print which contigs were dropped and how many records each one had.
report_dropped_contigs <- function(dropped_chrom_values, what) {
  if (length(dropped_chrom_values) == 0) return(invisible(NULL))
  counts.df <- tibble(chrom = dropped_chrom_values) %>% count(chrom, sort = TRUE)
  shown <- head(counts.df, 10)
  message(what, ": dropped ", length(dropped_chrom_values), " record(s) on ", nrow(counts.df),
          " other contig(s): ", paste0(shown$chrom, " (", shown$n, ")", collapse = ", "),
          if (nrow(counts.df) > nrow(shown)) ", ..." else "")
}

# Stable block identifier, e.g. "chr1:18000000-18100000". sprintf avoids
# scientific notation (paste0 would give "1.2e+08" for round numbers).
make_block_id <- function(chrom_values, starts, ends) {
  sprintf("%s:%.0f-%.0f", chrom_values, as.numeric(starts), as.numeric(ends))
}

# The same block in the naming of the 1000 Genomes haplograph (haploblocks.org):
# "chr21:14215892-14284114" -> "chr21_14215892-14284114". Haplograph cluster
# node ids add "_cluster<N>" to this, so proteins can be joined to clusters by
# prefix. Matches exactly when the partition is the haploblocks.org one.
make_haplograph_block_id <- function(block_id) {
  sub(":", "_", block_id, fixed = TRUE)
}

# Chromosome arm of each block from its midpoint. Blocks are half-open
# [start, end). A block that overlaps the centromere band is flagged, and
# still gets the arm of its midpoint.
assign_arm <- function(chrom_values, starts, ends, centromeres.df) {
  cen <- centromeres.df[match(chrom_values, centromeres.df$chrom), ]
  mid <- (starts + ends) / 2
  cen_mid <- (cen$cen_start + cen$cen_end) / 2
  arm <- ifelse(is.na(cen_mid), NA_character_,
                paste0(sub("^chr", "", chrom_values), ifelse(mid < cen_mid, "p", "q")))
  spans <- !is.na(cen_mid) & starts < cen$cen_end & ends > cen$cen_start
  tibble(arm = arm, spans_centromere = spans)
}

# Ensembl gene id without its version. The GENCODE "_PAR_Y" suffix is kept
# so the chrY pseudo-autosomal copies stay distinct from the chrX ones.
strip_version <- function(gene_ids) {
  sub("\\.[0-9]+(_PAR_Y)?$", "\\1", gene_ids)
}

# Block that contains each position (0-based, half-open blocks [start, end)).
# Returns block_id, NA where no block contains the position.
locate_in_blocks <- function(chrom_values, positions, blocks.df) {
  query.df <- tibble(.row = seq_along(chrom_values), chrom = chrom_values, pos = positions)
  hits.df <- query.df %>%
    filter(!is.na(pos)) %>%
    inner_join(blocks.df %>% select(chrom, start, end, block_id),
               by = join_by(chrom, pos >= start, pos < end))
  stopifnot(!anyDuplicated(hits.df$.row))   # blocks must not overlap
  hits.df$block_id[match(query.df$.row, hits.df$.row)]
}

# Number of blocks that a 0-based, half-open interval overlaps.
count_overlapping_blocks <- function(chrom_values, starts, ends, blocks.df) {
  query.df <- tibble(.row = seq_along(chrom_values), chrom = chrom_values, q_start = starts, q_end = ends)
  hits.df <- query.df %>%
    inner_join(blocks.df %>% select(chrom, start, end),
               by = join_by(chrom, q_start < end, q_end > start)) %>%
    count(.row)
  out <- hits.df$n[match(query.df$.row, hits.df$.row)]
  replace_na(out, 0L)
}

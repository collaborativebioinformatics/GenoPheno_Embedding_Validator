#!/usr/bin/env Rscript
#################################################
# Title: Haploblock partition                   #
# Date: November 10, 2026                       #
# Notes: Step 1. Reads the block partition      #
#   (CHROM, START, END, and SEGDUP_FRAC if      #
#   present, upper or lower case; half-open     #
#   [START, END)) and writes blocks.tsv with a  #
#   stable block_id, a running index, length,   #
#   chromosome arm, segdup fraction and the     #
#   genome build label (genome_build).          #
#   Chromosome names are converted to "chr"     #
#   style by default (1 -> chr1), so that       #
#   partitions and gene models written in       #
#   either style can be combined. Contigs other #
#   than chr1-22, X and Y are dropped and       #
#   reported; an empty result is an error.      #
#   Every later layer is keyed on block_id.     #
#   Optional: join the                          #
#   haploblocks.org / HaploGraph block table on #
#   exact coordinates (in_haploblocks_org,      #
#   hb_block_id, hb_n_clusters, ...).           #
#################################################

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(script_dir, "utils.R"))

##############
# Parameters #
##############

opt <- parse_args_required(list(
  make_option("--blocks", type = "character", help = "block partition TSV (CHROM START END ...)"),
  make_option("--centromeres", type = "character", help = "centromere bands TSV (chrom cen_start cen_end)"),
  make_option("--haploblocks_org", type = "character", default = NULL,
              help = "haploblocks.org block table (chrom start end hb_*); optional"),
  make_option("--chrom_style", type = "character", default = "chr",
              help = "chr: rename chromosomes to chr style (1 -> chr1); asis: keep the names as given [chr]"),
  make_option("--assembly", type = "character", default = "GRCh38",
              help = "genome build label stored in the output (column genome_build) [GRCh38]"),
  make_option("--out", type = "character", default = "blocks.tsv", help = "output TSV")
), required = c("blocks", "centromeres"))

chrom_style <- match.arg(opt$chrom_style, c("chr", "asis"))
chrom_order <- paste0("chr", c(1:22, "X", "Y"))

# Apply the chosen chromosome naming to a vector of chromosome names.
style_chrom <- function(chrom_values) {
  if (chrom_style == "chr") normalize_chrom(chrom_values) else chrom_values
}

#############
# Read data #
#############

# Read every column as text first, so that numeric-looking chromosome names
# (1, 2, ...) stay text; START and END are converted below with a check.
partition.df <- read_tsv_dot(opt$blocks, col_types = cols(.default = "c"))
# Accept lower-case column names too (e.g. the public haploblocks.org table in assets/)
names(partition.df) <- toupper(names(partition.df))
centromeres.df <- read_tsv(opt$centromeres, col_types = cols(chrom = "c", .default = "d"), show_col_types = FALSE) %>%
  mutate(chrom = style_chrom(chrom))

if (!all(c("CHROM", "START", "END") %in% names(partition.df))) {
  stop("block file needs columns CHROM, START, END")
}
if (!"SEGDUP_FRAC" %in% names(partition.df)) partition.df$SEGDUP_FRAC <- NA_character_

##################
# Build the table #
##################

# Convert START and END to numbers; text that is not a number is an error.
as_coordinate <- function(values, what) {
  numbers <- suppressWarnings(as.numeric(values))
  if (anyNA(numbers)) stop(what, " has missing or non-numeric values in the block file")
  numbers
}

all_blocks.df <- partition.df %>%
  transmute(chrom = style_chrom(CHROM),
            start = as_coordinate(START, "START"),
            end = as_coordinate(END, "END"),
            segdup_frac = as.numeric(SEGDUP_FRAC))

# Keep the standard chromosomes, say what was dropped, and refuse to go on
# with an empty partition (usually a chromosome naming problem).
report_dropped_contigs(all_blocks.df$chrom[!all_blocks.df$chrom %in% chrom_order], "block partition")
if (!any(all_blocks.df$chrom %in% chrom_order)) {
  stop("no block is left on chr1-22, X or Y. Chromosome names seen: ",
       paste(head(unique(all_blocks.df$chrom), 5), collapse = ", "),
       ". Check --chrom_style (now '", chrom_style, "').")
}

blocks.df <- all_blocks.df %>%
  filter(chrom %in% chrom_order) %>%
  mutate(start = as.integer(start), end = as.integer(end)) %>%
  arrange(factor(chrom, levels = chrom_order), start) %>%
  mutate(block_id = make_block_id(chrom, start, end),
         block_index = row_number(),
         length = end - start,
         mid = (start + end) / 2) %>%
  bind_cols(assign_arm(.$chrom, .$start, .$end, centromeres.df)) %>%
  mutate(genome_build = opt$assembly) %>%
  select(block_index, block_id, chrom, start, end, length, mid, arm,
         spans_centromere, segdup_frac, genome_build)

# The haploblocks.org block table shares the partition's coordinates exactly,
# so the join is on chrom/start/end. Blocks without a match get NA hb_* columns.
if (!is.null(opt$haploblocks_org)) {
  hb.df <- read_tsv(opt$haploblocks_org, show_col_types = FALSE,
                    col_types = cols(chrom = "c", start = "i", end = "i", hb_block_id = "c", .default = "d")) %>%
    mutate(chrom = style_chrom(chrom))
  blocks.df <- blocks.df %>%
    left_join(hb.df, by = c("chrom", "start", "end")) %>%
    mutate(in_haploblocks_org = !is.na(hb_block_id))
  message("haploblocks.org: ", sum(blocks.df$in_haploblocks_org), " of ", nrow(blocks.df),
          " blocks match exactly; ", nrow(hb.df) - sum(blocks.df$in_haploblocks_org),
          " haploblocks.org blocks are not in the partition")
}

# Sanity checks: unique ids, positive lengths, no overlaps within a chromosome.
if (anyDuplicated(blocks.df$block_id)) stop("duplicated blocks in the partition")
if (any(blocks.df$length <= 0)) stop("blocks with END <= START in the partition")
overlaps <- blocks.df %>% group_by(chrom) %>% filter(start < lag(end, default = 0L)) %>% nrow()
if (overlaps > 0) stop(overlaps, " blocks overlap the previous block")

###############
# Save output #
###############

write_tsv(blocks.df, opt$out, na = "")
message("blocks: ", nrow(blocks.df), " on ", n_distinct(blocks.df$chrom), " chromosomes (", opt$assembly, "); ",
        sum(blocks.df$spans_centromere), " span a centromere band")
